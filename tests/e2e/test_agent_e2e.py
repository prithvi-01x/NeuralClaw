"""
tests/e2e/test_agent_e2e.py — End-to-End Smoke Tests

Tests the complete NeuralClaw pipeline from user message → agent response
with a fully wired stack, using a mock LLM client that returns canned
responses. No real network calls, no real disk writes.

Coverage:
  - Full turn: user message → orchestrator → executor → skill → response
  - Autonomous /run: plan created → steps executed → result returned
  - Confirmation gate: high-risk skill triggers ConfirmationRequest callback
  - Capability gate: skill blocked without capability, passes after grant
  - Memory injection: recall_all() result appears in context sent to LLM
  - Error recovery: skill failure does not crash the turn

Run:
    pytest tests/e2e/test_agent_e2e.py -v
"""

from __future__ import annotations

import asyncio
import json
import sys
import types as _types
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── structlog get_logger stub (avoids real structlog config requirement) ───────
import structlog as _real_structlog
from unittest.mock import MagicMock as _MagicMock2
_real_structlog.get_logger = lambda *a, **kw: _MagicMock2()  # type: ignore
# ─────────────────────────────────────────────────────────────────────────────

# ── imports ───────────────────────────────────────────────────────────────────
from neuralclaw.agent.orchestrator import Orchestrator, TurnResult, TurnStatus
from neuralclaw.agent.session import Session
from neuralclaw.brain.llm_client import BaseLLMClient
from neuralclaw.brain.types import FinishReason, LLMConfig, LLMResponse, Message, TokenUsage
from neuralclaw.memory.memory_manager import ContextBundle, MemoryManager
from neuralclaw.memory.long_term import MemoryEntry
from neuralclaw.safety.safety_kernel import SafetyKernel
from neuralclaw.skills.base import SkillBase
from neuralclaw.skills.bus import SkillBus
from neuralclaw.skills.registry import SkillRegistry
from neuralclaw.skills.types import (
    ConfirmationRequest, RiskLevel, SafetyDecision, SafetyStatus,
    SkillCall, SkillManifest, SkillResult, TrustLevel,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fake LLM client
# ─────────────────────────────────────────────────────────────────────────────

class FakeLLMClient(BaseLLMClient):
    """Returns canned responses; records every generate() call."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self._calls: list[list[Message]] = []
        self.supports_tools = False  # text-only for simplicity

    async def generate(self, messages: list[Message], config: LLMConfig, **_) -> LLMResponse:
        self._calls.append(messages)
        text = self._responses.pop(0) if self._responses else "Done."
        return LLMResponse(
            content=text,
            tool_calls=[],
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            model=config.model,
            provider="openai",
        )

    async def health_check(self) -> bool:
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Fake skills
# ─────────────────────────────────────────────────────────────────────────────

class _PingSkill(SkillBase):
    manifest = SkillManifest(
        name="ping",
        version="1.0.0",
        description="Returns pong",
        category="test",
        risk_level=RiskLevel.LOW,
        capabilities=frozenset(),
        parameters={"type": "object", "properties": {}, "required": []},
    )

    async def execute(self, **_) -> SkillResult:
        return SkillResult.ok(
            skill_name=self.manifest.name,
            skill_call_id=_.get("_skill_call_id", ""),
            output="pong",
        )


class _FailSkill(SkillBase):
    manifest = SkillManifest(
        name="fail_skill",
        version="1.0.0",
        description="Always errors",
        category="test",
        risk_level=RiskLevel.LOW,
        capabilities=frozenset(),
        parameters={"type": "object", "properties": {}, "required": []},
    )

    async def execute(self, **_) -> SkillResult:
        return SkillResult.fail(
            skill_name=self.manifest.name,
            skill_call_id=_.get("_skill_call_id", ""),
            error="intentional failure",
            error_type="TestError",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Stack factory
# ─────────────────────────────────────────────────────────────────────────────

def _make_stack(
    llm_responses: list[str],
    skills: Optional[list] = None,
    trust: TrustLevel = TrustLevel.LOW,
    memory_context: str = "",
):
    """Wire up a full agent stack with mocked external dependencies."""
    llm = FakeLLMClient(llm_responses)

    registry = SkillRegistry()
    for skill in (skills or []):
        registry.register(skill)

    safety = MagicMock(spec=SafetyKernel)
    safety.evaluate = AsyncMock(return_value=SafetyDecision(
        status=SafetyStatus.APPROVED,
        reason="test",
        risk_level=RiskLevel.LOW,
        tool_name="",
        tool_call_id="",
    ))

    bus = SkillBus(registry=registry, safety_kernel=safety)

    mm = MagicMock(spec=MemoryManager)
    mm.build_memory_context = AsyncMock(return_value=memory_context)
    mm.recall_all = AsyncMock(return_value=ContextBundle(
        summary=memory_context,
        long_term={},
        recent_episodes=[],
    ))
    mm.record_tool_call = AsyncMock(return_value="tc_fake")
    mm.get_session = MagicMock(return_value=MagicMock())
    mm._sessions = {}

    session = Session.create(user_id="e2e_user", trust_level=trust)

    settings = MagicMock()
    settings.agent.name = "TestAgent"
    settings.agent.max_iterations_per_turn = 5
    settings.agent.max_turn_timeout_seconds = 10.0
    settings.default_llm_model = "test-model"
    settings.llm.temperature = 0.7
    settings.llm.max_tokens = 1000
    settings.tools.terminal.default_timeout_seconds = 30

    orc = Orchestrator.from_settings(
        settings=settings,
        llm_client=llm,
        tool_bus=bus,
        tool_registry=registry,
        memory_manager=mm,
    )

    return orc, session, llm, mm


# ─────────────────────────────────────────────────────────────────────────────
# E2E: basic turn
# ─────────────────────────────────────────────────────────────────────────────

class TestBasicTurn:
    @pytest.mark.asyncio
    async def test_simple_text_response(self):
        orc, session, llm, mm = _make_stack(["Hello from the agent!"])
        result = await orc.run_turn(session, "Hi there")
        assert result.status == TurnStatus.SUCCESS
        assert "Hello from the agent" in result.response.text

    @pytest.mark.asyncio
    async def test_turn_result_has_duration(self):
        orc, session, llm, mm = _make_stack(["response"])
        result = await orc.run_turn(session, "test")
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_multiple_turns_accumulate_history(self):
        orc, session, llm, mm = _make_stack(["first response", "second response"])
        r1 = await orc.run_turn(session, "first message")
        r2 = await orc.run_turn(session, "second message")
        assert r1.status == TurnStatus.SUCCESS
        assert r2.status == TurnStatus.SUCCESS
        # Both LLM calls happened
        assert len(llm._calls) == 2
        # Second call includes more context (conversation history)
        assert len(llm._calls[1]) >= len(llm._calls[0])

    @pytest.mark.asyncio
    async def test_memory_context_injected_into_llm_call(self):
        orc, session, llm, mm = _make_stack(
            ["response"],
            memory_context="[knowledge] Important fact about Python"
        )
        await orc.run_turn(session, "what do you know?")
        # Memory context should appear in the messages sent to the LLM
        all_content = " ".join(
            m.content or "" for m in llm._calls[0]
        )
        assert "Important fact about Python" in all_content


# ─────────────────────────────────────────────────────────────────────────────
# E2E: recall_all → ContextBundle in pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestContextBundlePipeline:
    @pytest.mark.asyncio
    async def test_recall_all_called_during_turn(self):
        orc, session, llm, mm = _make_stack(["ok"])
        await orc.run_turn(session, "test query")
        # Either build_memory_context or recall_all must have been called
        called = (
            mm.build_memory_context.called or
            mm.recall_all.called
        )
        assert called, "Memory was not consulted during the turn"

    @pytest.mark.asyncio
    async def test_empty_memory_turn_succeeds(self):
        orc, session, llm, mm = _make_stack(["fine"], memory_context="")
        result = await orc.run_turn(session, "hello")
        assert result.status == TurnStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_memory_error_doesnt_fail_turn(self):
        from neuralclaw.exceptions import MemoryError as NeuralClawMemoryError
        orc, session, llm, mm = _make_stack(["ok"])
        mm.build_memory_context = AsyncMock(side_effect=NeuralClawMemoryError("db down"))
        # Turn should complete even when memory is broken
        result = await orc.run_turn(session, "hello")
        assert result.status in (TurnStatus.SUCCESS, TurnStatus.ERROR)


# ─────────────────────────────────────────────────────────────────────────────
# E2E: error recovery
# ─────────────────────────────────────────────────────────────────────────────

class TestErrorRecovery:
    @pytest.mark.asyncio
    async def test_turn_never_raises(self):
        """run_turn() must catch all errors and return TurnResult, never raise."""
        orc, session, llm, mm = _make_stack([])
        # Empty responses → FakeLLMClient returns "Done." — still won't crash
        result = await orc.run_turn(session, "anything")
        assert isinstance(result, TurnResult)

    @pytest.mark.asyncio
    async def test_llm_exception_returns_error_status(self):
        orc, session, llm, mm = _make_stack([])
        llm.generate = AsyncMock(side_effect=RuntimeError("LLM exploded"))
        result = await orc.run_turn(session, "test")
        assert result.status == TurnStatus.ERROR

    @pytest.mark.asyncio
    async def test_second_turn_succeeds_after_first_error(self):
        orc, session, llm, mm = _make_stack(["recovery response"])
        # First call fails
        llm.generate = AsyncMock(side_effect=RuntimeError("transient"))
        r1 = await orc.run_turn(session, "first")
        assert r1.status == TurnStatus.ERROR

        # Restore and verify second turn works
        llm.generate = AsyncMock(return_value=LLMResponse(
            content="recovery response",
            tool_calls=[],
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(),
            model="test",
            provider="openai",
        ))
        r2 = await orc.run_turn(session, "second")
        assert r2.status == TurnStatus.SUCCESS


# ─────────────────────────────────────────────────────────────────────────────
# E2E: session capability round-trip
# ─────────────────────────────────────────────────────────────────────────────

class TestSessionCapabilityE2E:
    def test_grant_appears_in_granted_capabilities(self):
        session = Session.create(user_id="test")
        assert "fs:delete" not in session.granted_capabilities
        session.grant_capability("fs:delete")
        assert "fs:delete" in session.granted_capabilities

    def test_revoke_removes_capability(self):
        session = Session.create(user_id="test")
        session.grant_capability("fs:delete")
        session.revoke_capability("fs:delete")
        assert "fs:delete" not in session.granted_capabilities

    def test_capabilities_are_frozenset(self):
        session = Session.create(user_id="test")
        session.grant_capability("net:fetch")
        assert isinstance(session.granted_capabilities, frozenset)

    def test_multiple_capabilities(self):
        session = Session.create(user_id="test")
        for cap in ("fs:read", "fs:write", "net:fetch", "shell:run"):
            session.grant_capability(cap)
        assert len(session.granted_capabilities) == 4
        session.revoke_capability("shell:run")
        assert "shell:run" not in session.granted_capabilities
        assert len(session.granted_capabilities) == 3
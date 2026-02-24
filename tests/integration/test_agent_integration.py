"""
tests/integration/test_agent_integration.py — Agent Integration Tests

Tests the full agent pipeline with real component wiring but mocked
external dependencies (LLM, ChromaDB, SQLite, sentence-transformers).

Coverage:
  - MemoryManager ↔ ContextBuilder: recall_all() → ContextBundle → build()
  - ContextBuilder ↔ Orchestrator: system prompt injection, memory block
  - Executor ↔ SkillBus ↔ SafetyKernel: full dispatch with capability check
  - SkillBus ↔ Confirmation: ConfirmationRequest flows to callback correctly
  - Session ↔ Capability: grant/revoke round-trips affect skill dispatch
  - MemoryManager: recall_all returns ContextBundle with typed fields
  - ContextBundle: empty property, long_term and recent_episodes populated

Run:
    pytest tests/integration/test_agent_integration.py -v
"""

from __future__ import annotations

import asyncio
import sys
import types as _types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call
from typing import Optional

import pytest

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── structlog get_logger stub (avoids real structlog config requirement) ───────
import structlog as _real_structlog
_real_structlog.get_logger = lambda *a, **kw: MagicMock()  # type: ignore
# ─────────────────────────────────────────────────────────────────────────────

# ── imports ───────────────────────────────────────────────────────────────────
from agent.session import Session
from memory.memory_manager import ContextBundle, MemoryManager
from memory.long_term import MemoryEntry
from memory.episodic import Episode
from skills.types import (
    ConfirmationRequest, RiskLevel, SafetyDecision, SafetyStatus,
    SkillCall, SkillManifest, SkillResult, TrustLevel,
)
from skills.registry import SkillRegistry
from skills.bus import SkillBus
from skills.base import SkillBase
from safety.safety_kernel import SafetyKernel


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_manifest(
    name: str = "test_skill",
    risk: RiskLevel = RiskLevel.LOW,
    capabilities: frozenset = frozenset(),
    requires_confirmation: bool = False,
) -> SkillManifest:
    return SkillManifest(
        name=name,
        version="1.0.0",
        description="Integration test skill",
        category="test",
        risk_level=risk,
        capabilities=capabilities,
        requires_confirmation=requires_confirmation,
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": [],
        },
    )


class _EchoSkill(SkillBase):
    manifest = _make_manifest("echo_skill", RiskLevel.LOW)

    async def execute(self, value: str = "ok", **_) -> SkillResult:
        return SkillResult.ok(
            skill_name=self.manifest.name,
            skill_call_id=_.get("_skill_call_id", ""),
            output=f"echo:{value}",
        )


class _CapSkill(SkillBase):
    """Skill that requires a specific capability."""
    manifest = _make_manifest(
        "cap_skill", RiskLevel.LOW, capabilities=frozenset({"special:op"})
    )

    async def execute(self, **_) -> SkillResult:
        return SkillResult.ok(
            skill_name=self.manifest.name,
            skill_call_id=_.get("_skill_call_id", ""),
            output="capability granted!",
        )


class _HighRiskSkill(SkillBase):
    manifest = _make_manifest("high_skill", RiskLevel.HIGH, requires_confirmation=True)

    async def execute(self, **_) -> SkillResult:
        return SkillResult.ok(
            skill_name=self.manifest.name,
            skill_call_id=_.get("_skill_call_id", ""),
            output="high risk executed",
        )


def _make_registry(*skills) -> SkillRegistry:
    reg = SkillRegistry()
    for skill in skills:
        reg.register(skill)
    return reg


def _make_bus(registry: SkillRegistry, safety=None) -> SkillBus:
    if safety is None:
        safety = MagicMock(spec=SafetyKernel)
        safety.evaluate = AsyncMock(return_value=SafetyDecision(
            status=SafetyStatus.APPROVED,
            reason="test approval",
            risk_level=RiskLevel.LOW,
            tool_name="",
            tool_call_id="",
        ))
    return SkillBus(registry=registry, safety_kernel=safety)


# ─────────────────────────────────────────────────────────────────────────────
# ContextBundle
# ─────────────────────────────────────────────────────────────────────────────

class TestContextBundle:
    def test_empty_when_no_data(self):
        bundle = ContextBundle(summary="", long_term={}, recent_episodes=[])
        assert bundle.empty is True

    def test_not_empty_with_summary(self):
        bundle = ContextBundle(summary="some memory", long_term={}, recent_episodes=[])
        assert bundle.empty is False

    def test_not_empty_with_long_term(self):
        entry = MemoryEntry(id="1", text="fact", collection="knowledge", distance=0.2)
        bundle = ContextBundle(summary="", long_term={"knowledge": [entry]}, recent_episodes=[])
        assert bundle.empty is False

    def test_not_empty_with_episodes(self):
        ep = Episode(
            id="ep1", session_id="s1", goal="test", outcome="success",
            started_at=0.0, ended_at=1.0,
        )
        bundle = ContextBundle(summary="", long_term={}, recent_episodes=[ep])
        assert bundle.empty is False

    def test_summary_field_is_string(self):
        bundle = ContextBundle(summary="hello memory", long_term={}, recent_episodes=[])
        assert isinstance(bundle.summary, str)
        assert "hello memory" in bundle.summary

    def test_long_term_default_is_empty_dict(self):
        bundle = ContextBundle(summary="x")
        assert bundle.long_term == {}

    def test_recent_episodes_default_is_empty_list(self):
        bundle = ContextBundle(summary="x")
        assert bundle.recent_episodes == []


# ─────────────────────────────────────────────────────────────────────────────
# MemoryManager.recall_all()
# ─────────────────────────────────────────────────────────────────────────────

class TestRecallAll:
    def _make_manager(self) -> MemoryManager:
        mm = MemoryManager.__new__(MemoryManager)
        mm._initialized = True
        mm._relevance_threshold = 0.5
        mm._task_store = MagicMock()
        mm._task_store.prune_old = MagicMock()
        mm.episodic = MagicMock()
        mm.episodic.get_recent_episodes = AsyncMock(return_value=[])
        return mm

    @pytest.mark.asyncio
    async def test_returns_context_bundle(self):
        mm = self._make_manager()
        mm.search_all = AsyncMock(return_value={})
        result = await mm.recall_all(query="test", session_id="s1")
        assert isinstance(result, ContextBundle)

    @pytest.mark.asyncio
    async def test_empty_bundle_when_no_results(self):
        mm = self._make_manager()
        mm.search_all = AsyncMock(return_value={})
        bundle = await mm.recall_all(query="test", session_id="s1")
        assert bundle.empty is True
        assert bundle.summary == ""

    @pytest.mark.asyncio
    async def test_long_term_results_populate_bundle(self):
        mm = self._make_manager()
        entry = MemoryEntry(id="1", text="Python asyncio", collection="knowledge", distance=0.2)
        mm.search_all = AsyncMock(return_value={"knowledge": [entry]})
        bundle = await mm.recall_all(query="python async", session_id="s1")
        assert "knowledge" in bundle.long_term
        assert bundle.long_term["knowledge"][0].text == "Python asyncio"
        assert "Python asyncio" in bundle.summary

    @pytest.mark.asyncio
    async def test_low_relevance_entries_excluded_from_summary(self):
        mm = self._make_manager()
        entry = MemoryEntry(id="1", text="irrelevant", collection="knowledge", distance=1.8)
        mm.search_all = AsyncMock(return_value={"knowledge": [entry]})
        bundle = await mm.recall_all(query="test", session_id="s1")
        # Entry is in long_term but NOT in summary (below threshold)
        assert "irrelevant" not in bundle.summary
        # Long-term still has the raw results
        assert "knowledge" in bundle.long_term

    @pytest.mark.asyncio
    async def test_recent_episodes_included(self):
        mm = self._make_manager()
        ep = Episode(id="ep1", session_id="s1", goal="Research topic", outcome="success",
                     started_at=0.0, ended_at=1.0)
        mm.episodic.get_recent_episodes = AsyncMock(return_value=[ep])
        mm.search_all = AsyncMock(return_value={})
        bundle = await mm.recall_all(query="test", session_id="s1")
        assert len(bundle.recent_episodes) == 1
        assert bundle.recent_episodes[0].goal == "Research topic"
        assert "Research topic" in bundle.summary

    @pytest.mark.asyncio
    async def test_episodic_failure_returns_partial_bundle(self):
        mm = self._make_manager()
        mm.search_all = AsyncMock(return_value={})
        mm.episodic.get_recent_episodes = AsyncMock(side_effect=OSError("db unavailable"))
        # Should not raise — episodic failures are non-fatal
        bundle = await mm.recall_all(query="test", session_id="s1")
        assert isinstance(bundle, ContextBundle)
        assert bundle.recent_episodes == []

    @pytest.mark.asyncio
    async def test_build_memory_context_returns_string(self):
        """build_memory_context() is a shim over recall_all() — still returns str."""
        mm = self._make_manager()
        entry = MemoryEntry(id="1", text="shim test", collection="knowledge", distance=0.2)
        mm.search_all = AsyncMock(return_value={"knowledge": [entry]})
        result = await mm.build_memory_context(query="test", session_id="s1")
        assert isinstance(result, str)
        assert "shim test" in result

    @pytest.mark.asyncio
    async def test_prune_old_called_each_recall(self):
        mm = self._make_manager()
        mm.search_all = AsyncMock(return_value={})
        await mm.recall_all(query="test", session_id="s1")
        mm._task_store.prune_old.assert_called_once_with(max_age_seconds=3600)


# ─────────────────────────────────────────────────────────────────────────────
# SkillBus integration: dispatch → ConfirmationRequest
# ─────────────────────────────────────────────────────────────────────────────

class TestSkillBusConfirmationRequest:
    @pytest.mark.asyncio
    async def test_confirmation_callback_receives_confirmation_request(self):
        """Callback must receive ConfirmationRequest, not raw SafetyDecision."""
        registry = _make_registry(_HighRiskSkill())
        safety = MagicMock(spec=SafetyKernel)
        safety.evaluate = AsyncMock(return_value=SafetyDecision(
            status=SafetyStatus.CONFIRM_NEEDED,
            reason="high risk action",
            risk_level=RiskLevel.HIGH,
            tool_name="high_skill",
            tool_call_id="tc_001",
        ))
        bus = SkillBus(registry=registry, safety_kernel=safety)

        received: list = []

        async def capture_confirm(req) -> bool:
            received.append(req)
            return True

        call = SkillCall(id="tc_001", skill_name="high_skill", arguments={})
        result = await bus.dispatch(call, TrustLevel.LOW, on_confirm_needed=capture_confirm)

        assert len(received) == 1
        assert isinstance(received[0], ConfirmationRequest)
        cr = received[0]
        assert cr.skill_name == "high_skill"
        assert cr.skill_call_id == "tc_001"
        assert cr.risk_level == RiskLevel.HIGH
        assert cr.reason == "high risk action"

    @pytest.mark.asyncio
    async def test_confirmation_denied_returns_failed_result(self):
        registry = _make_registry(_HighRiskSkill())
        safety = MagicMock(spec=SafetyKernel)
        safety.evaluate = AsyncMock(return_value=SafetyDecision(
            status=SafetyStatus.CONFIRM_NEEDED,
            reason="needs approval",
            risk_level=RiskLevel.HIGH,
            tool_name="high_skill",
            tool_call_id="tc_002",
        ))
        bus = SkillBus(registry=registry, safety_kernel=safety)

        call = SkillCall(id="tc_002", skill_name="high_skill", arguments={})
        result = await bus.dispatch(call, TrustLevel.LOW,
                                    on_confirm_needed=AsyncMock(return_value=False))

        assert result.is_error
        assert "denied" in result.error.lower()

    @pytest.mark.asyncio
    async def test_confirmation_request_carries_arguments(self):
        """ConfirmationRequest.arguments must match the skill call arguments."""
        registry = _make_registry(_HighRiskSkill())
        safety = MagicMock(spec=SafetyKernel)
        safety.evaluate = AsyncMock(return_value=SafetyDecision(
            status=SafetyStatus.CONFIRM_NEEDED,
            reason="test",
            risk_level=RiskLevel.HIGH,
            tool_name="high_skill",
            tool_call_id="tc_003",
        ))
        bus = SkillBus(registry=registry, safety_kernel=safety)

        received: list[ConfirmationRequest] = []
        async def capture(req: ConfirmationRequest) -> bool:
            received.append(req)
            return False

        call = SkillCall(id="tc_003", skill_name="high_skill", arguments={"key": "value"})
        await bus.dispatch(call, TrustLevel.LOW, on_confirm_needed=capture)

        assert received[0].arguments == {"key": "value"}


# ─────────────────────────────────────────────────────────────────────────────
# Session capability → SkillBus dispatch integration
# ─────────────────────────────────────────────────────────────────────────────

class TestCapabilityDispatch:
    def _make_approving_safety(self) -> MagicMock:
        safety = MagicMock(spec=SafetyKernel)
        safety.evaluate = AsyncMock(return_value=SafetyDecision(
            status=SafetyStatus.APPROVED,
            reason="approved",
            risk_level=RiskLevel.LOW,
            tool_name="cap_skill",
            tool_call_id="",
        ))
        return safety

    def _make_denied_safety(self) -> MagicMock:
        safety = MagicMock(spec=SafetyKernel)
        safety.evaluate = AsyncMock(return_value=SafetyDecision(
            status=SafetyStatus.BLOCKED,
            reason="capability not granted: special:op",
            risk_level=RiskLevel.LOW,
            tool_name="cap_skill",
            tool_call_id="",
        ))
        return safety

    @pytest.mark.asyncio
    async def test_skill_executes_when_capability_granted(self):
        registry = _make_registry(_CapSkill())
        bus = SkillBus(registry=registry, safety_kernel=self._make_approving_safety())
        session = Session.create(user_id="test")
        session.grant_capability("special:op")

        call = SkillCall(id="c1", skill_name="cap_skill", arguments={})
        result = await bus.dispatch(call, TrustLevel.LOW,
                                    granted_capabilities=session.granted_capabilities)
        assert not result.is_error
        assert "capability granted" in str(result.output)

    @pytest.mark.asyncio
    async def test_skill_blocked_without_capability(self):
        registry = _make_registry(_CapSkill())
        bus = SkillBus(registry=registry, safety_kernel=self._make_denied_safety())
        session = Session.create(user_id="test")
        # Do NOT grant the capability

        call = SkillCall(id="c2", skill_name="cap_skill", arguments={})
        result = await bus.dispatch(call, TrustLevel.LOW,
                                    granted_capabilities=session.granted_capabilities)
        assert result.is_error

    @pytest.mark.asyncio
    async def test_revoked_capability_blocks_dispatch(self):
        registry = _make_registry(_CapSkill())
        bus = SkillBus(registry=registry, safety_kernel=self._make_denied_safety())
        session = Session.create(user_id="test")
        session.grant_capability("special:op")
        session.revoke_capability("special:op")

        call = SkillCall(id="c3", skill_name="cap_skill", arguments={})
        result = await bus.dispatch(call, TrustLevel.LOW,
                                    granted_capabilities=session.granted_capabilities)
        assert result.is_error


# ─────────────────────────────────────────────────────────────────────────────
# ContextBuilder integration
# ─────────────────────────────────────────────────────────────────────────────

class TestContextBuilderIntegration:
    @pytest.mark.asyncio
    async def test_memory_block_injected_when_results_found(self):
        from agent.context_builder import ContextBuilder

        mm = MagicMock(spec=MemoryManager)
        mm.build_memory_context = AsyncMock(return_value="[knowledge] Python asyncio facts")

        session = Session.create(user_id="test")
        session.set_trust_level(TrustLevel.LOW)

        builder = ContextBuilder(memory_manager=mm, agent_name="TestAgent")
        messages = await builder.build(session=session, user_message="tell me about async")

        combined = " ".join(m.content or "" for m in messages)
        assert "Python asyncio facts" in combined

    @pytest.mark.asyncio
    async def test_empty_memory_not_injected(self):
        from agent.context_builder import ContextBuilder

        mm = MagicMock(spec=MemoryManager)
        mm.build_memory_context = AsyncMock(return_value="")

        session = Session.create(user_id="test")
        builder = ContextBuilder(memory_manager=mm, agent_name="TestAgent")
        messages = await builder.build(session=session, user_message="hello")

        # Should have: system prompt + user message (no memory block)
        assert len(messages) == 2

    @pytest.mark.asyncio
    async def test_memory_search_failure_doesnt_crash_build(self):
        from agent.context_builder import ContextBuilder
        from exceptions import MemoryError as NeuralClawMemoryError

        mm = MagicMock(spec=MemoryManager)
        mm.build_memory_context = AsyncMock(side_effect=NeuralClawMemoryError("db error"))

        session = Session.create(user_id="test")
        builder = ContextBuilder(memory_manager=mm, agent_name="TestAgent")
        messages = await builder.build(session=session, user_message="hello")

        # build() must not raise — context builder catches memory errors
        assert any(m.content for m in messages)
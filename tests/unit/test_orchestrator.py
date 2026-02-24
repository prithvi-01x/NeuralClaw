"""
tests/unit/test_orchestrator.py — Phase 3: Orchestrator Unit Tests

Tests every termination condition of the agent loop with a mock LLM,
plus Executor and Reflector in isolation.

Test groups:
  - TurnStatus / TurnResult dataclass
  - _agent_loop termination: SUCCESS, ITER_LIMIT, CONTEXT_LIMIT, ERROR,
    BLOCKED, tool-error fallback (chat-only mode)
  - run_turn timeout → TIMEOUT
  - run_turn never raises (unhandled exception → ERROR result)
  - Executor: dispatch, reasoner pre-check, confirmation gate
  - Reflector: commit with lesson, no steps (no-op), error resilience
"""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── path setup ───────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── mock heavy deps before importing anything that needs them ─────────────────
import types as _types
for _mod in ["aiosqlite", "chromadb", "sentence_transformers", "structlog"]:
    if _mod not in sys.modules:
        sys.modules[_mod] = _types.ModuleType(_mod)

# structlog needs a get_logger stub
_structlog = sys.modules["structlog"]
_structlog.get_logger = lambda *a, **kw: MagicMock()  # type: ignore
_structlog.contextvars = _types.ModuleType("structlog.contextvars")
_structlog.contextvars.bind_contextvars = lambda **kw: None  # type: ignore
_structlog.contextvars.unbind_contextvars = lambda *a: None  # type: ignore
_structlog.contextvars.clear_contextvars = lambda: None  # type: ignore

from agent.orchestrator import TurnResult, TurnStatus
from agent.executor import Executor
from agent.reflector import Reflector
from skills.registry import SkillRegistry
from skills.types import RiskLevel, SkillManifest, TrustLevel


# ─────────────────────────────────────────────────────────────────────────────
# Shared test helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_manifest(name="test_skill", risk=RiskLevel.LOW) -> SkillManifest:
    return SkillManifest(
        name=name,
        version="1.0.0",
        description="Test skill",
        category="test",
        risk_level=risk,
        capabilities=frozenset(),
        parameters={"type": "object", "properties": {}, "required": []},
    )


def _make_llm_response(content="Hello!", has_tool_calls=False, tool_calls=None):
    """Build a minimal fake LLMResponse."""
    resp = MagicMock()
    resp.content = content
    resp.has_tool_calls = has_tool_calls
    resp.tool_calls = tool_calls or []
    resp.is_complete = True
    resp.finish_reason = MagicMock()
    resp.finish_reason.value = "stop"
    resp.model = "test-model"
    resp.usage = MagicMock()
    resp.usage.input_tokens = 10
    resp.usage.output_tokens = 5
    return resp


def _make_tool_call(name="web_search", arguments=None, tc_id="tc-1"):
    from brain.types import ToolCall as BrainToolCall
    return BrainToolCall(
        id=tc_id,
        name=name,
        arguments=arguments or {"query": "test"},
    )


def _make_tool_result(name="web_search", content="result", is_error=False):
    result = MagicMock()
    result.content = content
    result.is_error = is_error
    result.name = name
    result.risk_level = MagicMock()
    result.risk_level.value = "LOW"
    result.duration_ms = 42.0
    return result


def _mock_session(trust="low"):
    session = MagicMock()
    session.id = "sess-test"
    session.user_id = "user-test"
    session.trust_level = TrustLevel(trust)
    session.active_plan = None
    session.tool_call_count = 0
    session.turn_count = 0
    session.granted_capabilities = frozenset()
    session.is_cancelled = MagicMock(return_value=False)
    session.reset_cancel = MagicMock()
    session.add_user_message = MagicMock()
    session.add_assistant_message = MagicMock()
    session.record_token_usage = MagicMock()
    session.record_tool_call = MagicMock()
    session._pending_confirmations = {}
    return session


async def _noop_build(**kwargs):
    """Stub for ContextBuilder.build()"""
    return []


def _make_orchestrator(llm_generate_fn, max_iterations=3, max_turn_timeout=30):
    """
    Build a minimal Orchestrator with mocked dependencies.
    llm_generate_fn: async callable that replaces llm.generate().
    """
    from agent.orchestrator import Orchestrator
    from brain.types import LLMConfig

    llm = MagicMock()
    llm.generate = AsyncMock(side_effect=llm_generate_fn)
    llm.supports_tools = True

    registry = MagicMock()
    registry.list_schemas = MagicMock(return_value=[])
    registry.list_names = MagicMock(return_value=[])
    registry.get_schema = MagicMock(return_value=None)

    bus = MagicMock()
    memory = MagicMock()
    memory.build_memory_context = AsyncMock(return_value="")
    memory.record_tool_call = AsyncMock()
    memory.store = AsyncMock()

    reasoner = MagicMock()
    reasoner.evaluate_tool_call = AsyncMock(return_value=MagicMock(proceed=True))
    reasoner.reflect = AsyncMock(return_value="lesson learned")

    orc = Orchestrator(
        llm_client=llm,
        llm_config=LLMConfig(model="test-model", temperature=0.0, max_tokens=100),
        tool_bus=bus,
        tool_registry=registry,
        memory_manager=memory,
        max_iterations=max_iterations,
        max_turn_timeout=max_turn_timeout,
    )

    # Patch ContextBuilder and capability checks
    orc._ctx.build = AsyncMock(side_effect=_noop_build)
    orc._reasoner = reasoner
    orc._executor = MagicMock()

    # Stub out capabilities so tools aren't suppressed
    with patch("agent.orchestrator.get_capabilities",
               return_value=MagicMock(supports_tools=True)):
        pass  # patch applied per-test below

    return orc, llm, registry


# ─────────────────────────────────────────────────────────────────────────────
# TurnStatus / TurnResult
# ─────────────────────────────────────────────────────────────────────────────

class TestTurnResult:
    def test_all_statuses_exist(self):
        expected = {"success", "iter_limit", "timeout", "blocked", "context_limit", "error"}
        actual = {s.value for s in TurnStatus}
        assert expected == actual

    def test_turn_result_frozen(self):
        from agent.response_synthesizer import AgentResponse, ResponseKind
        r = TurnResult(
            status=TurnStatus.SUCCESS,
            response=AgentResponse(kind=ResponseKind.TEXT, text="hi"),
        )
        with pytest.raises((AttributeError, TypeError)):
            r.status = TurnStatus.ERROR  # type: ignore

    def test_succeeded_property(self):
        from agent.response_synthesizer import AgentResponse, ResponseKind
        ok = TurnResult(
            status=TurnStatus.SUCCESS,
            response=AgentResponse(kind=ResponseKind.TEXT, text="ok"),
        )
        err = TurnResult(
            status=TurnStatus.ERROR,
            response=AgentResponse(kind=ResponseKind.ERROR, text="err"),
        )
        assert ok.succeeded is True
        assert err.succeeded is False

    def test_default_fields(self):
        from agent.response_synthesizer import AgentResponse, ResponseKind
        r = TurnResult(
            status=TurnStatus.ITER_LIMIT,
            response=AgentResponse(kind=ResponseKind.ERROR, text="x"),
        )
        assert r.steps_taken == 0
        assert r.duration_ms == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# _agent_loop termination conditions
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAgentLoopTermination:

    @patch("brain.capabilities.get_capabilities",
           return_value=MagicMock(supports_tools=True))
    async def test_success_no_tool_calls(self, _caps):
        """LLM returns text immediately → SUCCESS."""
        orc, llm, _ = _make_orchestrator(
            lambda **kw: _make_llm_response(content="Done!", has_tool_calls=False)
        )
        session = _mock_session()
        result = await orc._agent_loop(session, "hello")
        assert result.status == TurnStatus.SUCCESS
        assert "Done!" in result.response.text

    @patch("brain.capabilities.get_capabilities",
           return_value=MagicMock(supports_tools=True))
    async def test_iter_limit(self, _caps):
        """LLM always returns tool calls → ITER_LIMIT after max_iterations."""
        def _always_tool(**kw):
            tc = _make_tool_call()
            return _make_llm_response(has_tool_calls=True, tool_calls=[tc])

        orc, llm, registry = _make_orchestrator(_always_tool, max_iterations=2)
        registry.get_schema.return_value = _make_manifest()

        # executor returns a non-blocked result
        legacy_result = _make_tool_result()
        orc._executor.dispatch = AsyncMock(return_value=legacy_result)

        session = _mock_session()
        result = await orc._agent_loop(session, "loop forever")
        assert result.status == TurnStatus.ITER_LIMIT

    @patch("brain.capabilities.get_capabilities",
           return_value=MagicMock(supports_tools=True))
    async def test_context_limit(self, _caps):
        """LLMContextError → CONTEXT_LIMIT status."""
        from brain.llm_client import LLMContextError

        async def _raise_context(**kw):
            raise LLMContextError("Input too long")

        orc, _, _ = _make_orchestrator(_raise_context)
        session = _mock_session()
        result = await orc._agent_loop(session, "giant prompt")
        assert result.status == TurnStatus.CONTEXT_LIMIT
        assert "context" in result.response.text.lower() or "limit" in result.response.text.lower()

    @patch("brain.capabilities.get_capabilities",
           return_value=MagicMock(supports_tools=True))
    async def test_error_on_unknown_exception(self, _caps):
        """Unhandled LLM exception → ERROR status."""
        async def _blow_up(**kw):
            raise RuntimeError("Something unexpected")

        orc, _, _ = _make_orchestrator(_blow_up)
        session = _mock_session()
        result = await orc._agent_loop(session, "boom")
        assert result.status == TurnStatus.ERROR

    @patch("brain.capabilities.get_capabilities",
           return_value=MagicMock(supports_tools=True))
    async def test_blocked_on_critical_step(self, _caps):
        """Safety-blocked HIGH-risk tool call → BLOCKED status."""
        def _tool_call_resp(**kw):
            tc = _make_tool_call(name="terminal_exec")
            return _make_llm_response(has_tool_calls=True, tool_calls=[tc])

        orc, llm, registry = _make_orchestrator(_tool_call_resp, max_iterations=3)

        # Return HIGH risk manifest
        registry.get_schema.return_value = _make_manifest("terminal_exec", RiskLevel.HIGH)

        # Safety kernel blocks the call
        blocked_result = _make_tool_result(is_error=True, content="blocked by safety kernel")
        orc._executor.dispatch = AsyncMock(return_value=blocked_result)

        session = _mock_session()
        result = await orc._agent_loop(session, "do risky thing")
        assert result.status == TurnStatus.BLOCKED

    @patch("brain.capabilities.get_capabilities",
           return_value=MagicMock(supports_tools=True))
    async def test_tool_error_fallback_to_chat_only(self, _caps):
        """
        LLMInvalidRequestError with tool-related message → fallback to chat-only
        and succeed on retry.
        """
        from brain.llm_client import LLMInvalidRequestError

        call_count = {"n": 0}

        async def _tool_error_then_ok(**kw):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise LLMInvalidRequestError("does not support tools")
            return _make_llm_response(content="Chat response", has_tool_calls=False)

        orc, _, registry = _make_orchestrator(_tool_error_then_ok)
        registry.list_schemas.return_value = [_make_manifest()]
        session = _mock_session()
        result = await orc._agent_loop(session, "hi")
        assert result.status == TurnStatus.SUCCESS
        assert call_count["n"] == 2  # retried without tools

    @patch("brain.capabilities.get_capabilities",
           return_value=MagicMock(supports_tools=True))
    async def test_cancelled_session(self, _caps):
        """Cancelled session before first LLM call → ERROR with cancelled response."""
        async def _never_called(**kw):
            raise AssertionError("LLM should not be called")

        orc, _, _ = _make_orchestrator(_never_called)
        session = _mock_session()
        session.is_cancelled.return_value = True
        result = await orc._agent_loop(session, "hi")
        assert result.status == TurnStatus.ERROR


# ─────────────────────────────────────────────────────────────────────────────
# run_turn — always returns TurnResult, never raises
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestRunTurn:

    @patch("brain.capabilities.get_capabilities",
           return_value=MagicMock(supports_tools=True))
    @patch("agent.orchestrator.bind_session")
    @patch("agent.orchestrator.clear_session")
    async def test_returns_turn_result(self, _clr, _bind, _caps):
        orc, _, _ = _make_orchestrator(
            lambda **kw: _make_llm_response(content="Hi!")
        )
        session = _mock_session()
        result = await orc.run_turn(session, "hello")
        assert isinstance(result, TurnResult)
        assert result.status == TurnStatus.SUCCESS

    @patch("brain.capabilities.get_capabilities",
           return_value=MagicMock(supports_tools=True))
    @patch("agent.orchestrator.bind_session")
    @patch("agent.orchestrator.clear_session")
    async def test_timeout_returns_turn_result(self, _clr, _bind, _caps):
        """Turn timeout → TurnResult with TIMEOUT status, never raises."""
        async def _hang(**kw):
            await asyncio.sleep(999)

        orc, _, _ = _make_orchestrator(_hang, max_turn_timeout=0)
        session = _mock_session()
        result = await orc.run_turn(session, "slow task")
        assert isinstance(result, TurnResult)
        assert result.status == TurnStatus.TIMEOUT

    @patch("agent.orchestrator.bind_session")
    @patch("agent.orchestrator.clear_session")
    async def test_never_raises_on_crash(self, _clr, _bind):
        """Even a crash inside _agent_loop must produce an ERROR TurnResult."""
        orc, _, _ = _make_orchestrator(
            lambda **kw: _make_llm_response()
        )
        session = _mock_session()

        # Force a crash inside the loop
        orc._agent_loop = AsyncMock(side_effect=RuntimeError("Kernel panic"))
        result = await orc.run_turn(session, "crash please")
        assert isinstance(result, TurnResult)
        assert result.status == TurnStatus.ERROR

    @patch("agent.orchestrator.bind_session")
    @patch("agent.orchestrator.clear_session")
    async def test_duration_ms_populated(self, _clr, _bind):
        orc, _, _ = _make_orchestrator(
            lambda **kw: _make_llm_response()
        )
        orc._agent_loop = AsyncMock(return_value=TurnResult(
            status=TurnStatus.SUCCESS,
            response=MagicMock(text="done"),
        ))
        session = _mock_session()
        result = await orc.run_turn(session, "time me")
        assert isinstance(result.duration_ms, float)


# ─────────────────────────────────────────────────────────────────────────────
# Executor
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestExecutor:

    def _make_executor(self, dispatch_result=None, reasoner_proceed=True):
        registry = MagicMock()
        bus = MagicMock()
        memory = MagicMock()
        memory.record_tool_call = AsyncMock()

        reasoner = MagicMock()
        reasoner.evaluate_tool_call = AsyncMock(
            return_value=MagicMock(proceed=reasoner_proceed, concern="concern", reasoning="reason")
        )

        if dispatch_result is None:
            dispatch_result = _make_tool_result()
        bus.dispatch = AsyncMock(return_value=dispatch_result)

        return Executor(registry, bus, reasoner, memory), registry, bus, reasoner

    async def test_dispatch_low_risk_skips_reasoner(self):
        executor, registry, bus, reasoner = self._make_executor()
        registry.get_schema.return_value = _make_manifest("web_search", RiskLevel.LOW)

        btc = _make_tool_call("web_search")
        session = _mock_session()

        result = await executor.dispatch(btc, session)
        reasoner.evaluate_tool_call.assert_not_awaited()
        bus.dispatch.assert_awaited_once()

    async def test_dispatch_high_risk_runs_reasoner(self):
        executor, registry, bus, reasoner = self._make_executor(reasoner_proceed=True)
        registry.get_schema.return_value = _make_manifest("terminal_exec", RiskLevel.HIGH)

        btc = _make_tool_call("terminal_exec", {"command": "ls"})
        session = _mock_session()

        await executor.dispatch(btc, session)
        reasoner.evaluate_tool_call.assert_awaited_once()
        bus.dispatch.assert_awaited_once()

    async def test_reasoner_block_skips_bus(self):
        executor, registry, bus, reasoner = self._make_executor(reasoner_proceed=False)
        registry.get_schema.return_value = _make_manifest("terminal_exec", RiskLevel.HIGH)

        btc = _make_tool_call("terminal_exec")
        session = _mock_session()

        result = await executor.dispatch(btc, session)
        bus.dispatch.assert_not_awaited()
        assert result.is_error

    async def test_records_tool_call_to_memory(self):
        executor, registry, bus, reasoner = self._make_executor()
        registry.get_schema.return_value = _make_manifest("web_search", RiskLevel.LOW)

        btc = _make_tool_call("web_search")
        session = _mock_session()

        # Patch asyncio.create_task so we can verify memory recording
        with patch("agent.executor.asyncio.create_task") as mock_task:
            await executor.dispatch(btc, session)
            mock_task.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# Reflector
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestReflector:

    def _make_reflector(self, lesson="lesson learned"):
        reasoner = MagicMock()
        reasoner.reflect = AsyncMock(return_value=lesson)

        memory = MagicMock()
        memory.add_reflection = AsyncMock()
        memory.commit_episode = AsyncMock()

        return Reflector(reasoner, memory), reasoner, memory

    async def test_commit_no_steps_is_noop(self):
        reflector, reasoner, memory = self._make_reflector()
        session = _mock_session()
        await reflector.commit(session, steps_taken=[], goal="test")
        reasoner.reflect.assert_not_awaited()
        memory.add_reflection.assert_not_awaited()

    async def test_commit_calls_reasoner_and_stores_lesson(self):
        reflector, reasoner, memory = self._make_reflector(lesson="remember X")
        session = _mock_session()
        session.active_plan = None

        await reflector.commit(session, steps_taken=["step 1"], goal="my goal")
        reasoner.reflect.assert_awaited_once()
        memory.add_reflection.assert_awaited_once()

    async def test_commit_commits_episode_when_plan_has_episode(self):
        reflector, reasoner, memory = self._make_reflector()
        session = _mock_session()
        plan = MagicMock()
        plan.episode_id = "ep-123"
        session.active_plan = plan

        await reflector.commit(session, steps_taken=["step 1"], goal="goal")
        memory.commit_episode.assert_awaited_once()
        call_kwargs = memory.commit_episode.call_args[1]
        assert call_kwargs["episode_id"] == "ep-123"

    async def test_commit_survives_reasoner_error(self):
        """Reflector must not raise even if Reasoner.reflect() fails."""
        reflector, reasoner, memory = self._make_reflector()
        reasoner.reflect = AsyncMock(side_effect=RuntimeError("LLM died"))
        session = _mock_session()

        # Should not raise
        await reflector.commit(session, steps_taken=["step 1"], goal="goal")

    async def test_commit_survives_memory_error(self):
        """Reflector must not raise even if memory.add_reflection() fails."""
        reflector, reasoner, memory = self._make_reflector()
        memory.add_reflection = AsyncMock(side_effect=RuntimeError("DB down"))
        session = _mock_session()

        await reflector.commit(session, steps_taken=["step 1"], goal="goal")

    async def test_no_episode_commit_without_plan(self):
        reflector, reasoner, memory = self._make_reflector()
        session = _mock_session()
        session.active_plan = None

        await reflector.commit(session, steps_taken=["step 1"], goal="goal")
        memory.commit_episode.assert_not_awaited()
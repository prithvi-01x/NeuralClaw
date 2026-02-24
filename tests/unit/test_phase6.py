"""
tests/unit/test_phase6.py — Phase 6: Observability & Error Handling Tests

Tasks covered:
  32 — TraceContext dataclass and structlog binding
  33 — trace_id / turn_id propagated through orchestrator turns
  34 — NeuralClawError hierarchy: all subclasses, correct inheritance
  35 — No bare except Exception in core dispatch paths (structural check)
  36 — RetryPolicy: delays, retryable classification, per-skill overrides
  38 — SkillBus retry: success on 2nd attempt, exhausted retries, non-retryable

Run:
    pytest tests/unit/test_phase6.py -v
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Task 32 & 33: TraceContext
# ─────────────────────────────────────────────────────────────────────────────

class TestTraceContext:
    def test_default_trace_id_has_prefix(self):
        from observability.trace import TraceContext
        ctx = TraceContext()
        assert ctx.trace_id.startswith("trc_")

    def test_for_session_derives_from_session_id(self):
        from observability.trace import TraceContext
        ctx = TraceContext.for_session("sess_abcdef12")
        assert ctx.trace_id == "trc_sess_abc"

    def test_for_session_no_id_generates_random(self):
        from observability.trace import TraceContext
        ctx = TraceContext.for_session()
        assert ctx.trace_id.startswith("trc_")

    def test_turn_id_starts_as_none(self):
        from observability.trace import TraceContext
        ctx = TraceContext()
        assert ctx.turn_id is None

    def test_new_turn_sets_turn_id(self):
        from observability.trace import TraceContext
        ctx = TraceContext()
        ctx.new_turn()
        assert ctx.turn_id is not None
        assert ctx.turn_id.startswith("trn_")

    def test_new_turn_returns_self(self):
        from observability.trace import TraceContext
        ctx = TraceContext()
        assert ctx.new_turn() is ctx

    def test_successive_turns_get_unique_ids(self):
        from observability.trace import TraceContext
        ctx = TraceContext()
        ctx.new_turn()
        t1 = ctx.turn_id
        ctx.new_turn()
        t2 = ctx.turn_id
        assert t1 != t2

    def test_clear_turn_removes_turn_id(self):
        from observability.trace import TraceContext
        ctx = TraceContext()
        ctx.new_turn()
        ctx.clear_turn()
        assert ctx.turn_id is None

    def test_as_dict_includes_both_ids_when_turn_active(self):
        from observability.trace import TraceContext
        ctx = TraceContext(trace_id="trc_test1234")
        ctx.new_turn()
        d = ctx.as_dict()
        assert d["trace_id"] == "trc_test1234"
        assert "turn_id" in d and d["turn_id"].startswith("trn_")

    def test_as_dict_omits_turn_id_when_not_set(self):
        from observability.trace import TraceContext
        ctx = TraceContext(trace_id="trc_test1234")
        d = ctx.as_dict()
        assert d == {"trace_id": "trc_test1234"}

    def test_bind_calls_structlog_with_both_ids(self):
        from observability.trace import TraceContext
        ctx = TraceContext(trace_id="trc_abc")
        ctx.new_turn()
        with patch("structlog.contextvars.bind_contextvars") as mock_bind:
            ctx.bind()
            kwargs = mock_bind.call_args[1]
            assert kwargs["trace_id"] == "trc_abc"
            assert "turn_id" in kwargs

    def test_bind_without_turn_omits_turn_id(self):
        from observability.trace import TraceContext
        ctx = TraceContext(trace_id="trc_abc")
        with patch("structlog.contextvars.bind_contextvars") as mock_bind:
            ctx.bind()
            kwargs = mock_bind.call_args[1]
            assert "turn_id" not in kwargs

    def test_clear_calls_unbind(self):
        from observability.trace import TraceContext
        ctx = TraceContext()
        ctx.new_turn()
        with patch("structlog.contextvars.unbind_contextvars"):
            ctx.clear()
            assert ctx.turn_id is None

    def test_repr_contains_ids(self):
        from observability.trace import TraceContext
        ctx = TraceContext(trace_id="trc_testid")
        ctx.new_turn()
        r = repr(ctx)
        assert "trc_testid" in r
        assert "trn_" in r


# ─────────────────────────────────────────────────────────────────────────────
# Task 34: NeuralClawError hierarchy
# ─────────────────────────────────────────────────────────────────────────────

class TestErrorHierarchy:
    def test_all_exceptions_importable(self):
        from exceptions import (
            NeuralClawError,
            AgentError, PlanError, TurnTimeoutError, IterationLimitError,
            SkillError, SkillNotFoundError, SkillTimeoutError,
            SkillValidationError, SkillDisabledError,
            SafetyError, CapabilityDeniedError, CommandNotAllowedError,
            ConfirmationDeniedError,
            MemoryError, MemoryNotInitializedError, MemoryStoreError,
            LLMError, LLMConnectionError, LLMRateLimitError,
            LLMContextError, LLMInvalidRequestError,
        )  # just verify no ImportError

    def test_agent_errors_inherit_neuralclaw(self):
        from exceptions import NeuralClawError, PlanError, TurnTimeoutError, IterationLimitError
        for cls in (PlanError, TurnTimeoutError, IterationLimitError):
            assert issubclass(cls, NeuralClawError), f"{cls} not subclass of NeuralClawError"

    def test_skill_errors_inherit_skill_error(self):
        from exceptions import SkillError, SkillNotFoundError, SkillTimeoutError, SkillValidationError, SkillDisabledError
        for cls in (SkillNotFoundError, SkillTimeoutError, SkillValidationError, SkillDisabledError):
            assert issubclass(cls, SkillError)

    def test_safety_errors_inherit_safety_error(self):
        from exceptions import SafetyError, CapabilityDeniedError, CommandNotAllowedError, ConfirmationDeniedError
        for cls in (CapabilityDeniedError, CommandNotAllowedError, ConfirmationDeniedError):
            assert issubclass(cls, SafetyError)

    def test_memory_errors_inherit_memory_error(self):
        from exceptions import MemoryError, MemoryNotInitializedError, MemoryStoreError
        for cls in (MemoryNotInitializedError, MemoryStoreError):
            assert issubclass(cls, MemoryError)

    def test_llm_errors_re_exported(self):
        from exceptions import LLMError, LLMConnectionError, LLMRateLimitError
        assert issubclass(LLMConnectionError, LLMError)
        assert issubclass(LLMRateLimitError, LLMError)

    def test_capability_denied_carries_fields(self):
        from exceptions import CapabilityDeniedError
        err = CapabilityDeniedError(skill="web_fetch", capability="net:fetch")
        assert err.skill == "web_fetch"
        assert err.capability == "net:fetch"
        assert "net:fetch" in str(err)

    def test_command_not_allowed_carries_command(self):
        from exceptions import CommandNotAllowedError
        err = CommandNotAllowedError(command="rm -rf /")
        assert err.command == "rm -rf /"
        assert "rm -rf /" in str(err)

    def test_skill_errors_catchable_as_neuralclaw(self):
        from exceptions import NeuralClawError, SkillTimeoutError
        with pytest.raises(NeuralClawError):
            raise SkillTimeoutError("timed out")

    def test_custom_message_overrides_default_for_capability_denied(self):
        from exceptions import CapabilityDeniedError
        err = CapabilityDeniedError(skill="s", capability="c", message="custom msg")
        assert str(err) == "custom msg"


# ─────────────────────────────────────────────────────────────────────────────
# Task 35: No bare except Exception in core kernel dispatch paths
# ─────────────────────────────────────────────────────────────────────────────

class TestNoBareExceptInKernel:
    def _bare_except_lines(self, filepath: str) -> list[int]:
        lines = open(filepath).readlines()
        return [i + 1 for i, line in enumerate(lines)
                if line.strip().startswith("except Exception")]

    def test_orchestrator_has_minimal_bare_except(self):
        bad = self._bare_except_lines("agent/orchestrator.py")
        # Only the two _refresh_capabilities silent-pass probes are allowed
        assert len(bad) <= 2, (
            f"agent/orchestrator.py has {len(bad)} bare 'except Exception' at lines {bad}. "
            f"Expected ≤2 (only _refresh_capabilities probes allowed)."
        )

    def test_reflector_has_no_bare_except(self):
        bad = self._bare_except_lines("agent/reflector.py")
        assert bad == [], f"agent/reflector.py has bare 'except Exception' at lines {bad}"

    def test_skills_bus_has_no_bare_except_in_dispatch(self):
        bad = self._bare_except_lines("skills/bus.py")
        # The import-time fallback at module top is the only one allowed
        assert len(bad) <= 1, (
            f"skills/bus.py has {len(bad)} bare 'except Exception' at lines {bad}. "
            f"Expected ≤1 (import-time fallback only)."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Task 36: RetryPolicy
# ─────────────────────────────────────────────────────────────────────────────

class TestRetryPolicy:
    def test_default_max_attempts_is_3(self):
        from skills.bus import RetryPolicy
        assert RetryPolicy().max_attempts == 3

    def test_skill_timeout_is_retryable(self):
        from skills.bus import RetryPolicy
        assert RetryPolicy().is_retryable("SkillTimeoutError") is True

    def test_llm_rate_limit_is_retryable(self):
        from skills.bus import RetryPolicy
        assert RetryPolicy().is_retryable("LLMRateLimitError") is True

    def test_value_error_is_not_retryable(self):
        from skills.bus import RetryPolicy
        assert RetryPolicy().is_retryable("ValueError") is False

    def test_none_is_not_retryable(self):
        from skills.bus import RetryPolicy
        assert RetryPolicy().is_retryable(None) is False

    def test_delay_increases_exponentially(self):
        from skills.bus import RetryPolicy
        p = RetryPolicy(base_delay=1.0, max_delay=30.0, jitter=False)
        assert p.delay_for_attempt(1) < p.delay_for_attempt(2) < p.delay_for_attempt(3)

    def test_delay_capped_at_max(self):
        from skills.bus import RetryPolicy
        p = RetryPolicy(base_delay=10.0, max_delay=15.0, jitter=False)
        assert p.delay_for_attempt(10) <= 15.0

    def test_jitter_stays_within_bounds(self):
        from skills.bus import RetryPolicy
        p = RetryPolicy(base_delay=2.0, max_delay=30.0, jitter=True)
        for _ in range(30):
            d = p.delay_for_attempt(1)
            assert 1.0 <= d <= 2.0  # jitter is [50%, 100%] of base_delay

    def test_terminal_exec_override_is_max_1(self):
        from skills.bus import _SKILL_RETRY_OVERRIDES
        override = _SKILL_RETRY_OVERRIDES.get("terminal_exec")
        assert override is not None
        assert override.max_attempts == 1


# ─────────────────────────────────────────────────────────────────────────────
# Task 38: SkillBus retry behaviour
# ─────────────────────────────────────────────────────────────────────────────

def _make_skill(name="test_skill", enabled=True, timeout=5):
    skill = MagicMock()
    skill.manifest = MagicMock()
    skill.manifest.name = name
    skill.manifest.enabled = enabled
    skill.manifest.risk_level = MagicMock()
    skill.manifest.requires_confirmation = False
    skill.manifest.parameters = {"type": "object", "properties": {}, "required": []}
    skill.manifest.timeout_seconds = timeout
    skill.manifest.description = "test"
    skill.manifest.category = "test"
    skill.validate = AsyncMock()
    return skill


def _make_call(name="test_skill"):
    from skills.types import SkillCall
    return SkillCall(id="call_001", skill_name=name, arguments={})


def _make_registry(name, skill):
    r = MagicMock()
    r.get_or_none.return_value = skill
    r.list_names.return_value = [name]
    return r


class TestSkillBusRetry:
    @pytest.mark.asyncio
    async def test_success_first_attempt_no_retry(self):
        from skills.bus import SkillBus
        from skills.types import SkillResult, TrustLevel
        skill = _make_skill()
        skill.execute = AsyncMock(return_value=SkillResult.ok("test_skill", "call_001", "done"))
        bus = SkillBus(_make_registry("test_skill", skill), safety_kernel=None)
        result = await bus.dispatch(_make_call(), TrustLevel.HIGH)
        assert result.success
        assert skill.execute.call_count == 1

    @pytest.mark.asyncio
    async def test_success_on_second_attempt(self):
        """Timeout on first attempt, succeeds on second."""
        from skills.bus import SkillBus, RetryPolicy, _SKILL_RETRY_OVERRIDES
        from skills.types import SkillResult, TrustLevel
        skill = _make_skill()
        calls = []

        async def execute(**kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise asyncio.TimeoutError()
            return SkillResult.ok("test_skill", "call_001", "ok on retry")

        skill.execute = execute
        fast_policy = RetryPolicy(max_attempts=3, base_delay=0.001, jitter=False)
        with patch.dict(_SKILL_RETRY_OVERRIDES, {"test_skill": fast_policy}):
            bus = SkillBus(_make_registry("test_skill", skill), safety_kernel=None)
            result = await bus.dispatch(_make_call(), TrustLevel.HIGH)

        assert result.success, f"expected success, got: {result.error}"
        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_exhausts_retries_returns_failure(self):
        """Always times out — should exhaust max_attempts and return error."""
        from skills.bus import SkillBus, RetryPolicy, _SKILL_RETRY_OVERRIDES
        from skills.types import TrustLevel
        skill = _make_skill()
        skill.execute = AsyncMock(side_effect=asyncio.TimeoutError())
        fast_policy = RetryPolicy(max_attempts=2, base_delay=0.001, jitter=False)
        with patch.dict(_SKILL_RETRY_OVERRIDES, {"test_skill": fast_policy}):
            bus = SkillBus(_make_registry("test_skill", skill), safety_kernel=None)
            result = await bus.dispatch(_make_call(), TrustLevel.HIGH)

        assert not result.success
        assert result.error_type == "SkillTimeoutError"
        assert skill.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_non_retryable_error_fails_immediately(self):
        """ValueError is not in retryable_errors — no retry, fail on first attempt."""
        from skills.bus import SkillBus, RetryPolicy, _SKILL_RETRY_OVERRIDES
        from skills.types import TrustLevel
        skill = _make_skill()
        skill.execute = AsyncMock(side_effect=ValueError("bad input"))
        fast_policy = RetryPolicy(max_attempts=3, base_delay=0.001, jitter=False)
        with patch.dict(_SKILL_RETRY_OVERRIDES, {"test_skill": fast_policy}):
            bus = SkillBus(_make_registry("test_skill", skill), safety_kernel=None)
            result = await bus.dispatch(_make_call(), TrustLevel.HIGH)

        assert not result.success
        # ValueError is a BaseException (not NeuralClawError) so caught by BaseException branch → no retry
        assert skill.execute.call_count == 1

    @pytest.mark.asyncio
    async def test_terminal_exec_never_retried(self):
        """terminal_exec has max_attempts=1 — must not retry even on timeout."""
        from skills.bus import SkillBus
        from skills.types import TrustLevel
        skill = _make_skill(name="terminal_exec")
        skill.execute = AsyncMock(side_effect=asyncio.TimeoutError())
        bus = SkillBus(_make_registry("terminal_exec", skill), safety_kernel=None)
        result = await bus.dispatch(_make_call("terminal_exec"), TrustLevel.HIGH)
        assert not result.success
        assert skill.execute.call_count == 1  # exactly 1 — no retry

    @pytest.mark.asyncio
    async def test_skill_not_found_returns_error_result(self):
        from skills.bus import SkillBus
        from skills.types import TrustLevel
        r = MagicMock()
        r.get_or_none.return_value = None
        r.list_names.return_value = []
        bus = SkillBus(r, safety_kernel=None)
        result = await bus.dispatch(_make_call(), TrustLevel.LOW)
        assert not result.success
        assert result.error_type == "SkillNotFoundError"

    @pytest.mark.asyncio
    async def test_bus_never_raises_on_unexpected_error(self):
        """dispatch() must always return SkillResult — never raise."""
        from skills.bus import SkillBus
        from skills.types import TrustLevel
        skill = _make_skill()
        skill.execute = AsyncMock(side_effect=RuntimeError("kernel panic"))
        bus = SkillBus(_make_registry("test_skill", skill), safety_kernel=None)
        result = await bus.dispatch(_make_call(), TrustLevel.HIGH)
        assert not result.success
        assert "RuntimeError" in (result.error_type or "")
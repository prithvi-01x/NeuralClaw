"""
agent/executor.py — Skill Executor

Resolves a single BrainToolCall into a SkillResult by routing through
the SkillBus pipeline. Extracted from Orchestrator._dispatch_one so
each stage is independently testable with mocked dependencies.

Responsibilities:
  - Optional Reasoner pre-check for HIGH/CRITICAL risk calls
  - Confirmation callback wiring
  - Delegation to SkillBus.dispatch() (native — no legacy bridge)
  - Fire-and-forget memory recording

Usage:
    executor = Executor(registry, bus, reasoner, memory)
    result = await executor.dispatch(brain_tool_call, session, on_response_cb)

Phase 3 (core-hardening): New file. Replaces orchestrator._dispatch_one.
Phase 6 (debt): Removed tools/ bridge. dispatch() is now native SkillCall/SkillResult.
"""

from __future__ import annotations

import asyncio
from typing import Callable, Optional

from brain.types import ToolCall as BrainToolCall
from observability.logger import get_logger
from skills.bus import SkillBus
from skills.registry import SkillRegistry
from skills.types import RiskLevel, SkillCall, SkillResult, TrustLevel

log = get_logger(__name__)

# Risk threshold above which the Reasoner runs before dispatch
_REASON_THRESHOLD = RiskLevel.HIGH


from agent.utils import fire_and_forget as _fire_and_forget


class Executor:
    """
    Dispatches a single BrainToolCall through the full skill pipeline.

    Stateless between calls — safe to share across concurrent turns.
    All dependencies are injected; no global singletons.
    """

    def __init__(
        self,
        registry: SkillRegistry,
        bus: SkillBus,
        reasoner,               # agent.reasoner.Reasoner — loose typing avoids circular import
        memory_manager,         # memory.memory_manager.MemoryManager
        confirmation_timeout: int = 120,
    ) -> None:
        self._registry = registry
        self._bus = bus
        self._reasoner = reasoner
        self._memory = memory_manager
        self._confirmation_timeout = confirmation_timeout

    async def dispatch(
        self,
        btc: BrainToolCall,
        session,                                # agent.session.Session
        on_response: Optional[Callable] = None, # streaming callback for confirmation UI
    ) -> SkillResult:
        """
        Dispatch a single tool call end-to-end.

        Steps:
          1. Reasoner pre-check (HIGH/CRITICAL risk only)
          2. Confirmation callback wiring
          3. SkillBus.dispatch() — native, no legacy bridge
          4. Record to episodic memory (fire-and-forget)

        Returns:
            SkillResult — always. Never raises.
        """
        manifest = self._registry.get_schema(btc.name)

        # ── 1. Reasoner pre-check (high-risk only) ────────────────────────────
        if manifest and manifest.risk_level >= _REASON_THRESHOLD:
            verdict = await self._reasoner.evaluate_tool_call(
                tool_name=btc.name,
                tool_args=btc.arguments,
                goal=session.active_plan.goal if session.active_plan else "user request",
                step=(
                    session.active_plan.current_step.description
                    if session.active_plan and session.active_plan.current_step
                    else ""
                ),
            )
            if not verdict.proceed:
                log.warning(
                    "executor.reasoner_blocked",
                    tool=btc.name,
                    concern=verdict.concern,
                )
                return _make_error_result(
                    btc.id,
                    btc.name,
                    f"Reasoner blocked: {verdict.concern or verdict.reasoning}",
                )

        # ── 2. Confirmation callback ──────────────────────────────────────────
        async def _on_confirm(confirm_req) -> bool:
            from agent.response_synthesizer import ResponseSynthesizer
            future = session.register_confirmation(confirm_req.skill_call_id)
            if on_response:
                synth = ResponseSynthesizer()
                on_response(synth.confirmation_request(confirm_req))
            try:
                return await asyncio.wait_for(future, timeout=self._confirmation_timeout)
            except asyncio.TimeoutError:
                log.warning(
                    "executor.confirm_timeout",
                    tool_call_id=confirm_req.skill_call_id,
                )
                session.cancel_confirmation(confirm_req.skill_call_id)
                return False
            except asyncio.CancelledError:
                log.info(
                    "executor.confirm_cancelled",
                    tool_call_id=confirm_req.skill_call_id,
                )
                return False

        # ── 3. Dispatch via SkillBus (native) ────────────────────────────────
        # session.trust_level is already a skills.TrustLevel — same enum.
        # Guard handles any interface mismatch from tests or old callers.
        trust_level: TrustLevel = session.trust_level
        if not isinstance(trust_level, TrustLevel):
            try:
                trust_level = TrustLevel(session.trust_level.value)
            except (ValueError, AttributeError):
                log.error(
                    "executor.trust_coerce_failed",
                    raw=getattr(session.trust_level, "value", session.trust_level),
                )
                trust_level = TrustLevel.LOW  # safe fallback — never over-trust

        skill_call = SkillCall(
            id=btc.id,
            skill_name=btc.name,
            arguments=btc.arguments,
        )

        result: SkillResult = await self._bus.dispatch(
            skill_call,
            trust_level,
            _on_confirm,
            granted_capabilities=session.granted_capabilities,
        )

        session.record_tool_call()

        # ── 4. Memory record (fire-and-forget) ────────────────────────────────
        _fire_and_forget(self._memory.record_tool_call(
            session_id=session.id,
            tool_name=btc.name,
            arguments=btc.arguments,
            result=(result.content or "")[:500],
            is_error=result.is_error,
            risk_level=result.risk_level.value,
            duration_ms=result.duration_ms,
            episode_id=(
                session.active_plan.episode_id if session.active_plan else None
            ),
        ), label="record_tool_call")

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _make_error_result(tool_call_id: str, name: str, error: str) -> SkillResult:
    """Return a SkillResult representing a blocked/failed call."""
    return SkillResult.fail(
        skill_name=name,
        skill_call_id=tool_call_id,
        error=error,
        error_type="ExecutorError",
        blocked=True,
    )
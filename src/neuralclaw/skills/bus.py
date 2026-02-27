"""
skills/bus.py — Skill Bus

Routes SkillCall objects from the orchestrator through the full pipeline:
  1. Registry lookup  — is the skill registered and enabled?
  2. Arg validation   — JSON Schema required-fields + type check
  3. Pre-validation   — SkillBase.validate() for semantic checks
  4. Safety kernel    — risk score, path/command whitelist, confirmation gate
  5. Execution        — async with timeout, all exceptions caught
  6. Result norm      — SkillResult always returned, never raises

Key design decisions:
  - Per-call on_confirm_needed callback avoids mutating shared state during
    parallel dispatches (race condition fix from v1.0.1 preserved here).
  - Retry is intentionally NOT applied to terminal_exec (not idempotent).
    Retryable errors: SkillTimeoutError only (configurable in future phase).
  - SkillResult is always returned — the bus never raises to its caller.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

from neuralclaw.skills.registry import SkillRegistry
from neuralclaw.skills.types import (
    ConfirmationRequest, RiskLevel, SafetyDecision,
    SkillCall, SkillResult, TrustLevel,
)
from neuralclaw.exceptions import NeuralClawError

from neuralclaw.observability.logger import portable_log

if TYPE_CHECKING:
    from neuralclaw.safety.safety_kernel import SafetyKernel

_log = portable_log(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Task 36: RetryPolicy — per-skill exponential backoff with jitter
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RetryPolicy:
    """
    Exponential backoff retry policy for retryable skill errors.

    Only errors whose type name appears in retryable_errors are retried.
    Skills listed in _SKILL_RETRY_OVERRIDES get their own policy.

    Usage:
        policy = RetryPolicy()                 # defaults: 3 attempts, 1s base
        policy = RetryPolicy(max_attempts=1)   # disable retry for a skill
    """
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    jitter: bool = True
    retryable_errors: frozenset[str] = field(
        default_factory=lambda: frozenset({"SkillTimeoutError", "LLMRateLimitError"})
    )

    def is_retryable(self, error_type: str | None) -> bool:
        return error_type in self.retryable_errors

    def delay_for_attempt(self, attempt: int) -> float:
        """Return sleep duration (seconds) before the given retry attempt (1-indexed)."""
        delay = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
        if self.jitter:
            # Full jitter in [80%, 120%] of the computed delay — keeps the mean
            # close to the calculated value while decorrelating concurrent retries.
            delay *= (0.8 + random.random() * 0.4)
        return delay


# Default policy and per-skill overrides
_DEFAULT_RETRY = RetryPolicy()
_SKILL_RETRY_OVERRIDES: dict[str, RetryPolicy] = {
    "terminal_exec": RetryPolicy(max_attempts=1),  # never retry shell (not idempotent)
    "web_fetch":     RetryPolicy(max_attempts=3),
}

# Max output size fed back to LLM
MAX_RESULT_CHARS = 8_000

# Default timeout when manifest doesn't specify one
DEFAULT_TIMEOUT_SECONDS = 30.0





def configure_retry_policy(settings) -> None:
    """
    Apply config.yaml ``skills.retry.*`` values to the module-level retry
    defaults and per-skill overrides.

    Called once at kernel startup so that edits to config.yaml take effect
    without code changes.  Safe to call multiple times (idempotent).

    Args:
        settings: A ``config.settings.Settings`` (or anything with
                  ``settings.skills.retry``).
    """
    global _DEFAULT_RETRY
    retry_cfg = getattr(getattr(settings, "skills", None), "retry", None)
    if retry_cfg is None:
        return

    _DEFAULT_RETRY = RetryPolicy(
        max_attempts=getattr(retry_cfg, "max_attempts", 3),
        base_delay=getattr(retry_cfg, "base_delay", 1.0),
        max_delay=getattr(retry_cfg, "max_delay", 30.0),
        jitter=getattr(retry_cfg, "jitter", True),
        retryable_errors=frozenset(
            getattr(retry_cfg, "retryable_errors", ["SkillTimeoutError", "LLMRateLimitError"])
        ),
    )

    overrides = getattr(retry_cfg, "overrides", {}) or {}
    for skill_name, override in overrides.items():
        max_att = getattr(override, "max_attempts", None)
        if max_att is None and isinstance(override, dict):
            max_att = override.get("max_attempts")
        if max_att is not None:
            _SKILL_RETRY_OVERRIDES[skill_name] = RetryPolicy(
                max_attempts=max_att,
                base_delay=_DEFAULT_RETRY.base_delay,
                max_delay=_DEFAULT_RETRY.max_delay,
                jitter=_DEFAULT_RETRY.jitter,
                retryable_errors=_DEFAULT_RETRY.retryable_errors,
            )

    _log("info", "skill_bus.retry_policy_configured",
         default_max_attempts=_DEFAULT_RETRY.max_attempts,
         overrides=list(_SKILL_RETRY_OVERRIDES.keys()))


class SkillBus:
    """
    Central dispatcher for skill invocations.

    Usage:
        bus = SkillBus(registry, safety_kernel)
        result = await bus.dispatch(skill_call, trust_level=TrustLevel.LOW,
                                    granted_capabilities=session.granted_capabilities)
    """

    def __init__(
        self,
        registry: SkillRegistry,
        safety_kernel: SafetyKernel,
        default_timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        on_confirm_needed: Optional[Callable] = None,
        extra_skill_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        self._registry          = registry
        self._safety            = safety_kernel
        self._default_timeout   = default_timeout_seconds
        self._on_confirm_needed = on_confirm_needed
        self._extra_kwargs      = extra_skill_kwargs or {}  # injected into every execute() call

    # ── Primary dispatch ──────────────────────────────────────────────────────

    async def dispatch(
        self,
        call: SkillCall,
        trust_level: TrustLevel = TrustLevel.LOW,
        on_confirm_needed: Optional[Callable] = None,
        granted_capabilities: frozenset = frozenset(),
    ) -> SkillResult:
        """
        Dispatch a SkillCall through the full pipeline.

        Args:
            call:              The immutable skill call from the orchestrator.
            trust_level:       Session trust level for safety evaluation.
            on_confirm_needed: Per-call confirmation callback (takes priority
                               over the bus-level callback). Signature:
                               async (SafetyDecision) -> bool

        Returns:
            SkillResult — always. Never raises.
        """
        start = time.monotonic()

        _log("info", 
            "skill_bus.dispatch",
            skill=call.skill_name,
            call_id=call.id,
            trust=trust_level.value,
        )

        # ── 1. Registry lookup ─────────────────────────────────────────────
        skill = self._registry.get_or_none(call.skill_name)
        if skill is None:
            return SkillResult.fail(
                skill_name=call.skill_name,
                skill_call_id=call.id,
                error=(
                    f"Skill '{call.skill_name}' is not registered. "
                    f"Available: {sorted(self._registry.list_names())}"
                ),
                error_type="SkillNotFoundError",
            )

        if not skill.manifest.enabled:
            return SkillResult.fail(
                skill_name=call.skill_name,
                skill_call_id=call.id,
                error=f"Skill '{call.skill_name}' is disabled.",
                error_type="SkillDisabledError",
            )

        # ── 2. JSON Schema argument validation ────────────────────────────
        arg_error = _validate_args(call.arguments, skill.manifest.parameters)
        if arg_error:
            return SkillResult.fail(
                skill_name=call.skill_name,
                skill_call_id=call.id,
                error=f"Invalid arguments: {arg_error}",
                error_type="SkillValidationError",
                duration_ms=_ms(start),
            )

        # ── 3. Semantic pre-validation (skill-specific) ───────────────────
        try:
            await skill.validate(**call.arguments)
        except asyncio.CancelledError:
            raise  # never swallow — propagate clean cancel to the orchestrator
        except NeuralClawError as e:
            return SkillResult.fail(
                skill_name=call.skill_name,
                skill_call_id=call.id,
                error=str(e),
                error_type=type(e).__name__,
                duration_ms=_ms(start),
            )
        except BaseException as e:
            return SkillResult.fail(
                skill_name=call.skill_name,
                skill_call_id=call.id,
                error=str(e),
                error_type=type(e).__name__,
                duration_ms=_ms(start),
            )

        # ── 4. Safety kernel — called directly (no tools/ bridge) ───────────
        try:
            decision: Optional[SafetyDecision] = None
            if self._safety is not None:
                decision = await self._safety.evaluate(
                    skill_call=call,
                    manifest=skill.manifest,
                    trust_level=trust_level,
                    granted_capabilities=granted_capabilities,
                )
        except asyncio.CancelledError:
            raise
        except (NeuralClawError, BaseException) as e:
            _log("error", "skill_bus.safety_error", skill=call.skill_name, error=str(e), error_type=type(e).__name__)
            return SkillResult.fail(
                skill_name=call.skill_name,
                skill_call_id=call.id,
                error=f"Safety evaluation failed: {e}",
                error_type="SafetyError",
                duration_ms=_ms(start),
            )

        if decision is not None and decision.is_blocked:
            _log("warning", "skill_bus.blocked", skill=call.skill_name, reason=decision.reason)
            error_msg = f"Blocked by safety kernel: {decision.reason}"
            # U7: Append user-facing grant hint if blocked due to missing capabilities
            from neuralclaw.skills.discovery import capability_hint_from_reason
            hint = capability_hint_from_reason(decision.reason)
            if hint:
                error_msg += f"\n{hint}"
            return SkillResult.fail(
                skill_name=call.skill_name,
                skill_call_id=call.id,
                error=error_msg,
                error_type="SafetyBlockedError",
                duration_ms=_ms(start),
                blocked=True,
            )

        if decision is not None and decision.needs_confirmation:
            confirm_req = ConfirmationRequest.from_decision(decision, arguments=call.arguments)
            approved = await self._handle_confirmation(
                confirm_req,
                per_call_callback=on_confirm_needed,
            )
            if not approved:
                return SkillResult.fail(
                    skill_name=call.skill_name,
                    skill_call_id=call.id,
                    error="Action denied — user did not confirm.",
                    error_type="ConfirmationDeniedError",
                    duration_ms=_ms(start),
                )

        # ── 5. Execute with retry ─────────────────────────────────────────
        timeout = skill.manifest.timeout_seconds or self._default_timeout
        kwargs = dict(call.arguments)
        kwargs["_skill_call_id"] = call.id   # inject so skill can set result id
        kwargs.update(self._extra_kwargs)    # injected bus-level config (allowed_paths, etc.)

        policy = _SKILL_RETRY_OVERRIDES.get(call.skill_name, _DEFAULT_RETRY)
        result = None
        for attempt in range(1, policy.max_attempts + 1):
            try:
                result = await asyncio.wait_for(
                    skill.execute(**kwargs),
                    timeout=timeout,
                )
                duration = _ms(start)  # capture here so it is always defined post-loop
                break  # success — exit retry loop
            except asyncio.TimeoutError:
                duration = _ms(start)
                err_type = "SkillTimeoutError"
                _log("warning", "skill_bus.timeout",
                    skill=call.skill_name, timeout=timeout,
                    attempt=attempt, max_attempts=policy.max_attempts,
                    duration_ms=duration)
                if attempt < policy.max_attempts and policy.is_retryable(err_type):
                    delay = policy.delay_for_attempt(attempt)
                    _log("info", "skill_bus.retry",
                        skill=call.skill_name, attempt=attempt, delay_s=round(delay, 2))
                    await asyncio.sleep(delay)
                    continue
                return SkillResult.fail(
                    skill_name=call.skill_name,
                    skill_call_id=call.id,
                    error=f"Skill timed out after {timeout}s (attempt {attempt}/{policy.max_attempts})",
                    error_type=err_type,
                    duration_ms=duration,
                )
            except asyncio.CancelledError:
                raise   # propagate clean cancel — never convert to a skill error
            except NeuralClawError as e:
                duration = _ms(start)
                err_type = type(e).__name__
                _log("warning", "skill_bus.execution_error",
                    skill=call.skill_name, error=str(e), error_type=err_type,
                    attempt=attempt, duration_ms=duration)
                if attempt < policy.max_attempts and policy.is_retryable(err_type):
                    delay = policy.delay_for_attempt(attempt)
                    _log("info", "skill_bus.retry",
                        skill=call.skill_name, attempt=attempt, delay_s=round(delay, 2))
                    await asyncio.sleep(delay)
                    continue
                return SkillResult.fail(
                    skill_name=call.skill_name,
                    skill_call_id=call.id,
                    error=f"{err_type}: {e}",
                    error_type=err_type,
                    duration_ms=duration,
                )
            except BaseException as e:
                duration = _ms(start)
                _log("error", "skill_bus.unexpected_error",
                    skill=call.skill_name, error=str(e),
                    error_type=type(e).__name__, duration_ms=duration, exc_info=True)
                return SkillResult.fail(
                    skill_name=call.skill_name,
                    skill_call_id=call.id,
                    error=f"{type(e).__name__}: {e}",
                    error_type=type(e).__name__,
                    duration_ms=duration,
                )

        # Build the final SkillResult exactly once
        risk_val = decision.risk_level.value if decision else skill.manifest.risk_level.value

        if isinstance(result, SkillResult) and result.success:
            output = result.output
            meta = {**result.metadata, "risk_level": risk_val}
        elif isinstance(result, SkillResult):
            # Failed SkillResult — honour it as-is
            return result
        else:
            # Skill returned a raw value (str/dict/etc.) — wrap it
            output = result
            meta = {"risk_level": risk_val}

        # Truncate large string outputs
        if isinstance(output, str) and len(output) > MAX_RESULT_CHARS:
            output = (
                output[:MAX_RESULT_CHARS]
                + f"\n\n[Output truncated — {len(output) - MAX_RESULT_CHARS} chars omitted]"
            )
            meta["truncated"] = True

        result = SkillResult.ok(
            skill_name=call.skill_name,
            skill_call_id=call.id,
            output=output,
            duration_ms=duration,
            metadata=meta,
        )

        _log("info", 
            "skill_bus.success",
            skill=call.skill_name,
            call_id=call.id,
            duration_ms=round(duration, 1),
        )
        return result

    async def _handle_confirmation(
        self,
        confirm_req: ConfirmationRequest,
        per_call_callback: Optional[Callable] = None,
    ) -> bool:
        """
        Request user confirmation. Passes a ConfirmationRequest (not raw
        SafetyDecision) to the callback so the UI has a stable contract.
        Uses per-call callback first to avoid shared-state mutation during
        parallel dispatches.
        """
        callback = per_call_callback or self._on_confirm_needed
        if callback is None:
            _log("warning",
                "skill_bus.confirm_no_handler",
                skill=confirm_req.skill_name,
            )
            return False

        try:
            return await callback(confirm_req)
        except asyncio.CancelledError:
            raise
        except BaseException as e:
            _log("error", "skill_bus.confirm_handler_error", error=str(e), error_type=type(e).__name__)
            return False



# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ms(start: float) -> float:
    return (time.monotonic() - start) * 1000


def _validate_args(arguments: dict, schema: dict) -> Optional[str]:
    """
    Validate arguments against a JSON Schema dict.
    Returns an error string or None if valid.
    Checks: required fields present + declared types match.
    """
    required = schema.get("required", [])
    properties = schema.get("properties", {})

    for field in required:
        if field not in arguments:
            return f"Missing required field: '{field}'"

    _TYPE_MAP: dict[str, type | tuple] = {
        "string":  str,
        "integer": int,
        "number":  (int, float),
        "boolean": bool,
        "array":   list,
        "object":  dict,
    }
    for field, value in arguments.items():
        prop = properties.get(field)
        if prop is None:
            continue
        json_type = prop.get("type")
        if json_type is None:
            continue
        expected = _TYPE_MAP.get(json_type)
        if expected is None:
            continue
        if json_type == "integer" and isinstance(value, bool):
            return f"Field '{field}': expected integer, got boolean"
        if not isinstance(value, expected):
            return f"Field '{field}': expected {json_type}, got {type(value).__name__}"

    return None


def _bridge_risk_back(value: str) -> "RiskLevel":
    """Convert a risk level string to skills.types.RiskLevel."""
    try:
        return RiskLevel(value.upper() if isinstance(value, str) else value)
    except (ValueError, AttributeError):
        return RiskLevel.LOW
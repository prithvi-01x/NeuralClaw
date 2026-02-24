"""
safety/safety_kernel.py — Safety Kernel

Evaluates every SkillCall before execution and returns an immutable
SafetyDecision (APPROVED | CONFIRM_NEEDED | BLOCKED).

Design principles
-----------------
* Operates entirely on skills.types — zero coupling to the legacy tools/ system.
* Fail-closed: unknown tools → BLOCKED, unknown risk → CRITICAL.
* Capability-based: checks session.granted_capabilities against
  skill.manifest.capabilities before risk scoring.
* Typed decision: SafetyDecision is a frozen dataclass; no string sniffing.

Phase 3 (core-hardening): Ported from tools.types to skills.types.
                          Added capability-based permission checks.
"""

from __future__ import annotations

from typing import Optional

from observability.logger import get_logger
from safety.risk_scorer import score_tool_call
from safety.whitelist import check_command, check_path
from skills.types import (
    RiskLevel,
    SafetyDecision,
    SafetyStatus,
    SkillCall,
    SkillManifest,
    TrustLevel,
)

log = get_logger(__name__)


class SafetyKernel:
    """
    Gate-keeper for all skill executions.

    Stateless between calls — safe to share across sessions.
    All configuration is injected at construction time.
    """

    def __init__(
        self,
        allowed_paths:   Optional[list[str]] = None,
        whitelist_extra: Optional[list[str]] = None,
    ) -> None:
        self._allowed_paths   = allowed_paths   or []
        self._whitelist_extra = whitelist_extra or []

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    async def evaluate(
        self,
        skill_call:           SkillCall,
        manifest:             SkillManifest,
        trust_level:          TrustLevel = TrustLevel.LOW,
        granted_capabilities: frozenset  = frozenset(),
    ) -> SafetyDecision:
        """
        Evaluate a skill call and return a SafetyDecision.

        Order of checks (first failure wins):
          1. Skill enabled check
          2. Capability check (manifest.capabilities ⊆ granted_capabilities)
          3. Category-specific whitelist (terminal / filesystem)
          4. Argument-based risk scoring
          5. Trust-level gate

        Returns SafetyDecision — never raises.
        """
        name    = skill_call.skill_name
        call_id = skill_call.id

        # ── 1. Enabled check ─────────────────────────────────────────────────
        if hasattr(manifest, "enabled") and not manifest.enabled:
            reason = f"Skill '{name}' is disabled in configuration."
            log.info("safety.blocked.disabled", skill=name, call_id=call_id)
            return self._decision(skill_call, RiskLevel.MEDIUM,
                                  SafetyStatus.BLOCKED, reason)

        # ── 2. Capability check ───────────────────────────────────────────────
        required = manifest.capabilities or frozenset()
        if required and not required.issubset(granted_capabilities):
            missing = required - granted_capabilities
            reason = (
                f"Skill '{name}' requires capabilities {sorted(missing)} "
                f"that have not been granted to this session."
            )
            log.warning(
                "safety.blocked.capability",
                skill=name,
                call_id=call_id,
                required=sorted(required),
                granted=sorted(granted_capabilities),
                missing=sorted(missing),
            )
            return self._decision(skill_call, RiskLevel.MEDIUM,
                                  SafetyStatus.BLOCKED, reason)

        # ── 3. Category-specific whitelist checks ─────────────────────────────
        if manifest.category == "terminal":
            allowed, wl_reason, is_hr = check_command(
                skill_call.arguments.get("command", ""),
                extra_allowed=self._whitelist_extra,
            )
            if not allowed:
                log.warning("safety.blocked.terminal", skill=name, reason=wl_reason)
                return self._decision(skill_call, RiskLevel.HIGH,
                                      SafetyStatus.BLOCKED, wl_reason)

        elif manifest.category == "filesystem":
            path_arg = skill_call.arguments.get("path", "")
            operation = "write" if any(
                kw in name for kw in ("write", "append", "delete", "move")
            ) else "read"
            allowed, wl_reason = check_path(path_arg, self._allowed_paths, operation)
            if not allowed:
                log.warning("safety.blocked.path", skill=name, reason=wl_reason)
                return self._decision(skill_call, RiskLevel.HIGH,
                                      SafetyStatus.BLOCKED, wl_reason)

        # ── 4. Argument-based risk scoring ────────────────────────────────────
        effective_risk, score_reason = score_tool_call(skill_call, manifest)

        # ── 5. Trust-level gate ────────────────────────────────────────────────
        status, final_reason = self._apply_trust(
            effective_risk, trust_level, score_reason, manifest
        )

        log.debug(
            "safety.decision",
            skill=name,
            call_id=call_id,
            status=status.value,
            risk=effective_risk.value,
            reason=final_reason,
        )
        return self._decision(skill_call, effective_risk, status, final_reason)

    # ─────────────────────────────────────────────────────────────────────────
    # Internals
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _apply_trust(
        risk:     RiskLevel,
        trust:    TrustLevel,
        reason:   str,
        manifest: SkillManifest,
    ) -> tuple[SafetyStatus, str]:
        """
        Map (risk, trust) → (SafetyStatus, final_reason).

        Trust=LOW:    confirm on HIGH+  (block on CRITICAL if not confirmable)
        Trust=MEDIUM: confirm on CRITICAL only
        Trust=HIGH:   auto-approve everything
        """
        requires_confirm = getattr(manifest, "requires_confirmation", False)

        if trust == TrustLevel.HIGH:
            return SafetyStatus.APPROVED, reason

        if trust == TrustLevel.MEDIUM:
            if risk == RiskLevel.CRITICAL:
                if requires_confirm:
                    return SafetyStatus.CONFIRM_NEEDED, f"CRITICAL risk requires confirmation: {reason}"
                return SafetyStatus.BLOCKED, f"CRITICAL risk auto-blocked at MEDIUM trust: {reason}"
            return SafetyStatus.APPROVED, reason

        # trust == LOW (default)
        if risk == RiskLevel.CRITICAL:
            if requires_confirm:
                return SafetyStatus.CONFIRM_NEEDED, f"CRITICAL risk requires confirmation: {reason}"
            return SafetyStatus.BLOCKED, f"CRITICAL risk auto-blocked at LOW trust: {reason}"
        if risk == RiskLevel.HIGH:
            return SafetyStatus.CONFIRM_NEEDED, f"HIGH risk requires confirmation: {reason}"
        return SafetyStatus.APPROVED, reason

    @staticmethod
    def _decision(
        skill_call: SkillCall,
        risk:       RiskLevel,
        status:     SafetyStatus,
        reason:     str,
    ) -> SafetyDecision:
        return SafetyDecision(
            status=status,
            reason=reason,
            risk_level=risk,
            tool_name=skill_call.skill_name,
            tool_call_id=skill_call.id,
        )
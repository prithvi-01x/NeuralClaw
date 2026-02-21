"""
safety/safety_kernel.py — Safety Kernel

The gatekeeper that sits between the LLM and tool execution.
Every tool call MUST pass through here before it executes.

Decision flow:
  1. Is the tool registered and enabled?
  2. Check blocked patterns (terminal) / blocked paths (filesystem)
  3. Score the risk level (baseline + argument escalation)
  4. Apply trust level: does this risk require confirmation?
  5. Emit audit log entry
  6. Return SafetyDecision: APPROVED | BLOCKED | CONFIRM_NEEDED
"""

from __future__ import annotations

from typing import Optional

from observability.logger import get_logger
from safety.risk_scorer import score_tool_call
from safety.whitelist import check_command, check_path
from tools.types import (
    RiskLevel,
    SafetyDecision,
    SafetyStatus,
    ToolCall,
    ToolSchema,
    TrustLevel,
)

log = get_logger(__name__)

# Risk threshold at which confirmation is required, per trust level
_CONFIRM_THRESHOLD: dict[TrustLevel, RiskLevel] = {
    TrustLevel.LOW: RiskLevel.HIGH,       # confirm HIGH and CRITICAL
    TrustLevel.MEDIUM: RiskLevel.CRITICAL, # confirm only CRITICAL
    TrustLevel.HIGH: None,                 # never confirm (auto-approve all)
}


class SafetyKernel:
    """
    Evaluates tool calls for safety before execution.

    Usage:
        kernel = SafetyKernel(allowed_paths=["~/agent_files"])
        decision = await kernel.evaluate(tool_call, schema, trust_level)
        if decision.is_approved:
            result = await tool.execute(tool_call)
        elif decision.needs_confirmation:
            # ask the user via Telegram / CLI
        else:
            # blocked — return error to LLM
    """

    def __init__(
        self,
        allowed_paths: Optional[list[str]] = None,
        extra_allowed_commands: Optional[list[str]] = None,
    ):
        self.allowed_paths = allowed_paths or ["~/agent_files"]
        self.extra_allowed_commands = extra_allowed_commands or []

    async def evaluate(
        self,
        tool_call: ToolCall,
        schema: ToolSchema,
        trust_level: TrustLevel = TrustLevel.LOW,
    ) -> SafetyDecision:
        """
        Evaluate a tool call and return a safety decision.

        Args:
            tool_call:   The tool call from the LLM.
            schema:      The tool's registered schema.
            trust_level: The current session's trust level.

        Returns:
            SafetyDecision with status APPROVED, BLOCKED, or CONFIRM_NEEDED.
        """
        # ── Step 1: Tool enabled? ─────────────────────────────────────────────
        if not schema.enabled:
            return self._decision(
                tool_call, RiskLevel.LOW, SafetyStatus.BLOCKED,
                f"Tool '{tool_call.name}' is disabled in configuration"
            )

        # ── Step 2: Category-specific whitelist checks ────────────────────────
        if schema.category == "terminal":
            command = tool_call.arguments.get("command", "")
            allowed, reason, is_high_risk = check_command(
                command, self.extra_allowed_commands
            )
            if not allowed:
                return self._decision(
                    tool_call, RiskLevel.CRITICAL, SafetyStatus.BLOCKED, reason
                )
            # If command is high-risk, ensure at least HIGH risk level
            if is_high_risk:
                effective_risk = RiskLevel.HIGH
                score_reason = f"High-risk command detected: {reason}"
            else:
                effective_risk, score_reason = score_tool_call(tool_call, schema)

        elif schema.category == "filesystem":
            path = tool_call.arguments.get(
                "path",
                tool_call.arguments.get("file_path", "")
            )
            operation = "write" if "write" in tool_call.name or "create" in tool_call.name else "read"
            if path:
                path_allowed, path_reason = check_path(
                    path, self.allowed_paths, operation
                )
                if not path_allowed:
                    return self._decision(
                        tool_call, RiskLevel.CRITICAL, SafetyStatus.BLOCKED, path_reason
                    )
            effective_risk, score_reason = score_tool_call(tool_call, schema)

        else:
            # For all other tools, just score based on args
            effective_risk, score_reason = score_tool_call(tool_call, schema)

        # ── Step 3: Apply trust level to determine final status ───────────────
        confirm_threshold = _CONFIRM_THRESHOLD.get(trust_level)

        if effective_risk == RiskLevel.CRITICAL and confirm_threshold is None:
            # Even HIGH trust requires explicit opt-in to CRITICAL actions
            # (user must set /trust critical explicitly — future feature)
            status = SafetyStatus.CONFIRM_NEEDED
            final_reason = f"CRITICAL risk action requires confirmation regardless of trust level"
        elif confirm_threshold is None:
            status = SafetyStatus.APPROVED
            final_reason = score_reason
        elif effective_risk >= confirm_threshold:
            status = SafetyStatus.CONFIRM_NEEDED
            final_reason = (
                f"Risk level {effective_risk.value} meets confirmation threshold "
                f"({confirm_threshold.value}) for trust level {trust_level.value}"
            )
        else:
            status = SafetyStatus.APPROVED
            final_reason = score_reason

        decision = self._decision(tool_call, effective_risk, status, final_reason)

        # ── Step 4: Audit log ─────────────────────────────────────────────────
        log_fn = log.warning if status != SafetyStatus.APPROVED else log.info
        log_fn(
            "safety.decision",
            tool=tool_call.name,
            tool_call_id=tool_call.id,
            status=status.value,
            risk_level=effective_risk.value,
            trust_level=trust_level.value,
            reason=final_reason,
        )

        return decision

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _decision(
        self,
        tool_call: ToolCall,
        risk_level: RiskLevel,
        status: SafetyStatus,
        reason: str,
    ) -> SafetyDecision:
        return SafetyDecision(
            status=status,
            reason=reason,
            risk_level=risk_level,
            tool_name=tool_call.name,
            tool_call_id=tool_call.id,
        )
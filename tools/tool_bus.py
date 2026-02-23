"""
tools/tool_bus.py — Tool Bus

The central dispatcher that sits between the LLM brain and tool execution.
Every tool call from the LLM is routed through here.

Flow:
  LLM tool_call → ToolBus.dispatch()
    → Registry lookup (is tool registered?)
    → Parameter validation (JSON schema)
    → Safety Kernel (APPROVED / BLOCKED / CONFIRM_NEEDED)
    → Handler execution (async, with timeout)
    → ToolResult (success or error)
    → Audit log
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Callable, Optional

from observability.logger import get_logger
from safety.safety_kernel import SafetyKernel
from tools.tool_registry import ToolRegistry
from tools.types import (
    RiskLevel,
    SafetyStatus,
    ToolCall,
    ToolResult,
    TrustLevel,
)

log = get_logger(__name__)

# Max output size fed back to LLM — truncate beyond this
MAX_RESULT_CHARS = 8_000

# Default tool execution timeout
DEFAULT_TIMEOUT_SECONDS = 30.0


class ToolBus:
    """
    Routes tool calls from the LLM through safety checks to execution.

    Usage:
        bus = ToolBus(registry, safety_kernel)
        result = await bus.dispatch(tool_call, trust_level=TrustLevel.LOW)
    """

    def __init__(
        self,
        registry: ToolRegistry,
        safety_kernel: SafetyKernel,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        on_confirm_needed: Optional[Callable] = None,
    ):
        """
        Args:
            registry:         Tool registry with all registered tools.
            safety_kernel:    Safety kernel instance.
            timeout_seconds:  Max seconds a tool may run before being cancelled.
            on_confirm_needed: Async callback when confirmation is needed.
                               Signature: async (SafetyDecision) -> bool
                               Return True to approve, False to deny.
                               If None, CONFIRM_NEEDED is treated as BLOCKED.
        """
        self.registry = registry
        self.safety = safety_kernel
        self.timeout_seconds = timeout_seconds
        self.on_confirm_needed = on_confirm_needed

    async def dispatch(
        self,
        tool_call: ToolCall,
        trust_level: TrustLevel = TrustLevel.LOW,
        on_confirm_needed: Optional[Callable] = None,
    ) -> ToolResult:
        """
        Dispatch a tool call through the full pipeline.

        Args:
            tool_call:        The tool call from the LLM.
            trust_level:      Current session trust level.
            on_confirm_needed: Per-call confirmation callback override.
                               If provided, takes priority over self.on_confirm_needed.
                               Signature: async (SafetyDecision) -> bool

        Returns:
            ToolResult (always — never raises, errors are captured in result).
        """
        start_ms = time.monotonic() * 1000

        log.info(
            "tool_bus.dispatch",
            tool=tool_call.name,
            tool_call_id=tool_call.id,
            trust_level=trust_level.value,
        )

        # ── Step 1: Registry lookup ───────────────────────────────────────────
        schema = self.registry.get_schema(tool_call.name)
        if schema is None:
            return ToolResult.error(
                tool_call.id,
                tool_call.name,
                f"Unknown tool '{tool_call.name}'. Available tools: "
                f"{self.registry.list_names()}",
            )

        handler = self.registry.get_handler(tool_call.name)
        if handler is None:
            return ToolResult.error(
                tool_call.id,
                tool_call.name,
                f"Tool '{tool_call.name}' has no handler registered.",
            )

        # ── Step 2: Parameter validation ──────────────────────────────────────
        validation_error = _validate_args(tool_call.arguments, schema.parameters)
        if validation_error:
            return ToolResult.error(
                tool_call.id,
                tool_call.name,
                f"Invalid parameters: {validation_error}",
                risk_level=schema.risk_level,
            )

        # ── Step 3: Safety kernel ─────────────────────────────────────────────
        decision = await self.safety.evaluate(tool_call, schema, trust_level)

        if decision.is_blocked:
            log.warning(
                "tool_bus.blocked",
                tool=tool_call.name,
                reason=decision.reason,
            )
            return ToolResult.error(
                tool_call.id,
                tool_call.name,
                f"Blocked by safety kernel: {decision.reason}",
                risk_level=decision.risk_level,
            )

        if decision.needs_confirmation:
            approved = await self._handle_confirmation(decision, on_confirm_needed)
            if not approved:
                return ToolResult.error(
                    tool_call.id,
                    tool_call.name,
                    "Action denied by user.",
                    risk_level=decision.risk_level,
                )

        # ── Step 4: Execute with timeout ──────────────────────────────────────
        try:
            raw_result = await asyncio.wait_for(
                handler(**tool_call.arguments),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError:
            duration_ms = time.monotonic() * 1000 - start_ms
            log.error(
                "tool_bus.timeout",
                tool=tool_call.name,
                timeout_seconds=self.timeout_seconds,
                duration_ms=duration_ms,
            )
            return ToolResult.error(
                tool_call.id,
                tool_call.name,
                f"Tool '{tool_call.name}' timed out after {self.timeout_seconds}s",
                risk_level=schema.risk_level,
            )
        except Exception as e:
            duration_ms = time.monotonic() * 1000 - start_ms
            log.error(
                "tool_bus.execution_error",
                tool=tool_call.name,
                error=str(e),
                duration_ms=duration_ms,
                exc_info=True,
            )
            return ToolResult.error(
                tool_call.id,
                tool_call.name,
                f"Tool execution failed: {type(e).__name__}: {e}",
                risk_level=schema.risk_level,
            )

        # ── Step 5: Normalise and truncate result ─────────────────────────────
        duration_ms = time.monotonic() * 1000 - start_ms
        content = _normalise_result(raw_result)
        content = _truncate(content, MAX_RESULT_CHARS)

        log.info(
            "tool_bus.success",
            tool=tool_call.name,
            tool_call_id=tool_call.id,
            duration_ms=round(duration_ms, 1),
            result_chars=len(content),
            risk_level=schema.risk_level.value,
        )

        return ToolResult.success(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            content=content,
            risk_level=schema.risk_level,
            duration_ms=duration_ms,
        )

    async def _handle_confirmation(self, decision, per_call_callback: Optional[Callable] = None) -> bool:
        """
        Request user confirmation for a high-risk action.
        Returns True if approved, False if denied.

        Uses per_call_callback if provided (avoids mutating shared state during
        parallel tool dispatch), falling back to self.on_confirm_needed.
        """
        callback = per_call_callback or self.on_confirm_needed
        if callback is None:
            log.warning(
                "tool_bus.confirm_needed_no_handler",
                tool=decision.tool_name,
                reason="No confirmation handler registered — treating as denied",
            )
            return False

        try:
            return await callback(decision)
        except Exception as e:
            log.error("tool_bus.confirm_handler_error", error=str(e))
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _validate_args(arguments: dict, schema: dict) -> Optional[str]:
    """
    Validate tool arguments against the JSON schema.
    Returns an error string if invalid, None if valid.

    Checks:
      1. All required fields are present.
      2. Provided values match the declared JSON Schema types.
         This catches bad LLM-generated arguments (e.g. string where int
         expected) before the tool handler is invoked, giving the LLM a
         clear error it can act on rather than an obscure TypeError mid-tool.
    """
    required = schema.get("required", [])
    properties = schema.get("properties", {})

    # 1. Required-field presence check
    for field in required:
        if field not in arguments:
            return f"Missing required field: '{field}'"

    # 2. Type check every provided argument against its schema declaration
    _JSON_TYPE_MAP: dict[str, type | tuple] = {
        "string":  str,
        "integer": int,
        "number":  (int, float),
        "boolean": bool,
        "array":   list,
        "object":  dict,
    }
    for field, value in arguments.items():
        prop_schema = properties.get(field)
        if prop_schema is None:
            continue  # unknown field — lenient, don't block
        json_type = prop_schema.get("type")
        if json_type is None:
            continue  # no type declared
        expected = _JSON_TYPE_MAP.get(json_type)
        if expected is None:
            continue  # unrecognised type string
        # bool is a subclass of int in Python, so check bool explicitly first
        if json_type == "integer" and isinstance(value, bool):
            return f"Field '{field}': expected integer, got boolean"
        if not isinstance(value, expected):
            actual = type(value).__name__
            return f"Field '{field}': expected {json_type}, got {actual}"

    return None


def _normalise_result(result) -> str:
    """Convert any tool return value to a string."""
    if result is None:
        return "Done."
    if isinstance(result, str):
        return result
    if isinstance(result, (dict, list)):
        try:
            return json.dumps(result, indent=2, default=str)
        except Exception:
            return str(result)
    return str(result)


def _truncate(text: str, max_chars: int) -> str:
    """Truncate result if too long, with a notice."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    return (
        truncated
        + f"\n\n[Output truncated — {len(text) - max_chars} chars omitted. "
        f"Total: {len(text)} chars]"
    )
"""
tools/types.py — Tool System Data Models

Shared types used across the tool registry, tool bus, safety kernel,
and all tool implementations.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Risk / Permission enums
# ─────────────────────────────────────────────────────────────────────────────


class RiskLevel(str, Enum):
    """
    Risk level of a tool call. Used by the Safety Kernel to decide
    whether to auto-approve, ask for confirmation, or block.
    """
    LOW = "LOW"           # Read-only, no side effects (search, file_read)
    MEDIUM = "MEDIUM"     # Write operations with limited blast radius (file_write)
    HIGH = "HIGH"         # System-level actions (terminal_exec, git push)
    CRITICAL = "CRITICAL" # Destructive, irreversible (rm -rf, format disk)

    def _order(self) -> int:
        return [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL].index(self)

    def __lt__(self, other: "RiskLevel") -> bool:
        return self._order() < other._order()

    def __le__(self, other: "RiskLevel") -> bool:
        return self._order() <= other._order()

    def __gt__(self, other: "RiskLevel") -> bool:
        return self._order() > other._order()

    def __ge__(self, other: "RiskLevel") -> bool:
        return self._order() >= other._order()


class TrustLevel(str, Enum):
    """
    Trust level granted to the current session by the user.
    Controls how much the agent can do without asking for confirmation.
    """
    LOW = "low"       # Confirm all HIGH+ actions (default)
    MEDIUM = "medium" # Confirm only CRITICAL actions
    HIGH = "high"     # Auto-approve everything (use with caution)


class SafetyStatus(str, Enum):
    APPROVED = "approved"
    BLOCKED = "blocked"
    CONFIRM_NEEDED = "confirm_needed"


# ─────────────────────────────────────────────────────────────────────────────
# Tool registration metadata
# ─────────────────────────────────────────────────────────────────────────────


class ToolSchema(BaseModel):
    """
    Full metadata for a registered tool.
    Stored in ToolRegistry and used by the LLM + Safety Kernel.
    """
    name: str
    description: str
    risk_level: RiskLevel = RiskLevel.LOW
    requires_confirmation: bool = False
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}, "required": []}
    )
    category: str = "general"      # e.g. "filesystem", "terminal", "search"
    enabled: bool = True

    def to_llm_schema(self) -> dict[str, Any]:
        """Return the schema in a format the LLM brain expects."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Runtime tool call / result types
# ─────────────────────────────────────────────────────────────────────────────


class ToolCall(BaseModel):
    """A tool invocation from the LLM, before execution."""
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """The result of a tool call after execution."""
    tool_call_id: str
    name: str
    content: str                        # JSON string or plain text
    is_error: bool = False
    duration_ms: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW

    @classmethod
    def success(
        cls,
        tool_call_id: str,
        name: str,
        content: str,
        risk_level: RiskLevel = RiskLevel.LOW,
        duration_ms: float = 0.0,
    ) -> "ToolResult":
        return cls(
            tool_call_id=tool_call_id,
            name=name,
            content=content,
            is_error=False,
            risk_level=risk_level,
            duration_ms=duration_ms,
        )

    @classmethod
    def error(
        cls,
        tool_call_id: str,
        name: str,
        error_message: str,
        risk_level: RiskLevel = RiskLevel.LOW,
    ) -> "ToolResult":
        return cls(
            tool_call_id=tool_call_id,
            name=name,
            content=f"Error: {error_message}",
            is_error=True,
            risk_level=risk_level,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Safety kernel decision
# ─────────────────────────────────────────────────────────────────────────────


class SafetyDecision(BaseModel):
    """The Safety Kernel's verdict on a tool call."""
    status: SafetyStatus
    reason: str
    risk_level: RiskLevel
    tool_name: str
    tool_call_id: str

    @property
    def is_approved(self) -> bool:
        return self.status == SafetyStatus.APPROVED

    @property
    def is_blocked(self) -> bool:
        return self.status == SafetyStatus.BLOCKED

    @property
    def needs_confirmation(self) -> bool:
        return self.status == SafetyStatus.CONFIRM_NEEDED
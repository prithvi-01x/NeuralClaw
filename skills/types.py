"""
skills/types.py — Skill System Data Contracts

All dataclasses and enums shared across the skill layer.
Nothing in skills/ imports from tools/ — these types stand alone.

Phase 2 (core-hardening): New types introduced here.
  - SkillManifest:   static metadata every skill must declare
  - SkillCall:       immutable snapshot of one invocation (never mutated post-creation)
  - SkillResult:     typed result returned from every skill execution
  - SafetyStatus:    verdict enum from the safety kernel
  - SafetyDecision:  full safety verdict including risk, reason, and blocked flag
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Risk / Trust (mirrors tools/types.py — skills/ does not import tools/)
# ─────────────────────────────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"

    def _order(self) -> int:
        return ["LOW", "MEDIUM", "HIGH", "CRITICAL"].index(self.value)

    def __lt__(self, other: "RiskLevel") -> bool: return self._order() < other._order()
    def __le__(self, other: "RiskLevel") -> bool: return self._order() <= other._order()
    def __gt__(self, other: "RiskLevel") -> bool: return self._order() > other._order()
    def __ge__(self, other: "RiskLevel") -> bool: return self._order() >= other._order()


class TrustLevel(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


# ─────────────────────────────────────────────────────────────────────────────
# SkillManifest — static, declared as ClassVar on every skill
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SkillManifest:
    """
    Static metadata for a skill. Declared as a ClassVar on SkillBase subclasses.

    Rules:
      - name must be snake_case, globally unique across all loaded skills.
      - version must be a semver string ("1.0.0").
      - capabilities is a frozenset of capability strings ("fs:read", "net:fetch", etc.).
      - parameters is a JSON Schema dict that the ToolBus/LLM uses.
      - timeout_seconds: how long the skill may run before being cancelled.
    """
    name: str
    version: str
    description: str
    category: str
    risk_level: RiskLevel
    parameters: dict = field(default_factory=dict)
    capabilities: frozenset = field(default_factory=frozenset)
    timeout_seconds: int = 30
    requires_confirmation: bool = False
    enabled: bool = True

    def to_llm_schema(self) -> dict[str, Any]:
        """Return the schema in the format LLM brain expects."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def to_tool_schema_dict(self) -> dict[str, Any]:
        """Return a full dict mirroring ToolSchema for backward-compat bridge."""
        return {
            "name": self.name,
            "description": self.description,
            "risk_level": self.risk_level,
            "requires_confirmation": self.requires_confirmation,
            "parameters": self.parameters,
            "category": self.category,
            "enabled": self.enabled,
        }


# ─────────────────────────────────────────────────────────────────────────────
# SkillCall — immutable invocation snapshot
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SkillCall:
    """
    An immutable snapshot of one skill invocation from the LLM.

    Created once. Never mutated. The safety kernel evaluates this object
    and returns a verdict; the SkillBus then executes it unchanged.
    """
    id: str                               # matches LLM tool_call_id
    skill_name: str
    arguments: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.monotonic)


# ─────────────────────────────────────────────────────────────────────────────
# SkillResult — typed result from every execution
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SkillResult:
    """
    The result of a skill execution.

    Rules:
      - success=True means the skill ran and produced output.
      - success=False means it failed; error and error_type describe why.
      - blocked=True means the safety kernel hard-blocked this call (subset of success=False).
      - Skills must NEVER raise — they return SkillResult(success=False, ...) instead.
      - output is what the LLM receives as the tool result.
    """
    success: bool
    output: Any                            # str, dict, list — anything JSON-serialisable
    skill_name: str
    skill_call_id: str
    error: Optional[str] = None
    error_type: Optional[str] = None      # exception class name
    duration_ms: float = 0.0
    blocked: bool = False                  # True when safety kernel hard-blocked the call
    metadata: dict = field(default_factory=dict)

    @classmethod
    def ok(
        cls,
        skill_name: str,
        skill_call_id: str,
        output: Any,
        duration_ms: float = 0.0,
        metadata: Optional[dict] = None,
    ) -> "SkillResult":
        return cls(
            success=True,
            output=output,
            skill_name=skill_name,
            skill_call_id=skill_call_id,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )

    @classmethod
    def fail(
        cls,
        skill_name: str,
        skill_call_id: str,
        error: str,
        error_type: str = "SkillError",
        duration_ms: float = 0.0,
        blocked: bool = False,
    ) -> "SkillResult":
        return cls(
            success=False,
            output=None,
            skill_name=skill_name,
            skill_call_id=skill_call_id,
            error=error,
            error_type=error_type,
            duration_ms=duration_ms,
            blocked=blocked,
        )

    def to_llm_content(self) -> str:
        """Return the string the LLM sees as the tool result."""
        import json
        if not self.success:
            return f"ERROR ({self.error_type}): {self.error}"
        if isinstance(self.output, str):
            return self.output
        try:
            return json.dumps(self.output, indent=2, default=str)
        except Exception:
            return str(self.output)

    # ── Compatibility shims ───────────────────────────────────────────────────
    # These properties make SkillResult a drop-in for the legacy ToolResult
    # that the orchestrator, response_synthesizer, and memory recorder read.
    # They let us delete tools/ without touching every call site at once.

    @property
    def content(self) -> str:
        """LLM-facing content string (ToolResult compat)."""
        return self.to_llm_content()

    @property
    def name(self) -> str:
        """Skill name (ToolResult compat — ToolResult.name == skill_name)."""
        return self.skill_name

    @property
    def is_error(self) -> bool:
        """True when the skill failed (ToolResult compat)."""
        return not self.success

    @property
    def risk_level(self) -> "RiskLevel":
        """Risk level pulled from metadata (ToolResult compat)."""
        raw = self.metadata.get("risk_level", "LOW")
        try:
            return RiskLevel(raw.upper() if isinstance(raw, str) else raw)
        except (ValueError, AttributeError):
            return RiskLevel.LOW


# ─────────────────────────────────────────────────────────────────────────────
# Safety types — live here so safety/ can import skills.types without tools/
# ─────────────────────────────────────────────────────────────────────────────

class SafetyStatus(str, Enum):
    """Verdict from the safety kernel for a single skill call."""
    APPROVED       = "approved"
    BLOCKED        = "blocked"
    CONFIRM_NEEDED = "confirm_needed"


@dataclass(frozen=True)
class SafetyDecision:
    """The safety kernel's full verdict on a skill call."""
    status:       SafetyStatus
    reason:       str
    risk_level:   RiskLevel
    tool_name:    str
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


# ─────────────────────────────────────────────────────────────────────────────
# Skill-layer errors  (full hierarchy lives in exceptions.py)
# ─────────────────────────────────────────────────────────────────────────────

class SkillNotFoundError(Exception):
    """Raised by SkillRegistry when a skill name is not registered."""

# SkillValidationError lives in exceptions.py (inherits NeuralClawError → SkillError)
# so the SkillBus catches it via `except NeuralClawError`.
# Re-exported here for backward compatibility with all existing imports.
from exceptions import SkillValidationError as SkillValidationError  # noqa: E402
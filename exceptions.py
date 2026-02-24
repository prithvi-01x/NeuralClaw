"""
exceptions.py — NeuralClaw Unified Error Hierarchy

All NeuralClaw-specific exceptions live here. Every layer of the stack
raises typed subclasses of NeuralClawError — never bare Exception.

Import from here, not from individual modules:
    from exceptions import SkillTimeoutError, CapabilityDeniedError

Hierarchy:
    NeuralClawError
    ├── AgentError
    │   ├── PlanError
    │   ├── TurnTimeoutError
    │   └── IterationLimitError
    ├── SkillError
    │   ├── SkillNotFoundError
    │   ├── SkillTimeoutError
    │   ├── SkillValidationError
    │   └── SkillDisabledError
    ├── SafetyError
    │   ├── CapabilityDeniedError
    │   ├── CommandNotAllowedError
    │   └── ConfirmationDeniedError
    ├── MemoryError
    │   ├── MemoryNotInitializedError
    │   └── MemoryStoreError
    └── LLMError  (re-exported from brain for convenience)
        ├── LLMConnectionError
        ├── LLMRateLimitError
        ├── LLMContextError
        └── LLMInvalidRequestError
"""

from __future__ import annotations


# ─────────────────────────────────────────────────────────────────────────────
# Root
# ─────────────────────────────────────────────────────────────────────────────

class NeuralClawError(Exception):
    """Base class for all NeuralClaw exceptions."""


# ─────────────────────────────────────────────────────────────────────────────
# Agent layer
# ─────────────────────────────────────────────────────────────────────────────

class AgentError(NeuralClawError):
    """Base for agent orchestration errors."""


class PlanError(AgentError):
    """Failed to produce a valid plan for the given goal."""


class TurnTimeoutError(AgentError):
    """A single agent turn exceeded its maximum allowed time."""


class IterationLimitError(AgentError):
    """The agent loop hit max_iterations without reaching a terminal state."""


# ─────────────────────────────────────────────────────────────────────────────
# Skill layer
# ─────────────────────────────────────────────────────────────────────────────

class SkillError(NeuralClawError):
    """Base for all skill-related errors."""


class SkillNotFoundError(SkillError):
    """Requested skill is not registered in the SkillRegistry."""


class SkillTimeoutError(SkillError):
    """Skill execution exceeded its configured timeout_seconds."""


class SkillValidationError(SkillError):
    """Skill arguments failed validation (JSON Schema or semantic checks)."""


class SkillDisabledError(SkillError):
    """Skill is registered but has been disabled in configuration."""


# ─────────────────────────────────────────────────────────────────────────────
# Safety layer
# ─────────────────────────────────────────────────────────────────────────────

class SafetyError(NeuralClawError):
    """Base for all safety/permission errors."""


class CapabilityDeniedError(SafetyError):
    """Skill requires a capability not granted to the current session."""

    def __init__(self, skill: str, capability: str, message: str = "") -> None:
        self.skill = skill
        self.capability = capability
        super().__init__(
            message or f"Skill '{skill}' requires capability '{capability}' which has not been granted."
        )


class CommandNotAllowedError(SafetyError):
    """Shell command is not on the whitelist."""

    def __init__(self, command: str, message: str = "") -> None:
        self.command = command
        super().__init__(message or f"Command not allowed by whitelist: '{command}'")


class ConfirmationDeniedError(SafetyError):
    """User denied the confirmation prompt for a high-risk action."""


# ─────────────────────────────────────────────────────────────────────────────
# Memory layer
# ─────────────────────────────────────────────────────────────────────────────

class MemoryError(NeuralClawError):
    """Base for memory subsystem errors."""


class MemoryNotInitializedError(MemoryError):
    """MemoryManager.init() has not been called before first use."""


class MemoryStoreError(MemoryError):
    """A memory store operation (read or write) failed."""


# ─────────────────────────────────────────────────────────────────────────────
# LLM layer  (re-exported here for convenience — source of truth is brain/)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from brain.llm_client import (  # noqa: F401 — re-export
        LLMError,
        LLMConnectionError,
        LLMRateLimitError,
        LLMContextError,
        LLMInvalidRequestError,
    )
except ImportError:
    # Fallback so exceptions.py is importable even without the brain package
    class LLMError(NeuralClawError):  # type: ignore[no-redef]
        """LLM provider error."""

    class LLMConnectionError(LLMError):  # type: ignore[no-redef]
        """Network / connection failure to LLM provider."""

    class LLMRateLimitError(LLMError):  # type: ignore[no-redef]
        """LLM provider rate limit hit."""

    class LLMContextError(LLMError):  # type: ignore[no-redef]
        """Input too long for model context window."""

    class LLMInvalidRequestError(LLMError):  # type: ignore[no-redef]
        """Malformed request rejected by LLM provider."""


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: all public names
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    "NeuralClawError",
    # Agent
    "AgentError",
    "PlanError",
    "TurnTimeoutError",
    "IterationLimitError",
    # Skill
    "SkillError",
    "SkillNotFoundError",
    "SkillTimeoutError",
    "SkillValidationError",
    "SkillDisabledError",
    # Safety
    "SafetyError",
    "CapabilityDeniedError",
    "CommandNotAllowedError",
    "ConfirmationDeniedError",
    # Memory
    "MemoryError",
    "MemoryNotInitializedError",
    "MemoryStoreError",
    # LLM (re-exported)
    "LLMError",
    "LLMConnectionError",
    "LLMRateLimitError",
    "LLMContextError",
    "LLMInvalidRequestError",
]
"""
skills/base.py — SkillBase Abstract Base Class

Every NeuralClaw skill must subclass SkillBase and declare a ClassVar manifest.

Rules for skill authors:
  1. Declare `manifest: ClassVar[SkillManifest]` — static, not per-instance.
  2. Implement `async execute(**kwargs) -> SkillResult`.
  3. execute() must NEVER raise — catch all exceptions and return SkillResult.fail().
  4. Override validate() for argument-level pre-checks that go beyond JSON Schema.
  5. Skills are stateless — do not store call-specific state on self.
  6. Drop your skill file into skills/plugins/ — no core changes required.

Example:
    class GreetSkill(SkillBase):
        manifest = SkillManifest(
            name="greet",
            version="1.0.0",
            description="Return a greeting for a given name.",
            category="demo",
            risk_level=RiskLevel.LOW,
            capabilities=frozenset(),
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        )

        async def execute(self, name: str) -> SkillResult:
            return SkillResult.ok(
                skill_name=self.manifest.name,
                skill_call_id="",
                output=f"Hello, {name}!",
            )
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from neuralclaw.skills.types import SkillManifest, SkillResult, SkillValidationError


class SkillBase(ABC):
    """
    Abstract base class for all NeuralClaw skills.

    Subclass this, declare a `manifest` ClassVar, and implement `execute()`.
    The SkillLoader will discover and validate your subclass automatically.
    """

    manifest: ClassVar[SkillManifest]

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def validate(self, **kwargs) -> None:
        """
        Optional pre-execution argument validation beyond JSON Schema.

        Raise SkillValidationError with a clear message if arguments are
        semantically invalid (e.g. URL doesn't start with https, path contains
        traversal sequences, etc.).

        The SkillBus calls this before execute(). A SkillValidationError here
        returns a SkillResult.fail() — execute() is not called.

        Default implementation does nothing.
        """
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> SkillResult:
        """
        Execute the skill and return a SkillResult.

        MUST:
          - Be async.
          - Never raise — catch all exceptions, return SkillResult.fail().
          - Accept keyword arguments matching the manifest.parameters schema.
          - Return SkillResult.ok() on success, SkillResult.fail() on error.

        The skill_call_id for SkillResult should be passed in from the caller.
        Use kwargs.get("_skill_call_id", "") for the id if needed — the bus
        injects it before calling execute().
        """
        ...

    # ── Introspection ─────────────────────────────────────────────────────────

    @classmethod
    def skill_name(cls) -> str:
        return cls.manifest.name

    @classmethod
    def to_llm_schema(cls) -> dict:
        """Return the JSON schema dict for this skill in LLM tool-call format."""
        return cls.manifest.to_llm_schema()

    def __repr__(self) -> str:
        name = getattr(self.__class__, "manifest", None)
        n = name.name if name else self.__class__.__name__
        return f"<Skill:{n}>"

    # ── Class-level validation (called by SkillLoader) ────────────────────────

    @classmethod
    def _validate_manifest(cls) -> None:
        """
        Called by SkillLoader during discovery to validate the manifest is
        properly formed. Raises SkillValidationError with a descriptive message.
        """
        if not hasattr(cls, "manifest"):
            raise SkillValidationError(
                f"{cls.__name__} is missing a 'manifest' ClassVar. "
                f"Add: manifest = SkillManifest(...) as a class attribute."
            )

        m = cls.manifest
        if not isinstance(m, SkillManifest):
            raise SkillValidationError(
                f"{cls.__name__}.manifest must be a SkillManifest instance, "
                f"got {type(m).__name__}."
            )

        if not m.name or not m.name.replace("_", "").isalnum():
            raise SkillValidationError(
                f"{cls.__name__}.manifest.name '{m.name}' is invalid. "
                f"Must be non-empty and snake_case (letters, digits, underscores only)."
            )

        if not m.version or not _is_semver(m.version):
            raise SkillValidationError(
                f"{cls.__name__}.manifest.version '{m.version}' is not a valid "
                f"semver string. Use 'MAJOR.MINOR.PATCH' format (e.g. '1.0.0')."
            )

        if not m.description.strip():
            raise SkillValidationError(
                f"{cls.__name__}.manifest.description is empty. "
                f"Provide a description so the LLM knows when to use this skill."
            )

        if not m.category.strip():
            raise SkillValidationError(
                f"{cls.__name__}.manifest.category is empty. "
                f"Use a category like 'filesystem', 'terminal', 'web', or 'custom'."
            )

        if not isinstance(m.capabilities, frozenset):
            raise SkillValidationError(
                f"{cls.__name__}.manifest.capabilities must be a frozenset, "
                f"got {type(m.capabilities).__name__}."
            )

        if not isinstance(m.parameters, dict):
            raise SkillValidationError(
                f"{cls.__name__}.manifest.parameters must be a dict (JSON Schema), "
                f"got {type(m.parameters).__name__}."
            )


def _is_semver(version: str) -> bool:
    """Basic semver check: 'X.Y.Z' where X, Y, Z are non-negative integers."""
    parts = version.split(".")
    if len(parts) != 3:
        return False
    return all(p.isdigit() for p in parts)
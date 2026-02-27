"""
skills/registry.py — Skill Registry

Maps skill names to their SkillBase instances and manifests.
Populated by SkillLoader at startup. Read-only at runtime.

Usage:
    registry = SkillRegistry()
    registry.register(MySkill())

    skill = registry.get("web_search")
    manifest = registry.get_manifest("web_search")
    all_manifests = registry.list_manifests()
"""

from __future__ import annotations

from typing import Optional

from skills.base import SkillBase
from skills.types import SkillManifest, SkillNotFoundError


class SkillRegistry:
    """
    Central read/write store for registered skills.

    Thread-safe for reads. Not designed for concurrent writes.
    All writes happen at startup via SkillLoader.
    """

    def __init__(self) -> None:
        self._skills: dict[str, SkillBase] = {}
        self._manifests: dict[str, SkillManifest] = {}

    # ── Write (startup only) ──────────────────────────────────────────────────

    def register(self, skill_instance: SkillBase) -> None:
        """Register a skill instance. Raises ValueError on duplicate name."""
        name = skill_instance.manifest.name
        if name in self._skills:
            raise ValueError(
                f"Skill '{name}' is already registered. "
                f"Skill names must be globally unique across all skill files."
            )
        self._skills[name] = skill_instance
        self._manifests[name] = skill_instance.manifest

    def unregister(self, name: str) -> None:
        """Remove a skill (used in tests; not for production use)."""
        self._skills.pop(name, None)
        self._manifests.pop(name, None)

    # ── Read (runtime) ────────────────────────────────────────────────────────

    def get(self, name: str) -> SkillBase:
        """Return the skill instance. Raises SkillNotFoundError if not found."""
        if name not in self._skills:
            available = sorted(self._skills.keys())
            raise SkillNotFoundError(
                f"Skill '{name}' is not registered. "
                f"Available skills: {available}"
            )
        return self._skills[name]

    def get_or_none(self, name: str) -> Optional[SkillBase]:
        """Return the skill instance or None if not found."""
        return self._skills.get(name)

    def get_manifest(self, name: str) -> Optional[SkillManifest]:
        """Return the manifest for a skill, or None."""
        return self._manifests.get(name)

    def is_registered(self, name: str) -> bool:
        return name in self._skills

    def list_manifests(
        self,
        enabled_only: bool = True,
        granted: frozenset[str] | None = None,
    ) -> list[SkillManifest]:
        """
        Return all manifests, optionally filtering to enabled skills and
        skills that fit the granted capabilities.
        Empty capabilities mean the skill is always available.
        """
        manifests = list(self._manifests.values())
        if enabled_only:
            manifests = [m for m in manifests if m.enabled]
        
        if granted is not None:
            filtered = []
            for m in manifests:
                if not m.capabilities:
                    filtered.append(m)
                elif m.capabilities.issubset(granted):
                    filtered.append(m)
            manifests = filtered
            
        return manifests

    def list_names(self, enabled_only: bool = True) -> list[str]:
        return [m.name for m in self.list_manifests(enabled_only)]

    def to_llm_schemas(self) -> list[dict]:
        """Return all enabled skills as LLM tool-call schema dicts."""
        return [m.to_llm_schema() for m in self.list_manifests(enabled_only=True)]

    # ── Orchestrator-compat aliases (mirrors ToolRegistry API) ────────────────

    def list_schemas(
        self,
        enabled_only: bool = True,
        granted: frozenset[str] | None = None,
    ) -> list[SkillManifest]:
        """Alias for list_manifests(). Keeps orchestrator API surface identical."""
        return self.list_manifests(enabled_only=enabled_only, granted=granted)

    def get_schema(self, name: str) -> Optional[SkillManifest]:
        """Alias for get_manifest(). Returns None (not raise) for unknown names."""
        return self.get_manifest(name)

    def __len__(self) -> int:
        return len(self._skills)

    def __repr__(self) -> str:
        return f"<SkillRegistry skills={sorted(self._skills.keys())}>"
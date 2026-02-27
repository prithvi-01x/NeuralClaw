"""
skills/md_loader.py — Markdown Skill Loader

Scans directories for SKILL.md files (OpenClaw-compatible format) and
registers each one as a MarkdownSkill in the SkillRegistry.

Supported file layouts:
    skills/plugins/my_skill.md              ← single file
    skills/plugins/my_skill/SKILL.md        ← folder layout (OpenClaw standard)

SKILL.md frontmatter (YAML between --- delimiters):

    ---
    name: my_skill                    # required, snake_case
    description: What this skill does # required
    version: 1.0.0                    # required, semver
    category: web                     # optional, default: "custom"
    risk_level: LOW                   # optional, default: LOW
    capabilities: ["net:fetch"]       # optional, list of capability strings
    requires_confirmation: false      # optional, default: false
    enabled: true                     # optional, default: true
    ---

    Everything below the closing --- is the skill's instruction body.
    This gets injected into the LLM system prompt when the skill is active.

Compatible with OpenClaw's AgentSkills format — you can copy any
OpenClaw SKILL.md directly into skills/plugins/ and it will load.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from skills.md_skill import MarkdownSkill
from skills.registry import SkillRegistry
from skills.types import RiskLevel, SkillManifest, SkillValidationError

from observability.logger import portable_log

_log_msg = portable_log(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Frontmatter parser
# ─────────────────────────────────────────────────────────────────────────────

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)", re.DOTALL)

_RISK_MAP = {
    "low":      RiskLevel.LOW,
    "medium":   RiskLevel.MEDIUM,
    "high":     RiskLevel.HIGH,
    "critical": RiskLevel.CRITICAL,
}


import yaml

def _parse_md(text: str) -> tuple[dict, str]:
    """
    Parse a SKILL.md file into (frontmatter_dict, body_text).
    Uses PyYAML for robust parsing of the frontmatter block.
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text.strip()

    raw_fm, body = match.group(1), match.group(2).strip()
    
    try:
        fm = yaml.safe_load(raw_fm)
        if not isinstance(fm, dict):
            fm = {}
    except yaml.YAMLError as exc:
        _log_msg("warning", "md_skill.yaml_parse_failed", error=str(exc))
        fm = {}

    return fm, body


def _build_manifest(fm: dict, source_path: Path) -> SkillManifest:
    """Build a SkillManifest from parsed frontmatter."""

    name = fm.get("name", "").strip()
    if not name:
        # Derive from filename
        stem = source_path.stem if source_path.suffix == ".md" else source_path.parent.name
        name = re.sub(r"[^a-zA-Z0-9_]", "_", stem).lower().strip("_")

    if not name or not re.match(r"^[a-zA-Z0-9_]+$", name):
        raise SkillValidationError(
            f"Invalid skill name '{name}' in {source_path}. "
            "Must be snake_case (letters, digits, underscores only)."
        )

    description = fm.get("description", "").strip()
    if not description:
        raise SkillValidationError(
            f"Missing 'description' in {source_path}. "
            "Add a description so the LLM knows when to use this skill."
        )

    version = fm.get("version", "1.0.0").strip()
    if not re.match(r"^\d+\.\d+\.\d+$", version):
        version = "1.0.0"

    risk_raw = str(fm.get("risk_level", "LOW")).lower().strip()
    risk_level = _RISK_MAP.get(risk_raw, RiskLevel.LOW)

    caps_raw = fm.get("capabilities", [])
    if isinstance(caps_raw, str):
        caps_raw = [caps_raw] if caps_raw else []
    capabilities = frozenset(c.strip() for c in caps_raw if c.strip())

    return SkillManifest(
        name=name,
        version=version,
        description=description,
        category=str(fm.get("category", "custom")).strip() or "custom",
        risk_level=risk_level,
        capabilities=capabilities,
        requires_confirmation=bool(fm.get("requires_confirmation", False)),
        enabled=bool(fm.get("enabled", True)),
        parameters={},          # MD skills accept no direct parameters
        timeout_seconds=int(fm.get("timeout_seconds", 30)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────

class MarkdownSkillLoader:
    """
    Discovers and loads SKILL.md files from one or more directories.

    Supports two layouts:
      - Flat:   skills/plugins/my_skill.md
      - Folder: skills/plugins/my_skill/SKILL.md  (OpenClaw standard)

    Call load_all() once at startup alongside SkillLoader.load_all().
    """

    def load_all(
        self,
        skill_dirs: list[Path],
        registry: Optional[SkillRegistry] = None,
        strict: bool = False,
    ) -> SkillRegistry:
        """
        Discover and register all SKILL.md files found in skill_dirs.

        Args:
            skill_dirs: Directories to scan.
            registry:   Existing registry to populate. Creates new if None.
            strict:     If True, any parse error raises instead of warning+skip.

        Returns:
            Populated SkillRegistry (same object as input, or new).
        """
        reg = registry or SkillRegistry()
        loaded = 0
        skipped = 0

        for skill_dir in skill_dirs:
            skill_dir = Path(skill_dir)
            if not skill_dir.exists():
                continue

            md_files = self._discover(skill_dir)

            for md_path in md_files:
                try:
                    skill_cls = self._load_md_file(md_path)
                    reg.register(skill_cls())
                    _log_msg(
                        "debug",
                        "md_skill_loader.registered",
                        skill=skill_cls.manifest.name,
                        file=str(md_path),
                    )
                    loaded += 1
                except (SkillValidationError, ValueError) as e:
                    if strict:
                        raise SkillValidationError(
                            f"Failed to load markdown skill from {md_path}: {e}"
                        ) from e
                    _log_msg(
                        "warning",
                        "md_skill_loader.skipped",
                        file=str(md_path),
                        error=str(e),
                    )
                    skipped += 1

        _log_msg(
            "info",
            "md_skill_loader.complete",
            loaded=loaded,
            skipped=skipped,
            dirs=[str(d) for d in skill_dirs],
        )
        return reg

    def _discover(self, skill_dir: Path) -> list[Path]:
        """Find all SKILL.md files in a directory."""
        seen: set[Path] = set()
        found: list[Path] = []

        def _add(p: Path) -> None:
            resolved = p.resolve()
            if resolved not in seen:
                seen.add(resolved)
                found.append(p)

        # Flat: skills/plugins/my_skill.md
        for f in sorted(skill_dir.glob("*.md")):
            if not f.name.startswith("_"):
                _add(f)

        # Folder: skills/plugins/my_skill/SKILL.md
        for subdir in sorted(skill_dir.iterdir()):
            if subdir.is_dir() and not subdir.name.startswith("_"):
                skill_md = subdir / "SKILL.md"
                if skill_md.exists():
                    _add(skill_md)
                else:
                    # Also accept lowercase skill.md
                    skill_md_lower = subdir / "skill.md"
                    if skill_md_lower.exists():
                        _add(skill_md_lower)

        return found

    def _load_md_file(self, md_path: Path) -> type[MarkdownSkill]:
        """
        Parse a SKILL.md file and return a MarkdownSkill subclass.

        The returned class has:
          - manifest: SkillManifest built from frontmatter
          - instructions: str — the body text below the frontmatter
        """
        try:
            text = md_path.read_text(encoding="utf-8")
        except OSError as e:
            raise SkillValidationError(f"Cannot read {md_path}: {e}") from e

        fm, body = _parse_md(text)
        manifest = _build_manifest(fm, md_path)

        # Build a unique class name from skill name
        class_name = "".join(
            part.capitalize() for part in manifest.name.split("_")
        ) + "MdSkill"

        skill_cls = type(
            class_name,
            (MarkdownSkill,),
            {
                "manifest": manifest,
                "instructions": body,
            },
        )

        return skill_cls  # type: ignore[return-value]

    def get_instructions_block(self, registry: SkillRegistry) -> str:
        """
        Return a formatted system-prompt block containing all active
        markdown skill instructions.

        This is called by the orchestrator to inject skill instructions
        into the LLM context each turn.
        """
        from skills.md_skill import MarkdownSkill as _MdSkill

        blocks: list[str] = []
        for manifest in registry.list_manifests(enabled_only=True):
            skill = registry.get_or_none(manifest.name)
            if skill is None or not isinstance(skill, _MdSkill):
                continue
            instructions = skill.__class__.instructions.strip()
            if not instructions:
                continue
            blocks.append(
                f"## Skill: {manifest.name}\n"
                f"_{manifest.description}_\n\n"
                f"{instructions}"
            )

        if not blocks:
            return ""

        return (
            "<skill_instructions>\n"
            + "\n\n---\n\n".join(blocks)
            + "\n</skill_instructions>"
        )
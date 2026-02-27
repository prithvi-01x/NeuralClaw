"""
skills/clawhub/bridge_parser.py — Parse ClawHub SKILL.md Files

Reads a ClawHub skill folder's SKILL.md file, extracts YAML frontmatter,
body text, and references/ folder contents, and returns a structured
ClawhubSkillManifest dataclass with execution tier detection.

Handles frontmatter key aliases: metadata.openclaw, metadata.clawdbot,
metadata.clawdis are all equivalent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

try:
    from observability.logger import get_logger as _get_logger
    _log_raw = _get_logger(__name__)
    _STRUCTLOG = True
except ImportError:
    import logging as _logging
    _log_raw = _logging.getLogger(__name__)
    _STRUCTLOG = False


def _log(level: str, event: str, **kwargs) -> None:
    if _STRUCTLOG:
        getattr(_log_raw, level)(event, **kwargs)
    else:
        extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
        getattr(_log_raw, level)("%s %s", event, extra)


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClawhubRequires:
    """Dependencies declared in the ClawHub SKILL.md frontmatter."""
    env: list[str] = field(default_factory=list)
    bins: list[str] = field(default_factory=list)
    any_bins: list[str] = field(default_factory=list)
    configs: list[str] = field(default_factory=list)


@dataclass
class ClawhubInstallDirective:
    """An auto-install instruction from the ClawHub SKILL.md."""
    kind: str            # "brew", "node", "go", "uv"
    formula: str = ""    # for brew
    package: str = ""    # for node/go/uv
    bins: list[str] = field(default_factory=list)


@dataclass
class ClawhubSkillManifest:
    """Fully parsed ClawHub skill — ready for NeuralClaw bridge registration."""

    # From frontmatter
    name: str
    description: str
    version: str
    primary_env: Optional[str] = None
    emoji: str = "🔧"
    homepage: Optional[str] = None
    requires: ClawhubRequires = field(default_factory=ClawhubRequires)
    install_directives: list[ClawhubInstallDirective] = field(default_factory=list)

    # From parsing the body
    body: str = ""
    skill_dir: Path = field(default_factory=lambda: Path("."))
    extra_files: dict[str, str] = field(default_factory=dict)

    # Derived
    execution_tier: int = 1  # 1, 2, or 3


# ─────────────────────────────────────────────────────────────────────────────
# Frontmatter regex
# ─────────────────────────────────────────────────────────────────────────────

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)", re.DOTALL)

# Binaries that indicate HTTP/API usage (Tier 2 rather than Tier 3)
_HTTP_BINS = frozenset({"curl", "wget", "http", "httpie"})

# Accepted metadata namespace keys (ClawHub accepts several aliases)
_METADATA_KEYS = ("openclaw", "clawdbot", "clawdis")


# ─────────────────────────────────────────────────────────────────────────────
# Parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_metadata(fm: dict) -> dict:
    """
    Extract the openclaw metadata block from the frontmatter.
    Checks metadata.openclaw, metadata.clawdbot, metadata.clawdis.
    Returns the inner dict, or empty dict if none found.
    """
    meta = fm.get("metadata", {})
    if not isinstance(meta, dict):
        return {}
    for key in _METADATA_KEYS:
        if key in meta and isinstance(meta[key], dict):
            return meta[key]
    return {}


def _parse_requires(ocmeta: dict) -> ClawhubRequires:
    """Parse the requires block from openclaw metadata."""
    req = ocmeta.get("requires", {})
    if not isinstance(req, dict):
        return ClawhubRequires()
    return ClawhubRequires(
        env=_ensure_list(req.get("env", [])),
        bins=_ensure_list(req.get("bins", [])),
        any_bins=_ensure_list(req.get("anyBins", req.get("any_bins", []))),
        configs=_ensure_list(req.get("configs", [])),
    )


def _parse_install_directives(ocmeta: dict) -> list[ClawhubInstallDirective]:
    """Parse install directives from openclaw metadata."""
    raw = ocmeta.get("install", [])
    if not isinstance(raw, list):
        return []
    directives = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        directives.append(ClawhubInstallDirective(
            kind=str(item.get("kind", "")).strip(),
            formula=str(item.get("formula", "")).strip(),
            package=str(item.get("package", "")).strip(),
            bins=_ensure_list(item.get("bins", [])),
        ))
    return directives


def _ensure_list(val) -> list[str]:
    """Ensure a value is a list of strings."""
    if isinstance(val, list):
        return [str(v).strip() for v in val if v]
    if isinstance(val, str) and val.strip():
        return [val.strip()]
    return []


def _detect_tier(requires: ClawhubRequires,
                 install_directives: list[ClawhubInstallDirective]) -> int:
    """
    Determine execution tier:
      1 — Prompt-only (no bins, no install)
      2 — HTTP/API (only uses curl/wget/http bins)
      3 — Binary (needs CLI binaries or install directives)
    """
    has_install = len(install_directives) > 0
    has_bins = len(requires.bins) > 0 or len(requires.any_bins) > 0

    if not has_bins and not has_install:
        return 1

    # If the only required bins are HTTP tools, it's tier 2
    all_bins = set(requires.bins) | set(requires.any_bins)
    if all_bins and all_bins.issubset(_HTTP_BINS) and not has_install:
        return 2

    return 3


def _read_references(skill_dir: Path) -> dict[str, str]:
    """Read all .md files from a references/ subdirectory, if it exists."""
    refs_dir = skill_dir / "references"
    extra: dict[str, str] = {}
    if not refs_dir.is_dir():
        return extra
    for f in sorted(refs_dir.glob("*.md")):
        try:
            extra[f.name] = f.read_text(encoding="utf-8")
        except OSError:
            _log("warning", "clawhub_parser.ref_read_error", file=str(f))
    return extra


# ─────────────────────────────────────────────────────────────────────────────
# Main parse function
# ─────────────────────────────────────────────────────────────────────────────

def parse_clawhub_skill_md(skill_dir: Path) -> ClawhubSkillManifest:
    """
    Parse a ClawHub skill folder and return a ClawhubSkillManifest.

    Args:
        skill_dir: Path to a directory containing a SKILL.md file.

    Returns:
        Fully populated ClawhubSkillManifest.

    Raises:
        FileNotFoundError: if no SKILL.md exists in skill_dir.
        ValueError: if the frontmatter is missing critical fields.
    """
    # Find SKILL.md (case-insensitive)
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        skill_md = skill_dir / "skill.md"
    if not skill_md.exists():
        raise FileNotFoundError(f"No SKILL.md found in {skill_dir}")

    text = skill_md.read_text(encoding="utf-8")

    # Split frontmatter and body
    match = _FRONTMATTER_RE.match(text)
    if not match:
        # No frontmatter — treat entire file as body with defaults
        return ClawhubSkillManifest(
            name=_sanitize_name(skill_dir.name),
            description=f"ClawHub skill: {skill_dir.name}",
            version="1.0.0",
            body=text.strip(),
            skill_dir=skill_dir,
            execution_tier=1,
        )

    raw_fm, body = match.group(1), match.group(2).strip()

    try:
        fm = yaml.safe_load(raw_fm)
        if not isinstance(fm, dict):
            fm = {}
    except yaml.YAMLError as exc:
        _log("warning", "clawhub_parser.yaml_error", error=str(exc),
             skill=str(skill_dir))
        fm = {}

    # Extract openclaw metadata (handles aliases)
    ocmeta = _extract_metadata(fm)

    # Parse requirements and install directives
    requires = _parse_requires(ocmeta)
    install_directives = _parse_install_directives(ocmeta)

    # Tier detection
    tier = _detect_tier(requires, install_directives)

    # Read extra reference files
    extra_files = _read_references(skill_dir)

    # Append reference content to body for full LLM context
    full_body = body
    if extra_files:
        ref_blocks = []
        for fname, content in extra_files.items():
            ref_blocks.append(f"\n\n---\n## Reference: {fname}\n\n{content}")
        full_body = body + "".join(ref_blocks)

    # Build the manifest
    name = str(fm.get("name", "") or skill_dir.name).strip()
    description = str(fm.get("description", "") or f"ClawHub skill: {name}").strip()
    version = str(fm.get("version", "1.0.0") or "1.0.0").strip()
    primary_env = ocmeta.get("primaryEnv") or ocmeta.get("primary_env")
    emoji = str(ocmeta.get("emoji", "🔧") or "🔧")
    homepage = ocmeta.get("homepage")

    return ClawhubSkillManifest(
        name=name,
        description=description,
        version=version,
        primary_env=primary_env,
        emoji=emoji,
        homepage=homepage,
        requires=requires,
        install_directives=install_directives,
        body=full_body,
        skill_dir=skill_dir,
        extra_files=extra_files,
        execution_tier=tier,
    )


def _sanitize_name(raw: str) -> str:
    """Convert a raw skill name/slug to snake_case for NeuralClaw."""
    name = re.sub(r"[^a-zA-Z0-9_]", "_", raw).lower().strip("_")
    # Collapse multiple underscores
    name = re.sub(r"_+", "_", name)
    return name or "unknown_skill"

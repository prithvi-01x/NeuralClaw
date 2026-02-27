"""
skills/discovery.py — Skill Discovery Helpers

Shared logic for skill search, detail views, category grouping, and
grant hints — used by both CLI and Telegram interfaces.

U1: fuzzy_search_skills()  — fuzzy match by name + description
U2: skill_detail()         — full detail view for a single skill
U3: group_by_category()    — group manifests by category
U4: skill_type_icon()      — 🔧 callable vs 📄 markdown
U5: missing_grant_hints()  — ungranted caps with ≥1 associated skill
U7: capability_hint()      — auto-suggest /grant after blocked call
"""

from __future__ import annotations

from typing import Optional

from neuralclaw.skills.types import SkillManifest


# ── U4: Skill Type Indicator ─────────────────────────────────────────────────

def skill_type_icon(manifest: SkillManifest) -> str:
    """Return 🔧 for callable skills, 📄 for markdown/instruction skills."""
    # Markdown skills have empty parameters (no JSON schema properties)
    params = manifest.parameters
    has_properties = bool(params.get("properties")) if isinstance(params, dict) else False
    # If it has properties defined, it's a callable skill; otherwise markdown/instruction
    if has_properties or manifest.category == "terminal":
        return "🔧"
    return "📄"


# ── U1: Fuzzy Skill Search ──────────────────────────────────────────────────

def _token_overlap_score(query: str, text: str) -> float:
    """
    Simple token-overlap scorer: ratio of query tokens found in text.
    Returns 0.0–1.0.  No external dependencies needed.
    """
    q_tokens = set(query.lower().split())
    t_tokens = set(text.lower().split())
    if not q_tokens:
        return 0.0
    return len(q_tokens & t_tokens) / len(q_tokens)


def fuzzy_search_skills(
    manifests: list[SkillManifest],
    query: str,
    threshold: float = 0.3,
) -> list[SkillManifest]:
    """
    Return manifests matching the query by substring or token overlap.

    Matching strategy (lightweight, no external deps):
      1. Exact substring in name or description → score 1.0
      2. Token overlap between query and (name + description) → 0.0–1.0
    Results are sorted by score descending, filtered by threshold.
    """
    query_lower = query.lower().strip()
    if not query_lower:
        return manifests

    scored: list[tuple[float, SkillManifest]] = []
    for m in manifests:
        name_lower = m.name.lower()
        desc_lower = m.description.lower()
        combined = f"{name_lower} {desc_lower}"

        # Exact substring match → high score
        if query_lower in name_lower:
            score = 1.0
        elif query_lower in desc_lower:
            score = 0.8
        else:
            score = _token_overlap_score(query_lower, combined)

        if score >= threshold:
            scored.append((score, m))

    scored.sort(key=lambda x: (-x[0], x[1].name))
    return [m for _, m in scored]


# ── U2: Skill Detail View ───────────────────────────────────────────────────

def skill_detail(manifest: SkillManifest) -> str:
    """
    Build a multi-line detail view for a skill.
    Shows: icon, name, full description, parameters, risk, capabilities.
    """
    icon = skill_type_icon(manifest)
    lines = [
        f"{icon} **{manifest.name}** v{manifest.version}",
        f"",
        f"{manifest.description}",
        f"",
        f"**Category:** {manifest.category}",
        f"**Risk Level:** {manifest.risk_level.value}",
    ]

    # Parameters
    params = manifest.parameters
    properties = params.get("properties", {}) if isinstance(params, dict) else {}
    required_params = set(params.get("required", [])) if isinstance(params, dict) else set()

    if properties:
        lines.append("")
        lines.append("**Parameters:**")
        for pname, pschema in properties.items():
            ptype = pschema.get("type", "any")
            pdesc = pschema.get("description", "")
            req_marker = " *(required)*" if pname in required_params else ""
            lines.append(f"  • `{pname}` ({ptype}){req_marker} — {pdesc}")
    else:
        lines.append("")
        lines.append("**Parameters:** None (instruction/markdown skill)")

    # Capabilities
    if manifest.capabilities:
        lines.append("")
        lines.append(f"**Required Capabilities:** {', '.join(sorted(manifest.capabilities))}")

    if manifest.requires_confirmation:
        lines.append("")
        lines.append("⚠️ Requires user confirmation before execution.")

    return "\n".join(lines)


# ── U3: Category Grouping ───────────────────────────────────────────────────

def group_by_category(
    manifests: list[SkillManifest],
) -> dict[str, list[SkillManifest]]:
    """Group manifests by category, sorted alphabetically within each group."""
    groups: dict[str, list[SkillManifest]] = {}
    for m in manifests:
        groups.setdefault(m.category, []).append(m)
    # Sort skills within each group
    for skills in groups.values():
        skills.sort(key=lambda s: s.name)
    return dict(sorted(groups.items()))


# ── U5: Missing Grant Hints ─────────────────────────────────────────────────

def missing_grant_hints(
    manifests: list[SkillManifest],
    granted_capabilities: frozenset[str],
) -> list[str]:
    """
    Return capability strings that are:
      1. Required by at least one registered skill
      2. Not currently granted in this session

    Used to suggest /grant commands to the user after /tools.
    """
    all_caps: set[str] = set()
    for m in manifests:
        all_caps.update(m.capabilities)
    missing = sorted(all_caps - set(granted_capabilities))
    return missing


# ── U7: Auto-Suggest on Blocked ─────────────────────────────────────────────

def capability_hint_from_reason(reason: str) -> Optional[str]:
    """
    Parse a safety-blocked reason string and extract the missing capabilities.
    Returns a user-facing hint like:  "Run /grant fs:read to enable this."
    Returns None if the reason doesn't indicate a capability block.

    Expected format from safety_kernel.py:
      "Skill 'foo' requires capabilities ['fs:read', 'fs:write'] that have not been granted..."
    """
    import re
    match = re.search(r"requires capabilities \[([^\]]+)\]", reason)
    if not match:
        return None
    # Parse the capability list (they're in Python repr format: 'cap1', 'cap2')
    caps_raw = match.group(1)
    caps = [c.strip().strip("'\"") for c in caps_raw.split(",")]
    if not caps:
        return None
    grant_cmds = " ".join(f"/grant {c}" for c in caps)
    return f"💡 Run {grant_cmds} to enable this skill."

"""
skills/clawhub/env_injector.py — Environment Variable Injection for ClawHub Skills

Validates that required env vars are set and provides a clean interface
for injecting them into skill execution contexts.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuralclaw.skills.clawhub.bridge_parser import ClawhubRequires


def validate_env(requires: "ClawhubRequires") -> tuple[bool, list[str]]:
    """
    Validate that all required environment variables are set.

    Returns:
        (all_ok, list_of_missing_var_names)
    """
    missing = [v for v in requires.env if not os.environ.get(v)]
    return (len(missing) == 0, missing)


def get_env_values(requires: "ClawhubRequires") -> dict[str, str]:
    """
    Return a dict of env var name → value for all declared requirements.
    Only includes vars that are actually set.
    """
    return {
        v: os.environ[v]
        for v in requires.env
        if v in os.environ
    }


def format_missing_env_message(
    skill_name: str,
    missing: list[str],
    primary_env: str | None = None,
) -> str:
    """
    Build a user-friendly error message for missing env vars.
    """
    lines = [f"⚠  Skill '{skill_name}' requires the following environment variables:"]
    for v in missing:
        hint = " (primary)" if v == primary_env else ""
        lines.append(f"   • {v}{hint}")
    lines.append("")
    lines.append("Add them to your .env file and restart NeuralClaw:")
    for v in missing:
        lines.append(f"   {v}=your_value_here")
    return "\n".join(lines)

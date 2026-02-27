"""
skills/md_skill.py — Markdown Skill

A lightweight skill type loaded from a SKILL.md file.
Unlike Python skills (which implement execute()), markdown skills
work by injecting their instructions directly into the LLM system prompt.
The LLM then uses existing tools (web_fetch, terminal, filesystem, etc.)
to carry out whatever the skill describes.

This mirrors how OpenClaw skills work — a SKILL.md teaches the agent
what to do; no Python code required.

Usage:
    Drop a SKILL.md into skills/plugins/<skill-name>/
    or directly as skills/plugins/<skill-name>.md

    The agent will read the instructions and know how to use the skill
    the next time it's relevant.

SKILL.md format:
    ---
    name: my_skill
    description: One-line description the LLM uses to decide when to activate
    version: 1.0.0
    category: web
    risk_level: LOW          # LOW | MEDIUM | HIGH | CRITICAL (default: LOW)
    capabilities: []         # optional list of required capabilities
    requires_confirmation: false
    enabled: true
    ---

    ## Instructions

    Full natural-language instructions for the LLM...
"""

from __future__ import annotations

from typing import ClassVar

from neuralclaw.skills.base import SkillBase
from neuralclaw.skills.types import RiskLevel, SkillManifest, SkillResult


class MarkdownSkill(SkillBase):
    """
    A skill defined entirely in a SKILL.md file.

    The execute() method is never called by the LLM directly —
    markdown skills work by having their instructions injected into
    the system prompt. This class exists so the skill integrates
    cleanly with the SkillRegistry, SafetyKernel, and /tools display.
    """

    manifest: ClassVar[SkillManifest]

    # The raw markdown instructions from the SKILL.md body (below frontmatter)
    instructions: ClassVar[str] = ""

    async def execute(self, **kwargs) -> SkillResult:
        """
        Markdown skills don't execute directly — they guide the LLM
        via system prompt injection. This method returns the instructions
        as output in case the orchestrator calls it directly.
        """
        call_id = kwargs.get("_skill_call_id", "")
        return SkillResult.ok(
            skill_name=self.manifest.name,
            skill_call_id=call_id,
            output={
                "type": "markdown_skill",
                "instructions": self.__class__.instructions,
                "message": (
                    f"Skill '{self.manifest.name}' is a markdown skill. "
                    "Its instructions are injected into the system prompt "
                    "to guide your behaviour."
                ),
            },
        )
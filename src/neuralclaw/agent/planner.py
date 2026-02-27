"""
agent/planner.py — Task Planner

Decomposes a high-level goal into an ordered step list using the LLM.
Used in autonomous mode. Also handles recovery planning when a step fails.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from neuralclaw.brain.llm_client import BaseLLMClient
from neuralclaw.brain.types import LLMConfig, Message
from neuralclaw.observability.logger import get_logger

log = get_logger(__name__)

_PLAN_SYSTEM = """\
You are a task planner for an autonomous AI agent.
Break the goal into 3-7 concrete, ordered, single-action steps.
Each step should name the tool it will likely use (if any).
Return ONLY valid JSON — no markdown fences, no explanation.

Available tools: {tool_list}

Required format:
{{"steps": ["Step 1 description", "Step 2 description", ...],
  "estimated_duration": "30 seconds | 2 minutes | 10 minutes",
  "risk_level": "LOW | MEDIUM | HIGH"}}"""

_RECOVERY_SYSTEM = """\
You are a recovery planner. A step has failed. Propose recovery steps to insert
so execution can continue, or confirm recovery is impossible.
Return ONLY valid JSON — no markdown fences.

Required format:
{{"recovery_steps": ["Recovery step 1", ...],
  "can_recover": true,
  "skip_failed_step": true}}"""


class PlanResult:
    def __init__(self, steps: list[str], estimated_duration: str = "", risk_level: str = "MEDIUM"):
        self.steps = steps
        self.estimated_duration = estimated_duration
        self.risk_level = risk_level

    def __repr__(self) -> str:
        return f"<PlanResult steps={len(self.steps)} risk={self.risk_level}>"


class RecoveryResult:
    def __init__(self, recovery_steps: list[str], can_recover: bool, skip_failed_step: bool = False):
        self.recovery_steps = recovery_steps
        self.can_recover = can_recover
        self.skip_failed_step = skip_failed_step


class Planner:
    """Uses the LLM to decompose goals into step sequences."""

    def __init__(self, llm_client: BaseLLMClient, llm_config: LLMConfig):
        self._llm = llm_client
        # Low temperature for deterministic plans, short output
        self._config = LLMConfig(
            model=llm_config.model,
            temperature=0.2,
            max_tokens=512,
        )

    async def create_plan(
        self,
        goal: str,
        available_tools: list[str],
        context: str = "",
    ) -> PlanResult:
        """Ask the LLM to break the goal into steps."""
        tool_list = ", ".join(available_tools) if available_tools else "none"
        system = _PLAN_SYSTEM.format(tool_list=tool_list)

        user_content = f"Goal: {goal}"
        if context:
            user_content += f"\n\nRelevant context:\n{context[:1000]}"

        messages = [Message.system(system), Message.user(user_content)]

        log.info("planner.create_plan", goal=goal[:80], tools=len(available_tools))
        try:
            response = await self._llm.generate(messages=messages, config=self._config)
            return self._parse_plan(response.content or "")
        except Exception as e:
            log.warning("planner.create_plan_failed", error=str(e), error_type=type(e).__name__)
            return PlanResult(steps=[f"Complete the following task: {goal}"])

    async def create_recovery(
        self,
        goal: str,
        failed_step: str,
        error: str,
        steps_remaining: list[str],
    ) -> RecoveryResult:
        """Given a failed step, generate recovery steps."""
        user_content = (
            f"Goal: {goal}\n"
            f"Failed step: {failed_step}\n"
            f"Error: {error}\n"
            f"Remaining steps: {json.dumps(steps_remaining)}"
        )
        messages = [Message.system(_RECOVERY_SYSTEM), Message.user(user_content)]

        log.info("planner.recovery", failed_step=failed_step[:60])
        try:
            response = await self._llm.generate(messages=messages, config=self._config)
            return self._parse_recovery(response.content or "")
        except Exception as e:
            log.warning("planner.recovery_failed", error=str(e), error_type=type(e).__name__)
            return RecoveryResult(recovery_steps=[], can_recover=False)

    # ── Parsers ───────────────────────────────────────────────────────────────

    def _parse_plan(self, content: str) -> PlanResult:
        content = _strip_fences(content)
        try:
            data = json.loads(content)
            steps = [str(s) for s in data.get("steps", []) if s]
            if not steps:
                raise ValueError("empty steps")
            return PlanResult(
                steps=steps,
                estimated_duration=data.get("estimated_duration", ""),
                risk_level=data.get("risk_level", "MEDIUM").upper(),
            )
        except (json.JSONDecodeError, ValueError) as e:
            log.warning("planner.parse_plan_failed", error=str(e), raw=content[:200])
            # Fall back to line-based extraction.
            # Use regex to strip leading step numbers like "1.", "2)", "- ", etc.
            # (lstrip("0123456789.-) ") treated its arg as a character SET, not a
            # prefix pattern, and could eat leading digits from the step text itself.)
            _STEP_PREFIX = re.compile(r"^\d+[\.\)\-]\s*|^[-*•]\s+")
            lines = [
                _STEP_PREFIX.sub("", l.strip())
                for l in content.splitlines()
                if l.strip()
            ]
            lines = [l for l in lines if l and not l.startswith("{")]
            return PlanResult(steps=lines[:10] if lines else ["Execute the task"])

    def _parse_recovery(self, content: str) -> RecoveryResult:
        content = _strip_fences(content)
        try:
            data = json.loads(content)
            return RecoveryResult(
                recovery_steps=[str(s) for s in data.get("recovery_steps", [])],
                can_recover=bool(data.get("can_recover", False)),
                skip_failed_step=bool(data.get("skip_failed_step", False)),
            )
        except (json.JSONDecodeError, ValueError) as e:
            log.warning("planner.parse_recovery_failed", error=str(e))
            return RecoveryResult(recovery_steps=[], can_recover=False)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[1:end])
    return text.strip()
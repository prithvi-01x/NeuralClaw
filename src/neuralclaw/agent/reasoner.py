"""
agent/reasoner.py — Reasoning Engine

Lightweight chain-of-thought and self-critique for the orchestrator.
Used before executing high-risk tool calls, and for post-task reflection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from neuralclaw.brain.llm_client import BaseLLMClient, LLMError
from neuralclaw.brain.types import LLMConfig, Message
from neuralclaw.exceptions import LLMConnectionError, LLMRateLimitError, LLMContextError, LLMInvalidRequestError
from neuralclaw.observability.logger import get_logger

_LLM_ERRORS = (LLMError, LLMConnectionError, LLMRateLimitError, LLMContextError, LLMInvalidRequestError, OSError)

log = get_logger(__name__)

_EVAL_SYSTEM = """\
You are a safety-conscious reasoning engine for an autonomous AI agent.
Evaluate whether the proposed tool call is correct and safe given the current goal.
Be concise. Return ONLY valid JSON — no fences.

Format:
{{"proceed": true,
  "confidence": 0.0,
  "reasoning": "one sentence",
  "concern": "optional — only if proceed=false or confidence<0.7"}}"""

_THINK_SYSTEM = """\
You are a reasoning assistant. Think through the question in 2-4 sentences.
Identify the key factors, risks, and best course of action. Do not answer yet — just reason."""

_REFLECT_SYSTEM = """\
You are reflecting on a completed task. Extract a concise lesson learned (1-2 sentences)
that would help with similar tasks in the future.
Return ONLY the lesson as plain text — no JSON, no preamble."""


@dataclass
class EvalVerdict:
    proceed: bool
    confidence: float      # 0.0 – 1.0
    reasoning: str
    concern: str = ""

    @property
    def is_confident(self) -> bool:
        return self.confidence >= 0.7


class Reasoner:
    """LLM-powered reasoning assistant for the orchestrator."""

    def __init__(self, llm_client: BaseLLMClient, llm_config: LLMConfig):
        self._llm = llm_client
        # Shared low-temperature config for structured/deterministic calls (eval, think)
        self._config = LLMConfig(
            model=llm_config.model,
            temperature=0.1,
            max_tokens=200,
        )
        # Reflection generates prose lessons — needs more room than eval/think
        self._reflect_config = LLMConfig(
            model=llm_config.model,
            temperature=0.3,
            max_tokens=400,
        )

    async def evaluate_tool_call(
        self,
        tool_name: str,
        tool_args: dict,
        goal: str,
        step: str = "",
    ) -> EvalVerdict:
        """
        Evaluate whether a proposed tool call is appropriate and safe.
        Returns EvalVerdict; if verdict.proceed is False, the orchestrator skips it.
        """
        args_text = json.dumps(tool_args, indent=2)[:400]
        user_content = (
            f"Goal: {goal}\n"
            f"Current step: {step}\n"
            f"Proposed tool: {tool_name}\n"
            f"Arguments:\n{args_text}"
        )
        messages = [Message.system(_EVAL_SYSTEM), Message.user(user_content)]

        log.debug("reasoner.evaluating", tool=tool_name, goal=goal[:60])
        try:
            response = await self._llm.generate(messages=messages, config=self._config)
            return self._parse_verdict(response.content or "")
        except _LLM_ERRORS as e:
            log.warning("reasoner.evaluate_failed", error=str(e), error_type=type(e).__name__)
            return EvalVerdict(proceed=True, confidence=0.5, reasoning="Reasoning unavailable.")

    async def think(self, question: str, context: str = "") -> str:
        """
        Brief chain-of-thought before acting. Returns reasoning text.
        Logged internally; not shown to user by default.
        """
        user_content = f"Context: {context}\n\nQuestion: {question}" if context else question
        messages = [Message.system(_THINK_SYSTEM), Message.user(user_content)]

        try:
            response = await self._llm.generate(messages=messages, config=self._config)
            reasoning = response.content or ""
            log.debug("reasoner.think", reasoning=reasoning[:150])
            return reasoning
        except _LLM_ERRORS as e:
            log.warning("reasoner.think_failed", error=str(e), error_type=type(e).__name__)
            return ""

    async def reflect(self, goal: str, steps_taken: list[str], outcome: str) -> str:
        """
        After completing a task, extract a lesson for episodic memory.
        Returns a 1-2 sentence lesson as plain text.
        """
        steps_text = "\n".join(f"- {s}" for s in steps_taken[:10])
        user_content = f"Goal: {goal}\nSteps taken:\n{steps_text}\nOutcome: {outcome}"
        messages = [Message.system(_REFLECT_SYSTEM), Message.user(user_content)]

        try:
            response = await self._llm.generate(messages=messages, config=self._reflect_config)
            lesson = (response.content or "").strip()
            log.debug("reasoner.reflect", lesson=lesson[:150])
            return lesson
        except _LLM_ERRORS as e:
            log.warning("reasoner.reflect_failed", error=str(e), error_type=type(e).__name__)
            return ""

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _parse_verdict(self, content: str) -> EvalVerdict:
        content = content.strip()
        if content.startswith("```"):
            lines = content.splitlines()
            end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
            content = "\n".join(lines[1:end]).strip()

        try:
            data = json.loads(content)
            return EvalVerdict(
                proceed=bool(data.get("proceed", True)),
                confidence=float(data.get("confidence", 0.8)),
                reasoning=str(data.get("reasoning", "")),
                concern=str(data.get("concern", "")),
            )
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            log.warning("reasoner.parse_verdict_failed", error=str(e))
            return EvalVerdict(proceed=True, confidence=0.5, reasoning="Parse failed.")
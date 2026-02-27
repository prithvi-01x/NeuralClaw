"""
agent/response_synthesizer.py — Response Synthesizer

Converts raw agent outputs (LLMResponse, SkillResult, SafetyDecision)
into AgentResponse objects that Telegram / CLI can render.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from neuralclaw.brain.types import LLMResponse
from neuralclaw.skills.types import ConfirmationRequest, RiskLevel, SkillResult


class ResponseKind(str, Enum):
    TEXT = "text"
    TOOL_RESULT = "tool_result"
    PROGRESS = "progress"
    CONFIRMATION = "confirmation"
    ERROR = "error"
    PLAN = "plan"
    STATUS = "status"


@dataclass
class AgentResponse:
    """
    Unified output object from the agent, ready for the interface to render.

    Always has `text` (display string). Extra fields are set for specific kinds.
    """
    kind: ResponseKind
    text: str
    tool_name: Optional[str] = None
    tool_call_id: Optional[str] = None         # for CONFIRMATION routing
    risk_level: Optional[RiskLevel] = None     # for CONFIRMATION display
    metadata: dict = field(default_factory=dict)
    is_final: bool = True                       # False for streaming mid-turn updates

    def __str__(self) -> str:
        return self.text


class ResponseSynthesizer:
    """Formats raw agent outputs into AgentResponse objects."""

    # ── LLM response ──────────────────────────────────────────────────────────

    def from_llm(self, response: LLMResponse) -> AgentResponse:
        return AgentResponse(
            kind=ResponseKind.TEXT,
            text=response.content or "",
            is_final=response.is_complete,
            metadata={
                "model": response.model,
                "tokens_in": response.usage.input_tokens,
                "tokens_out": response.usage.output_tokens,
                "finish_reason": response.finish_reason.value,
            },
        )

    # ── Tool results ──────────────────────────────────────────────────────────

    def tool_progress(self, tool_name: str, step: int, total: int, detail: str = "") -> AgentResponse:
        bar = self._bar(step, total)
        text = f"⚙️ **{tool_name}** {bar} ({step}/{total})"
        if detail:
            text += f"\n_{detail}_"
        return AgentResponse(kind=ResponseKind.PROGRESS, text=text,
                             tool_name=tool_name, is_final=False,
                             metadata={"step": step, "total": total})

    def tool_success(self, result: SkillResult, summary: Optional[str] = None) -> AgentResponse:
        display = summary or _clip(result.content, 400)
        text = f"✅ **{result.name}** — {display}"
        return AgentResponse(kind=ResponseKind.TOOL_RESULT, text=text,
                             tool_name=result.name,
                             metadata={"duration_ms": round(result.duration_ms, 1),
                                       "risk_level": result.risk_level.value})

    def tool_error(self, result: SkillResult) -> AgentResponse:
        text = f"❌ **{result.name}** failed\n```\n{_clip(result.content, 300)}\n```"
        return AgentResponse(kind=ResponseKind.ERROR, text=text, tool_name=result.name,
                             metadata={"risk_level": result.risk_level.value})

    # ── Confirmation ──────────────────────────────────────────────────────────

    def confirmation_request(self, confirm_req: ConfirmationRequest) -> AgentResponse:
        risk_icon = {RiskLevel.HIGH: "⚠️", RiskLevel.CRITICAL: "🚨"}.get(confirm_req.risk_level, "❓")
        args_lines = []
        for k, v in confirm_req.arguments.items():
            val = str(v)
            if len(val) > 100:
                val = val[:100] + "…"
            args_lines.append(f"  **{k}**: `{val}`")
        args_text = ("\n" + "\n".join(args_lines)) if args_lines else ""

        text = (
            f"{risk_icon} **Confirmation Required**\n\n"
            f"Tool: `{confirm_req.skill_name}`\n"
            f"Risk: **{confirm_req.risk_level.value}**\n"
            f"Reason: {confirm_req.reason}"
            f"{args_text}\n\n"
            f"Allow this action?"
        )
        return AgentResponse(
            kind=ResponseKind.CONFIRMATION,
            text=text,
            tool_name=confirm_req.skill_name,
            tool_call_id=confirm_req.skill_call_id,
            risk_level=confirm_req.risk_level,
            is_final=False,
            metadata={"reason": confirm_req.reason},
        )

    # ── Plan / status ─────────────────────────────────────────────────────────

    def plan_preview(self, goal: str, steps: list[str]) -> AgentResponse:
        step_lines = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(steps))
        text = (
            f"📋 **Plan for:** {goal}\n\n"
            f"**Steps:**\n{step_lines}\n\n"
            f"_Executing now…_"
        )
        return AgentResponse(kind=ResponseKind.PLAN, text=text, is_final=False,
                             metadata={"goal": goal, "step_count": len(steps)})

    def status(self, session) -> AgentResponse:
        s = session.status_summary()
        plan_info = ""
        if session.active_plan:
            plan_info = (f"\n📋 **Plan:** {session.active_plan.goal[:60]}… "
                         f"({session.active_plan.progress_summary})")
        text = (
            f"🤖 **NeuralClaw Status**\n\n"
            f"Session: `{s['session_id']}`\n"
            f"Trust: **{s['trust_level'].upper()}**\n"
            f"Turns: {s['turns']}  ·  Tool calls: {s['tool_calls']}\n"
            f"Tokens: {s['tokens_in']:,} in / {s['tokens_out']:,} out\n"
            f"Uptime: {s['uptime_seconds']}s"
            f"{plan_info}"
        )
        return AgentResponse(kind=ResponseKind.STATUS, text=text, metadata=s)

    def error(self, message: str, detail: str = "") -> AgentResponse:
        text = f"❌ **Error:** {message}"
        if detail:
            text += f"\n```\n{_clip(detail, 500)}\n```"
        return AgentResponse(kind=ResponseKind.ERROR, text=text)

    def info(self, message: str) -> AgentResponse:
        """Informational notice — rendered as a non-error status message."""
        return AgentResponse(kind=ResponseKind.PROGRESS, text=message, is_final=False)

    def cancelled(self) -> AgentResponse:
        return AgentResponse(kind=ResponseKind.TEXT, text="🛑 Task cancelled.")

    def thinking(self) -> AgentResponse:
        return AgentResponse(kind=ResponseKind.PROGRESS, text="🤔 Thinking…",
                             is_final=False)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _bar(step: int, total: int, width: int = 10) -> str:
        if total == 0:
            return "░" * width
        filled = int(width * step / total)
        return "█" * filled + "░" * (width - filled)


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"…(+{len(text) - max_chars} chars)"
"""
agent/session.py — Per-Session State

One Session object exists per Telegram chat / CLI user.
Holds the conversation short-term memory, active autonomous plan,
trust level, cancellation signal, and confirmation futures.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from memory.short_term import ShortTermMemory
from observability.logger import get_logger
from tools.types import TrustLevel

log = get_logger(__name__)


@dataclass
class PlanStep:
    index: int
    description: str
    tool_hint: Optional[str] = None
    completed: bool = False
    result_summary: Optional[str] = None
    error: Optional[str] = None


@dataclass
class ActivePlan:
    id: str
    goal: str
    steps: list[PlanStep]
    current_step_index: int = 0
    episode_id: Optional[str] = None

    @property
    def current_step(self) -> Optional[PlanStep]:
        if self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    @property
    def is_complete(self) -> bool:
        return self.current_step_index >= len(self.steps)

    @property
    def progress_summary(self) -> str:
        done = sum(1 for s in self.steps if s.completed)
        return f"{done}/{len(self.steps)} steps complete"

    def advance(self) -> None:
        if self.current_step_index < len(self.steps):
            self.steps[self.current_step_index].completed = True
            self.current_step_index += 1


class Session:
    """All per-user runtime state for a single agent session."""

    def __init__(
        self,
        session_id: str,
        user_id: str = "local",
        trust_level: TrustLevel = TrustLevel.LOW,
        max_turns: int = 20,
    ):
        self.id = session_id
        self.user_id = user_id
        self.trust_level = trust_level
        self.created_at = time.time()

        self.memory = ShortTermMemory(
            session_id=session_id,
            user_id=user_id,
            max_turns=max_turns,
        )

        self.active_plan: Optional[ActivePlan] = None

        # Metrics
        self.turn_count: int = 0
        self.tool_call_count: int = 0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0

        # Cooperative cancellation
        self._cancel_event = asyncio.Event()

        # tool_call_id → Future[bool] for confirmation prompts
        self._pending_confirmations: dict[str, asyncio.Future] = {}

        log.debug("session.created", session_id=session_id, user_id=user_id)

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        user_id: str = "local",
        trust_level: TrustLevel = TrustLevel.LOW,
        max_turns: int = 20,
    ) -> "Session":
        return cls(
            session_id=f"sess_{uuid.uuid4().hex[:12]}",
            user_id=user_id,
            trust_level=trust_level,
            max_turns=max_turns,
        )

    # ── Message helpers ───────────────────────────────────────────────────────

    def add_user_message(self, content: str) -> None:
        from brain.types import Message
        self.memory.add_message(Message.user(content))
        self.turn_count += 1

    def add_assistant_message(self, content: str) -> None:
        from brain.types import Message
        self.memory.add_message(Message.assistant(content))

    def add_message(self, message) -> None:
        from brain.types import Role
        self.memory.add_message(message)
        if message.role == Role.USER:
            self.turn_count += 1

    def get_messages(self, system_prompt: Optional[str] = None) -> list:
        return self.memory.get_context_messages(system_prompt=system_prompt)

    def get_recent_messages(self, n: int = 40) -> list:
        return self.memory.conversation.get_recent(n)

    def clear_conversation(self) -> None:
        self.memory.clear_conversation()

    # ── Plan ──────────────────────────────────────────────────────────────────

    def set_plan(self, goal: str, steps: list[str], episode_id: Optional[str] = None) -> ActivePlan:
        self.active_plan = ActivePlan(
            id=f"plan_{uuid.uuid4().hex[:8]}",
            goal=goal,
            steps=[PlanStep(index=i, description=d) for i, d in enumerate(steps)],
            episode_id=episode_id,
        )
        log.info("session.plan_set", session_id=self.id, goal=goal[:80], steps=len(steps))
        return self.active_plan

    def clear_plan(self) -> None:
        self.active_plan = None

    # ── Trust ─────────────────────────────────────────────────────────────────

    def set_trust_level(self, level: TrustLevel) -> None:
        old = self.trust_level
        self.trust_level = level
        log.info("session.trust_changed", session_id=self.id, old=old.value, new=level.value)

    # ── Metrics ───────────────────────────────────────────────────────────────

    def record_token_usage(self, input_tokens: int, output_tokens: int) -> None:
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    def record_tool_call(self) -> None:
        self.tool_call_count += 1

    # ── Cancellation ─────────────────────────────────────────────────────────

    def cancel(self) -> None:
        self._cancel_event.set()
        log.info("session.cancel_requested", session_id=self.id)

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def reset_cancel(self) -> None:
        self._cancel_event.clear()

    # ── Confirmation ──────────────────────────────────────────────────────────

    def register_confirmation(self, tool_call_id: str) -> "asyncio.Future[bool]":
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        self._pending_confirmations[tool_call_id] = future
        return future

    def resolve_confirmation(self, tool_call_id: str, approved: bool) -> bool:
        future = self._pending_confirmations.pop(tool_call_id, None)
        if future is None or future.done():
            return False
        future.set_result(approved)
        log.info("session.confirmation_resolved", session_id=self.id,
                 tool_call_id=tool_call_id, approved=approved)
        return True

    # ── Summary ───────────────────────────────────────────────────────────────

    def status_summary(self) -> dict:
        return {
            "session_id": self.id,
            "user_id": self.user_id,
            "trust_level": self.trust_level.value,
            "turns": self.turn_count,
            "tool_calls": self.tool_call_count,
            "tokens_in": self.total_input_tokens,
            "tokens_out": self.total_output_tokens,
            "has_active_plan": self.active_plan is not None,
            "plan_progress": self.active_plan.progress_summary if self.active_plan else None,
            "cancelled": self.is_cancelled(),
            "uptime_seconds": round(time.time() - self.created_at, 1),
        }

    def __repr__(self) -> str:
        return (f"<Session id={self.id} user={self.user_id} "
                f"trust={self.trust_level.value} turns={self.turn_count}>")
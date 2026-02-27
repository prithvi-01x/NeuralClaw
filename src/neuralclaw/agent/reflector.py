"""
agent/reflector.py — Post-Turn Reflector

Commits a completed agent turn to episodic memory and optionally
extracts a lesson via the Reasoner's reflect() method.

Extracted from Orchestrator and ResponseSynthesizer (Phase 3 core-hardening).
Renamed from the concept of response_synthesizer's reflection logic.

Responsibilities:
  - Build a concise episode summary from step results
  - Call Reasoner.reflect() to extract a transferable lesson
  - Persist the lesson to long-term memory
  - Commit the episode record in episodic store

Usage (async, pure side-effects):
    reflector = Reflector(reasoner, memory_manager)
    await reflector.commit(session, steps_taken, goal, outcome)

Phase 3 (core-hardening): New file.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from neuralclaw.observability.logger import get_logger

log = get_logger(__name__)


class Reflector:
    """
    Post-turn reflection engine.

    Stateless between calls. Inject Reasoner and MemoryManager.
    """

    def __init__(self, reasoner, memory_manager) -> None:
        """
        Args:
            reasoner:        agent.reasoner.Reasoner instance.
            memory_manager:  memory.memory_manager.MemoryManager instance.
        """
        self._reasoner = reasoner
        self._memory = memory_manager

    async def commit(
        self,
        session,                            # agent.session.Session
        steps_taken: list[str],
        goal: str = "",
        outcome: str = "success",
    ) -> None:
        """
        Commit the completed turn to memory and extract a lesson.

        Args:
            session:      The current Session (provides session_id, plan info).
            steps_taken:  Human-readable step summaries from the turn.
            goal:         The user goal or task description.
            outcome:      "success" | "partial" | "error" — used for episode record.

        This method never raises. All errors are logged as warnings.
        """
        if not steps_taken:
            return

        # ── Extract a lesson from the Reasoner ────────────────────────────────
        lesson: Optional[str] = None
        try:
            lesson = await self._reasoner.reflect(
                goal=goal,
                steps_taken=steps_taken,
                outcome=outcome,
            )
        except asyncio.CancelledError:
            raise
        except (OSError, RuntimeError, ValueError) as e:
            log.warning("reflector.lesson_extract_failed", error=str(e), error_type=type(e).__name__)

        if lesson:
            try:
                await self._memory.add_reflection(
                    session.id,
                    lesson,
                    context=goal,
                )
            except asyncio.CancelledError:
                raise
            except (OSError, RuntimeError, ValueError) as e:
                log.warning("reflector.lesson_store_failed", error=str(e), error_type=type(e).__name__)

        # ── Commit the episode record ─────────────────────────────────────────
        episode_id = (
            session.active_plan.episode_id if session.active_plan else None
        )
        if episode_id:
            try:
                await self._memory.commit_episode(
                    episode_id=episode_id,
                    outcome=outcome,
                    summary=f"{outcome.capitalize()}: {goal}",
                    steps=steps_taken,
                    tool_count=session.tool_call_count,
                    turn_count=session.turn_count,
                )
            except asyncio.CancelledError:
                raise
            except (OSError, RuntimeError, ValueError) as e:
                log.warning("reflector.episode_commit_failed", error=str(e), error_type=type(e).__name__)

        log.info(
            "reflector.committed",
            session_id=session.id,
            steps=len(steps_taken),
            outcome=outcome,
            has_lesson=bool(lesson),
        )
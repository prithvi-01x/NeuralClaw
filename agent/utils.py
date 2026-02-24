"""
agent/utils.py — Shared Agent Utilities

Small helpers shared across the agent package.
Keeping these here avoids copy-pasting them across orchestrator.py
and executor.py (where they previously lived as separate duplicates).
"""

from __future__ import annotations

import asyncio

from observability.logger import get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Fire-and-forget helper
# ─────────────────────────────────────────────────────────────────────────────

# Module-level strong-reference set — prevents asyncio from GC-ing tasks mid-flight.
# Tasks are removed in the done-callback so the set never grows unbounded.
# Shared by orchestrator + executor so the full lifecycle of every background task
# is tracked in one place.
_BG_TASKS: set[asyncio.Task] = set()


def fire_and_forget(coro, label: str = "bg_task") -> asyncio.Task:
    """
    Schedule a coroutine as a background asyncio task.

    Unlike a bare asyncio.create_task(), this:
      1. Holds a strong reference so the GC cannot cancel the task mid-flight.
      2. Attaches a done-callback that logs any unhandled exception so failures
         are never silently swallowed.
      3. Removes the task from the reference set when complete (no unbounded growth).

    The label is included in the log entry for easy grep-ability.
    """
    task = asyncio.create_task(coro)
    _BG_TASKS.add(task)  # strong ref — prevents GC reaping

    def _on_done(t: asyncio.Task) -> None:
        _BG_TASKS.discard(t)  # release strong ref when finished
        if t.cancelled():
            return
        exc = t.exception()
        if exc is not None:
            log.warning(
                "bg_task.failed",
                label=label,
                error=str(exc),
                error_type=type(exc).__name__,
            )

    task.add_done_callback(_on_done)
    return task

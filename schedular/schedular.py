"""
scheduler/scheduler.py — Task Scheduler (stub)

Placeholder for the autonomous task scheduling subsystem.

Future responsibilities:
  - Cron-style recurring tasks (e.g. daily memory pruning)
  - Deferred goals: schedule a task to run at a future time
  - Bounded concurrency: honour max_concurrent_tasks from SchedulerConfig
  - Integration with the agent Orchestrator for background turn execution

Implementation note:
  - max_concurrent_tasks ≤ 3 to prevent SQLite locking under concurrent tasks
    (aiosqlite handles serialisation, but contention still increases latency)
  - Timezone-aware scheduling — always store scheduled times as UTC internally,
    display in settings.scheduler.timezone

This file is intentionally empty during the kernel hardening phase.
Do not add scheduling logic until Phases 1-6 are fully validated.
"""

from __future__ import annotations

from observability.logger import get_logger

log = get_logger(__name__)


class TaskScheduler:
    """
    Stub scheduler.  Not yet active.

    When implemented, accepts a KernelConfig and an Orchestrator reference
    and manages background task execution within the configured concurrency
    and timezone constraints.
    """

    def __init__(self, max_concurrent_tasks: int = 3, timezone: str = "UTC") -> None:
        self.max_concurrent_tasks = max_concurrent_tasks
        self.timezone = timezone
        log.debug(
            "scheduler.init",
            max_concurrent_tasks=max_concurrent_tasks,
            timezone=timezone,
            status="stub — not yet active",
        )

    async def start(self) -> None:
        """Start the scheduler event loop.  Stub — no-op."""
        log.info("scheduler.start.stub")

    async def stop(self) -> None:
        """Stop the scheduler and cancel pending tasks.  Stub — no-op."""
        log.info("scheduler.stop.stub")
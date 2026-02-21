"""
scheduler/scheduler.py — NeuralClaw Task Scheduler

Runs cron and interval tasks that trigger the agent autonomously.
Uses asyncio for non-blocking task execution with a configurable
max-concurrent-tasks limit.

Cron parsing is done with a lightweight internal parser (no external deps).
Supports standard 5-field cron expressions: minute hour day month weekday.

Example schedules:
    "0 8 * * *"    — every day at 08:00 UTC
    "*/30 * * * *" — every 30 minutes
    "0 9 * * 1"    — every Monday at 09:00 UTC
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Optional

from observability.logger import get_logger

log = get_logger(__name__)


class TaskStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    FAILED = "failed"
    DISABLED = "disabled"


@dataclass
class ScheduledTask:
    id: str
    name: str
    goal: str                               # The prompt/goal sent to the agent
    schedule_type: str                      # "cron" | "interval"
    cron_expression: Optional[str] = None   # e.g. "0 8 * * *"
    interval_seconds: Optional[float] = None

    enabled: bool = True
    status: TaskStatus = TaskStatus.IDLE
    last_run: Optional[float] = None        # Unix timestamp
    next_run: Optional[float] = None
    run_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None

    metadata: dict[str, Any] = field(default_factory=dict)


# Callable type for the agent run function
AgentRunFn = Callable[[str], Coroutine[Any, Any, str]]


class TaskScheduler:
    """
    Asyncio-based task scheduler for the NeuralClaw agent.

    Tasks are stored in memory. For persistence across restarts,
    tasks should be re-added at startup from config or a database.
    """

    def __init__(
        self,
        run_fn: AgentRunFn,
        timezone_name: str = "UTC",
        max_concurrent: int = 3,
        tick_interval: float = 30.0,    # How often to check for due tasks (seconds)
    ):
        """
        Args:
            run_fn:          Async function that runs an agent goal and returns a result string.
                             Signature: async (goal: str) -> str
            timezone_name:   Timezone for cron expressions (currently UTC only).
            max_concurrent:  Max tasks running simultaneously.
            tick_interval:   How often the scheduler wakes to check for due tasks.
        """
        self._run_fn = run_fn
        self._timezone = timezone_name
        self._max_concurrent = max_concurrent
        self._tick_interval = tick_interval

        self._tasks: dict[str, ScheduledTask] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._ticker: Optional[asyncio.Task] = None
        self._running_tasks: dict[str, asyncio.Task] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the scheduler tick loop."""
        log.info("scheduler.starting", tick_interval=self._tick_interval)
        self._ticker = asyncio.create_task(self._tick_loop())

    async def stop(self) -> None:
        """Stop the scheduler and cancel running tasks."""
        if self._ticker:
            self._ticker.cancel()
            try:
                await self._ticker
            except asyncio.CancelledError:
                pass

        # Cancel running task coroutines
        for task in self._running_tasks.values():
            task.cancel()

        if self._running_tasks:
            await asyncio.gather(*self._running_tasks.values(), return_exceptions=True)

        log.info("scheduler.stopped")

    # ── Task management ───────────────────────────────────────────────────────

    def add_cron(
        self,
        name: str,
        cron_expression: str,
        goal: str,
        enabled: bool = True,
        metadata: Optional[dict] = None,
    ) -> ScheduledTask:
        """
        Add a cron-scheduled task.

        Args:
            name:            Human-readable name.
            cron_expression: Standard 5-field cron: "minute hour day month weekday"
            goal:            The agent goal/prompt to execute.
            enabled:         Whether the task starts enabled.
        """
        task_id = f"cron_{uuid.uuid4().hex[:8]}"
        now = time.time()
        next_run = _cron_next(cron_expression, now)

        task = ScheduledTask(
            id=task_id,
            name=name,
            goal=goal,
            schedule_type="cron",
            cron_expression=cron_expression,
            enabled=enabled,
            next_run=next_run,
            metadata=metadata or {},
        )
        self._tasks[task_id] = task

        log.info("scheduler.task_added",
                 id=task_id, name=name, cron=cron_expression,
                 next_run=datetime.fromtimestamp(next_run, tz=timezone.utc).isoformat())
        return task

    def add_interval(
        self,
        name: str,
        interval_seconds: float,
        goal: str,
        run_immediately: bool = False,
        enabled: bool = True,
        metadata: Optional[dict] = None,
    ) -> ScheduledTask:
        """
        Add an interval-scheduled task.

        Args:
            name:             Human-readable name.
            interval_seconds: How often to run (e.g. 3600 for hourly).
            goal:             The agent goal/prompt to execute.
            run_immediately:  If True, schedule first run in 1 second.
        """
        task_id = f"interval_{uuid.uuid4().hex[:8]}"
        now = time.time()
        next_run = now + 1 if run_immediately else now + interval_seconds

        task = ScheduledTask(
            id=task_id,
            name=name,
            goal=goal,
            schedule_type="interval",
            interval_seconds=interval_seconds,
            enabled=enabled,
            next_run=next_run,
            metadata=metadata or {},
        )
        self._tasks[task_id] = task

        log.info("scheduler.task_added",
                 id=task_id, name=name, interval_seconds=interval_seconds)
        return task

    def remove(self, task_id: str) -> bool:
        """Remove a task. Returns True if it existed."""
        if task_id in self._tasks:
            del self._tasks[task_id]
            log.info("scheduler.task_removed", id=task_id)
            return True
        return False

    def enable(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if task:
            task.enabled = True
            log.info("scheduler.task_enabled", id=task_id)
            return True
        return False

    def disable(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if task:
            task.enabled = False
            task.status = TaskStatus.DISABLED
            log.info("scheduler.task_disabled", id=task_id)
            return True
        return False

    def list_tasks(self) -> list[ScheduledTask]:
        return list(self._tasks.values())

    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        return self._tasks.get(task_id)

    # ── Tick loop ─────────────────────────────────────────────────────────────

    async def _tick_loop(self) -> None:
        """Main loop: wake every tick_interval and fire due tasks."""
        log.info("scheduler.tick_loop.started")
        while True:
            try:
                await asyncio.sleep(self._tick_interval)
                await self._check_and_fire()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("scheduler.tick_loop.error", error=str(e))

    async def _check_and_fire(self) -> None:
        """Check all tasks and fire any that are due."""
        now = time.time()
        due = [
            t for t in self._tasks.values()
            if t.enabled
            and t.status == TaskStatus.IDLE
            and t.next_run is not None
            and t.next_run <= now
        ]

        for task in due:
            if task.id not in self._running_tasks:
                log.info("scheduler.task_firing", id=task.id, name=task.name)
                asyncio_task = asyncio.create_task(self._run_task(task))
                self._running_tasks[task.id] = asyncio_task
                asyncio_task.add_done_callback(
                    lambda fut, tid=task.id: self._running_tasks.pop(tid, None)
                )

    async def _run_task(self, task: ScheduledTask) -> None:
        """Execute a single scheduled task under the semaphore."""
        async with self._semaphore:
            task.status = TaskStatus.RUNNING
            task.last_run = time.time()
            task.run_count += 1

            log.info("scheduler.task.start", id=task.id, name=task.name, run=task.run_count)

            try:
                result = await self._run_fn(task.goal)
                log.info("scheduler.task.success",
                          id=task.id, name=task.name,
                          result_preview=(result or "")[:200])
                task.status = TaskStatus.IDLE
            except asyncio.CancelledError:
                task.status = TaskStatus.IDLE
                log.info("scheduler.task.cancelled", id=task.id)
                raise
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error_count += 1
                task.last_error = str(e)
                log.error("scheduler.task.error",
                           id=task.id, name=task.name, error=str(e))
                # After failure, still schedule next run
            finally:
                self._advance_schedule(task)

    def _advance_schedule(self, task: ScheduledTask) -> None:
        """Update next_run for the task after a run."""
        now = time.time()
        if task.schedule_type == "cron" and task.cron_expression:
            task.next_run = _cron_next(task.cron_expression, now)
        elif task.schedule_type == "interval" and task.interval_seconds:
            task.next_run = now + task.interval_seconds
        else:
            task.enabled = False  # One-shot: disable after running

        if task.next_run:
            log.debug("scheduler.task.next_run",
                      id=task.id,
                      next=datetime.fromtimestamp(task.next_run, tz=timezone.utc).isoformat())

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls, settings, run_fn: AgentRunFn) -> "TaskScheduler":
        """Create a TaskScheduler from NeuralClaw Settings."""
        sched_cfg = settings.scheduler or {}
        return cls(
            run_fn=run_fn,
            timezone_name=sched_cfg.get("timezone", "UTC"),
            max_concurrent=sched_cfg.get("max_concurrent_tasks", 3),
        )


# ── Lightweight cron parser ───────────────────────────────────────────────────


def _cron_next(expression: str, after: float) -> float:
    """
    Compute the next Unix timestamp after `after` that satisfies the cron expression.

    Supports: * , - / for each of the 5 fields: minute hour day month weekday.
    Weekday: 0=Sunday, 6=Saturday (standard cron).

    Raises ValueError for invalid expressions.
    """
    fields = expression.strip().split()
    if len(fields) != 5:
        raise ValueError(f"Invalid cron expression (need 5 fields): '{expression}'")

    minute_f, hour_f, day_f, month_f, weekday_f = fields

    # Start 1 minute after 'after', rounded to minute
    dt = datetime.fromtimestamp(after + 60, tz=timezone.utc).replace(second=0, microsecond=0)

    # Search up to 1 year to avoid infinite loops on bad expressions
    max_minutes = 525_600
    for _ in range(max_minutes):
        if (
            _matches(dt.month, month_f, 1, 12)
            and _matches(dt.day, day_f, 1, 31)
            and _matches(dt.isoweekday() % 7, weekday_f, 0, 6)  # isoweekday: Mon=1..Sun=7 → 0-6
            and _matches(dt.hour, hour_f, 0, 23)
            and _matches(dt.minute, minute_f, 0, 59)
        ):
            return dt.timestamp()
        dt = dt.replace(second=0, microsecond=0)
        dt = datetime.fromtimestamp(dt.timestamp() + 60, tz=timezone.utc)

    raise ValueError(f"Could not find next run time for cron '{expression}'")


def _matches(value: int, field: str, lo: int, hi: int) -> bool:
    """Check if `value` satisfies a cron field string."""
    if field == "*":
        return True

    for part in field.split(","):
        part = part.strip()

        # Step: */n or start/n or start-end/n
        if "/" in part:
            range_part, step_str = part.split("/", 1)
            step = int(step_str)
            if range_part == "*":
                start, end = lo, hi
            elif "-" in range_part:
                s, e = range_part.split("-", 1)
                start, end = int(s), int(e)
            else:
                start = int(range_part)
                end = hi
            if start <= value <= end and (value - start) % step == 0:
                return True

        # Range: a-b
        elif "-" in part:
            s, e = part.split("-", 1)
            if int(s) <= value <= int(e):
                return True

        # Exact value
        else:
            if int(part) == value:
                return True

    return False
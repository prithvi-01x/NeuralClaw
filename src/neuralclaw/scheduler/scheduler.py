"""
scheduler/scheduler.py — TaskScheduler (Phase D)

Autonomous task scheduler for NeuralClaw. Runs cron-style recurring tasks
as asyncio coroutines inside the main event loop. Calls
orchestrator.run_turn() for each task on its schedule.

Design
------
* Pure asyncio — no threads, no external cron daemon, no APScheduler.
* Each ScheduledTask has a cron expression (parsed by croniter) + a meta-skill
  name to invoke.
* Bounded concurrency: max_concurrent_tasks semaphore prevents pile-ups.
* Timezone-aware: all internal state in UTC.
* Fail-safe: task errors are caught, logged, and written to episodic memory.
  A failed task NEVER brings down the scheduler loop.
* Resource-lock: a per-task asyncio.Lock prevents overlapping runs of the same
  task if a previous run is still executing when the next trigger fires.
* Graceful shutdown: stop() cancels all watcher tasks and waits for them.

Three pre-configured tasks (Phase D roadmap, Table 20):
    daily_briefing        — meta_daily_assistant    — 07:00 UTC daily
    nightly_maintenance   — meta_system_maintenance — 02:00 UTC daily
    weekly_repo_audit     — meta_repo_audit         — Sunday 18:00 UTC

Usage::

    scheduler = TaskScheduler.from_settings(settings, orchestrator, memory_manager)
    await scheduler.start()   # non-blocking — runs as background asyncio Tasks
    ...
    await scheduler.stop()    # graceful shutdown
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from neuralclaw.observability.logger import get_logger
from neuralclaw.agent.session import Session
from neuralclaw.skills.types import TrustLevel

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Cron helper
# ─────────────────────────────────────────────────────────────────────────────

def _next_run_utc(cron_expr: str, after: datetime) -> datetime:
    """Return the next UTC datetime matching cron_expr after `after`."""
    try:
        from croniter import croniter  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "TaskScheduler requires the 'croniter' package. "
            "Install: pip install croniter"
        ) from exc
    it = croniter(cron_expr, after)
    return it.get_next(datetime).replace(tzinfo=timezone.utc)


# ─────────────────────────────────────────────────────────────────────────────
# ScheduledTask
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScheduledTask:
    """
    Configuration for one recurring scheduled task.

    name          Unique identifier — used in logs and memory keys.
    cron          5-field cron expression in UTC (e.g. '0 7 * * *').
    skill_name    Meta-skill (or any skill) to invoke.
    skill_args    Extra arguments forwarded in the invocation message.
    trust_level   Trust level for the synthetic session.
    enabled       False = skip without removing.
    timeout_s     Max seconds the task may run (default 600).
    """
    name: str
    cron: str
    skill_name: str
    skill_args: dict = field(default_factory=dict)
    trust_level: TrustLevel = TrustLevel.MEDIUM
    enabled: bool = True
    timeout_s: int = 600

    def next_run_after(self, dt: datetime) -> datetime:
        return _next_run_utc(self.cron, dt)

    def to_invocation_message(self) -> str:
        """Build the message sent to orchestrator.run_turn()."""
        if self.skill_args:
            import json
            return f"Run {self.skill_name} with args {json.dumps(self.skill_args)}"
        return f"Run {self.skill_name}"


# ─────────────────────────────────────────────────────────────────────────────
# TaskRun — runtime record for one execution
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TaskRun:
    run_id: str
    task_name: str
    started_at: float = field(default_factory=time.monotonic)
    finished_at: Optional[float] = None
    succeeded: Optional[bool] = None
    error: Optional[str] = None

    @property
    def duration_s(self) -> float:
        end = self.finished_at or time.monotonic()
        return end - self.started_at

    def finish(self, *, succeeded: bool, error: Optional[str] = None) -> None:
        self.finished_at = time.monotonic()
        self.succeeded = succeeded
        self.error = error


# ─────────────────────────────────────────────────────────────────────────────
# SchedulerStats
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SchedulerStats:
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    skipped_runs: int = 0
    last_run_at: Optional[str] = None
    last_run_task: Optional[str] = None
    last_error: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Phase D canonical tasks
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_TASKS: list[ScheduledTask] = [
    ScheduledTask(
        name="daily_briefing",
        cron="0 7 * * *",
        skill_name="meta_daily_assistant",
        trust_level=TrustLevel.MEDIUM,
        timeout_s=120,
    ),
    ScheduledTask(
        name="nightly_maintenance",
        cron="0 2 * * *",
        skill_name="meta_system_maintenance",
        trust_level=TrustLevel.MEDIUM,
        timeout_s=300,
    ),
    ScheduledTask(
        name="weekly_repo_audit",
        cron="0 18 * * 0",
        skill_name="meta_repo_audit",
        trust_level=TrustLevel.MEDIUM,
        timeout_s=180,
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# TaskScheduler
# ─────────────────────────────────────────────────────────────────────────────

class TaskScheduler:
    """
    Autonomous task scheduler for NeuralClaw.

    Manages ScheduledTask objects and fires each one at the right UTC time
    by calling orchestrator.run_turn() with a synthetic session.

    Lifecycle::

        scheduler = TaskScheduler(orchestrator, memory_manager)
        await scheduler.start()   # starts background watcher loops
        await scheduler.stop()    # cancels watchers, waits for in-flight runs

    Introspection::

        scheduler.stats           # SchedulerStats counters
        scheduler.list_tasks()    # List[dict] for /status display
        scheduler.task_history    # List[TaskRun] (last 100)
    """

    def __init__(
        self,
        orchestrator,
        memory_manager,
        tasks: Optional[list[ScheduledTask]] = None,
        max_concurrent_tasks: int = 3,
        timezone: str = "UTC",
    ) -> None:
        self._orchestrator = orchestrator
        self._memory = memory_manager
        self._tasks: list[ScheduledTask] = tasks if tasks is not None else list(_DEFAULT_TASKS)
        self._timezone = timezone

        self._semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self._task_locks: dict[str, asyncio.Lock] = {t.name: asyncio.Lock() for t in self._tasks}
        self._watcher_tasks: list[asyncio.Task] = []

        self.stats = SchedulerStats()
        self.task_history: list[TaskRun] = []
        self._history_limit = 100
        self._running = False

        log.info(
            "scheduler.init",
            tasks=[t.name for t in self._tasks],
            max_concurrent=max_concurrent_tasks,
            timezone=timezone,
        )

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls, settings, orchestrator, memory_manager) -> "TaskScheduler":
        tasks = list(_DEFAULT_TASKS)

        # Add heartbeat task — reads interval from scheduler config (default 30 min)
        heartbeat_interval = getattr(settings.scheduler, "heartbeat_interval_minutes", 30)
        heartbeat_enabled  = getattr(settings.scheduler, "heartbeat_enabled", True)
        if heartbeat_enabled:
            from neuralclaw.agent.heartbeat import build_heartbeat_task
            tasks.append(build_heartbeat_task(interval_minutes=heartbeat_interval))

        return cls(
            orchestrator=orchestrator,
            memory_manager=memory_manager,
            tasks=tasks,
            max_concurrent_tasks=settings.scheduler.max_concurrent_tasks,
            timezone=settings.scheduler.timezone,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start background watcher loops for all enabled tasks. Non-blocking."""
        if self._running:
            log.warning("scheduler.already_running")
            return
        self._running = True
        for task in self._tasks:
            if not task.enabled:
                log.info("scheduler.task_disabled", task=task.name)
                continue
            watcher = asyncio.create_task(
                self._watcher_loop(task),
                name=f"scheduler:watch:{task.name}",
            )
            self._watcher_tasks.append(watcher)
            log.info("scheduler.task_watching", task=task.name, cron=task.cron)
        log.info("scheduler.started", active_watchers=len(self._watcher_tasks))

    async def stop(self) -> None:
        """Cancel all watcher loops and wait for them to finish."""
        if not self._running:
            return
        self._running = False
        log.info("scheduler.stopping", watchers=len(self._watcher_tasks))
        for w in self._watcher_tasks:
            w.cancel()
        if self._watcher_tasks:
            await asyncio.gather(*self._watcher_tasks, return_exceptions=True)
        self._watcher_tasks.clear()
        log.info("scheduler.stopped")

    def add_task(self, task: ScheduledTask) -> None:
        """Register a new task. Starts its watcher immediately if already running."""
        if any(t.name == task.name for t in self._tasks):
            raise ValueError(f"Task '{task.name}' already registered.")
        self._tasks.append(task)
        self._task_locks[task.name] = asyncio.Lock()
        if self._running and task.enabled:
            watcher = asyncio.create_task(
                self._watcher_loop(task),
                name=f"scheduler:watch:{task.name}",
            )
            self._watcher_tasks.append(watcher)
            log.info("scheduler.task_added_live", task=task.name)

    def remove_task(self, name: str) -> bool:
        """Disable a task by name. Its watcher exits on the next sleep cycle."""
        for task in self._tasks:
            if task.name == name:
                task.enabled = False
                log.info("scheduler.task_disabled", task=name)
                return True
        return False

    def list_tasks(self) -> list[dict]:
        """Return a status summary of all tasks for /status display."""
        now_utc = datetime.now(timezone.utc)
        result = []
        for task in self._tasks:
            try:
                next_run = task.next_run_after(now_utc).isoformat()
            except Exception:
                next_run = "unknown (croniter not installed)"
            last = next((r for r in reversed(self.task_history) if r.task_name == task.name), None)
            result.append({
                "name": task.name,
                "cron": task.cron,
                "skill": task.skill_name,
                "enabled": task.enabled,
                "next_run_utc": next_run,
                "last_run_succeeded": last.succeeded if last else None,
                "last_run_duration_s": round(last.duration_s, 1) if last else None,
                "last_error": last.error if last else None,
            })
        return result

    # ── Watcher loop ──────────────────────────────────────────────────────────

    async def _watcher_loop(self, task: ScheduledTask) -> None:
        """Sleep until next cron trigger then fire _run_task(). Loops indefinitely."""
        log.info("scheduler.watcher.start", task=task.name, cron=task.cron)
        try:
            while self._running and task.enabled:
                now_utc = datetime.now(timezone.utc)
                try:
                    next_run = task.next_run_after(now_utc)
                except Exception as e:
                    log.error("scheduler.cron_error", task=task.name, error=str(e))
                    await asyncio.sleep(60)
                    continue

                sleep_s = max((next_run - now_utc).total_seconds(), 1)
                log.info(
                    "scheduler.watcher.sleeping",
                    task=task.name,
                    next_run=next_run.isoformat(),
                    sleep_s=round(sleep_s, 1),
                )

                try:
                    await asyncio.sleep(sleep_s)
                except asyncio.CancelledError:
                    log.info("scheduler.watcher.cancelled", task=task.name)
                    return

                if not (self._running and task.enabled):
                    break

                # Fire as a separate task so the watcher loop continues immediately
                asyncio.create_task(
                    self._run_task(task),
                    name=f"scheduler:run:{task.name}:{uuid.uuid4().hex[:6]}",
                )

        except asyncio.CancelledError:
            log.info("scheduler.watcher.cancelled", task=task.name)
        except Exception as e:
            log.error(
                "scheduler.watcher.crashed",
                task=task.name,
                error=str(e),
                error_type=type(e).__name__,
            )

    # ── Task execution ────────────────────────────────────────────────────────

    async def _run_task(self, task: ScheduledTask) -> None:
        """Execute one task run. Never raises — all errors are caught."""
        lock = self._task_locks.get(task.name)
        if lock is None:
            log.error("scheduler.no_lock", task=task.name)
            return

        if lock.locked():
            log.warning("scheduler.task_skipped.still_running", task=task.name)
            self.stats.skipped_runs += 1
            return

        run = TaskRun(run_id=uuid.uuid4().hex[:12], task_name=task.name)
        self.stats.total_runs += 1
        self.stats.last_run_task = task.name
        self.stats.last_run_at = datetime.now(timezone.utc).isoformat()

        async with lock:
            async with self._semaphore:
                log.info(
                    "scheduler.task_start",
                    task=task.name,
                    run_id=run.run_id,
                    skill=task.skill_name,
                )
                session = self._make_session(task)
                message = task.to_invocation_message()

                # ── Heartbeat: delegate to HeartbeatRunner instead of SkillBus ──
                if task.skill_name == "heartbeat":
                    try:
                        from neuralclaw.agent.heartbeat import HeartbeatRunner
                        runner = HeartbeatRunner(self._orchestrator, self._memory)
                        await asyncio.wait_for(runner.run(session), timeout=task.timeout_s)
                        run.finish(succeeded=True)
                        self.stats.successful_runs += 1
                        log.info("scheduler.heartbeat_complete", run_id=run.run_id,
                                 duration_s=round(run.duration_s, 1))
                    except asyncio.TimeoutError:
                        run.finish(succeeded=False, error="Heartbeat timed out")
                        self.stats.failed_runs += 1
                        self.stats.last_error = "Heartbeat timed out"
                        log.warning("scheduler.heartbeat_timeout", run_id=run.run_id)
                    except Exception as hb_err:
                        run.finish(succeeded=False, error=str(hb_err))
                        self.stats.failed_runs += 1
                        self.stats.last_error = str(hb_err)
                        log.error("scheduler.heartbeat_error", error=str(hb_err), exc_info=True)
                    self.task_history.append(run)
                    return
                # ── Normal skill dispatch ─────────────────────────────────────

                try:
                    turn_result = await asyncio.wait_for(
                        self._orchestrator.run_turn(session, message),
                        timeout=task.timeout_s,
                    )
                    succeeded = turn_result.succeeded
                    error_text = None if succeeded else (
                        turn_result.response.text[:200] if turn_result.response else "unknown error"
                    )
                    run.finish(succeeded=succeeded, error=error_text)

                    if succeeded:
                        self.stats.successful_runs += 1
                        log.info(
                            "scheduler.task_complete",
                            task=task.name,
                            run_id=run.run_id,
                            duration_s=round(run.duration_s, 1),
                            steps=turn_result.steps_taken,
                        )
                    else:
                        self.stats.failed_runs += 1
                        self.stats.last_error = error_text
                        log.warning(
                            "scheduler.task_failed",
                            task=task.name,
                            run_id=run.run_id,
                            status=turn_result.status,
                            error=error_text,
                        )

                except asyncio.TimeoutError:
                    err = f"Exceeded timeout ({task.timeout_s}s)"
                    run.finish(succeeded=False, error=err)
                    self.stats.failed_runs += 1
                    self.stats.last_error = err
                    log.error("scheduler.task_timeout", task=task.name, run_id=run.run_id)

                except asyncio.CancelledError:
                    run.finish(succeeded=False, error="Cancelled during shutdown")
                    log.info("scheduler.task_cancelled", task=task.name, run_id=run.run_id)
                    raise

                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
                    run.finish(succeeded=False, error=err)
                    self.stats.failed_runs += 1
                    self.stats.last_error = err
                    log.error(
                        "scheduler.task_error",
                        task=task.name,
                        run_id=run.run_id,
                        error=err,
                        exc_info=True,
                    )

                finally:
                    asyncio.create_task(
                        self._persist_run(task, run),
                        name=f"scheduler:persist:{run.run_id}",
                    )

                self.task_history.append(run)
                if len(self.task_history) > self._history_limit:
                    self.task_history = self.task_history[-self._history_limit:]

    # ── Session factory ───────────────────────────────────────────────────────

    def _make_session(self, task: ScheduledTask) -> Session:
        """Build a synthetic Session for a scheduled task run."""
        session = Session.create(
            user_id=f"scheduler:{task.name}",
            trust_level=task.trust_level,
        )
        for cap in ("net:scan", "net:fetch", "fs:read", "fs:write", "shell:run", "data:read", "data:write"):
            session.grant_capability(cap)
        return session

    # ── Persist run outcome ───────────────────────────────────────────────────

    async def _persist_run(self, task: ScheduledTask, run: TaskRun) -> None:
        """Write the completed run outcome to episodic memory."""
        if not self._memory:
            return
        try:
            content = (
                f"Scheduled task '{task.name}' completed.\n"
                f"Skill: {task.skill_name}\n"
                f"Run ID: {run.run_id}\n"
                f"Status: {'SUCCESS' if run.succeeded else 'FAILED'}\n"
                f"Duration: {run.duration_s:.1f}s\n"
                + (f"Error: {run.error}\n" if run.error else "")
            )
            await self._memory.store(content, collection="scheduler")
        except Exception as e:
            log.warning("scheduler.persist_failed", task=task.name, error=str(e))
"""
tests/unit/test_phase_d_scheduler.py — Phase D: TaskScheduler Unit Tests

Covers:
  - ScheduledTask: cron parsing, invocation message generation
  - TaskRun: lifecycle (start → finish), duration
  - TaskScheduler: start/stop, watcher lifecycle, concurrency lock (skip on overlap),
    semaphore limiting, task execution success/failure/timeout paths,
    add_task / remove_task, list_tasks, stats counters, history capping,
    _persist_run fire-and-forget, _make_session capability grants
  - Integration: three Phase D canonical tasks are registered by default
  - Safety: scheduler never propagates exceptions to caller
"""

from __future__ import annotations

import asyncio
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from schedular.schedular import (
    ScheduledTask,
    SchedulerStats,
    TaskRun,
    TaskScheduler,
    _DEFAULT_TASKS,
    _next_run_utc,
)
from skills.types import TrustLevel


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_turn_result(succeeded: bool = True, steps: int = 2) -> MagicMock:
    from agent.orchestrator import TurnResult, TurnStatus, AgentResponse
    tr = MagicMock()
    tr.succeeded = succeeded
    tr.steps_taken = steps
    tr.status = TurnStatus.SUCCESS if succeeded else TurnStatus.ERROR
    tr.response = MagicMock()
    tr.response.text = "Task completed." if succeeded else "Error occurred."
    return tr


def _make_orchestrator(succeeded: bool = True, delay: float = 0.0) -> MagicMock:
    orc = MagicMock()
    turn_result = _make_turn_result(succeeded)

    async def fake_run_turn(session, message):
        if delay:
            await asyncio.sleep(delay)
        return turn_result

    orc.run_turn = fake_run_turn
    return orc


def _make_memory() -> MagicMock:
    mem = MagicMock()
    mem.store = AsyncMock(return_value=None)
    return mem


def _make_settings(max_concurrent: int = 3, tz: str = "UTC") -> MagicMock:
    s = MagicMock()
    s.scheduler.max_concurrent_tasks = max_concurrent
    s.scheduler.timezone = tz
    return s


def _make_task(
    name: str = "test_task",
    cron: str = "0 7 * * *",
    skill_name: str = "meta_daily_assistant",
    enabled: bool = True,
    timeout_s: int = 10,
) -> ScheduledTask:
    return ScheduledTask(
        name=name,
        cron=cron,
        skill_name=skill_name,
        enabled=enabled,
        timeout_s=timeout_s,
    )


def _make_scheduler(
    orchestrator=None,
    memory=None,
    tasks: list | None = None,
    max_concurrent: int = 3,
) -> TaskScheduler:
    return TaskScheduler(
        orchestrator=orchestrator or _make_orchestrator(),
        memory_manager=memory or _make_memory(),
        tasks=tasks if tasks is not None else [],
        max_concurrent_tasks=max_concurrent,
        timezone="UTC",
    )


# ─────────────────────────────────────────────────────────────────────────────
# ScheduledTask tests
# ─────────────────────────────────────────────────────────────────────────────

class TestScheduledTask:

    def test_to_invocation_message_no_args(self):
        t = _make_task(skill_name="meta_daily_assistant")
        assert t.to_invocation_message() == "Run meta_daily_assistant"

    def test_to_invocation_message_with_args(self):
        t = ScheduledTask(
            name="t", cron="0 7 * * *", skill_name="meta_repo_audit",
            skill_args={"repo_path": "/home/user/project"},
        )
        msg = t.to_invocation_message()
        assert "meta_repo_audit" in msg
        assert "/home/user/project" in msg

    def test_next_run_after_returns_future_datetime(self):
        pytest.importorskip("croniter")
        t = _make_task(cron="0 7 * * *")
        now = datetime.now(timezone.utc)
        nxt = t.next_run_after(now)
        assert nxt > now
        assert nxt.tzinfo is not None

    def test_enabled_default_true(self):
        t = _make_task()
        assert t.enabled is True

    def test_trust_level_default_medium(self):
        t = _make_task()
        assert t.trust_level == TrustLevel.MEDIUM


# ─────────────────────────────────────────────────────────────────────────────
# TaskRun tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTaskRun:

    def test_initial_state(self):
        run = TaskRun(run_id="abc", task_name="daily_briefing")
        assert run.succeeded is None
        assert run.finished_at is None
        assert run.error is None

    def test_finish_success(self):
        run = TaskRun(run_id="abc", task_name="daily_briefing")
        run.finish(succeeded=True)
        assert run.succeeded is True
        assert run.finished_at is not None
        assert run.error is None

    def test_finish_failure(self):
        run = TaskRun(run_id="abc", task_name="daily_briefing")
        run.finish(succeeded=False, error="LLM timeout")
        assert run.succeeded is False
        assert run.error == "LLM timeout"

    def test_duration_before_finish(self):
        run = TaskRun(run_id="abc", task_name="t")
        time.sleep(0.01)
        assert run.duration_s >= 0.01

    def test_duration_after_finish(self):
        run = TaskRun(run_id="abc", task_name="t")
        run.finish(succeeded=True)
        d = run.duration_s
        time.sleep(0.05)
        assert abs(run.duration_s - d) < 0.01  # frozen after finish


# ─────────────────────────────────────────────────────────────────────────────
# SchedulerStats
# ─────────────────────────────────────────────────────────────────────────────

class TestSchedulerStats:

    def test_defaults_are_zero(self):
        s = SchedulerStats()
        assert s.total_runs == 0
        assert s.successful_runs == 0
        assert s.failed_runs == 0
        assert s.skipped_runs == 0
        assert s.last_run_at is None


# ─────────────────────────────────────────────────────────────────────────────
# TaskScheduler — construction and defaults
# ─────────────────────────────────────────────────────────────────────────────

class TestTaskSchedulerDefaults:

    def test_default_tasks_registered(self):
        """The three Phase D tasks must be present in _DEFAULT_TASKS."""
        names = {t.name for t in _DEFAULT_TASKS}
        assert "daily_briefing" in names
        assert "nightly_maintenance" in names
        assert "weekly_repo_audit" in names

    def test_default_tasks_have_correct_skills(self):
        skill_map = {t.name: t.skill_name for t in _DEFAULT_TASKS}
        assert skill_map["daily_briefing"] == "meta_daily_assistant"
        assert skill_map["nightly_maintenance"] == "meta_system_maintenance"
        assert skill_map["weekly_repo_audit"] == "meta_repo_audit"

    def test_default_tasks_cron_expressions(self):
        cron_map = {t.name: t.cron for t in _DEFAULT_TASKS}
        assert cron_map["daily_briefing"] == "0 7 * * *"
        assert cron_map["nightly_maintenance"] == "0 2 * * *"
        assert cron_map["weekly_repo_audit"] == "0 18 * * 0"

    def test_default_tasks_trust_medium(self):
        for t in _DEFAULT_TASKS:
            assert t.trust_level == TrustLevel.MEDIUM

    def test_scheduler_not_running_initially(self):
        s = _make_scheduler()
        assert not s._running

    def test_stats_zero_initially(self):
        s = _make_scheduler()
        assert s.stats.total_runs == 0

    def test_from_settings_factory(self):
        orc = _make_orchestrator()
        mem = _make_memory()
        settings = _make_settings(max_concurrent=2, tz="Europe/London")
        scheduler = TaskScheduler.from_settings(settings, orc, mem)
        assert scheduler._timezone == "Europe/London"
        assert scheduler._semaphore._value == 2


# ─────────────────────────────────────────────────────────────────────────────
# TaskScheduler — start / stop lifecycle
# ─────────────────────────────────────────────────────────────────────────────

class TestSchedulerLifecycle:

    @pytest.mark.asyncio
    async def test_start_sets_running_flag(self):
        s = _make_scheduler(tasks=[_make_task()])
        async def _noop(task):
            await asyncio.sleep(3600)
        with patch.object(s, "_watcher_loop", side_effect=_noop):
            await s.start()
        assert s._running
        await s.stop()

    @pytest.mark.asyncio
    async def test_start_creates_one_watcher_per_enabled_task(self):
        tasks = [_make_task("t1"), _make_task("t2"), _make_task("t3", enabled=False)]
        s = _make_scheduler(tasks=tasks)
        await s.start()
        await asyncio.sleep(0.01)  # let event loop tick
        assert len(s._watcher_tasks) == 2  # t3 is disabled
        await s.stop()

    @pytest.mark.asyncio
    async def test_stop_clears_watcher_tasks(self):
        s = _make_scheduler(tasks=[_make_task()])
        await s.start()
        await s.stop()
        assert len(s._watcher_tasks) == 0
        assert not s._running

    @pytest.mark.asyncio
    async def test_start_twice_is_safe(self):
        s = _make_scheduler(tasks=[_make_task()])
        await s.start()
        await s.start()  # second call should be a no-op
        await s.stop()

    @pytest.mark.asyncio
    async def test_stop_before_start_is_safe(self):
        s = _make_scheduler()
        await s.stop()  # should not raise


# ─────────────────────────────────────────────────────────────────────────────
# TaskScheduler — add_task / remove_task
# ─────────────────────────────────────────────────────────────────────────────

class TestTaskManagement:

    def test_add_task_before_start(self):
        s = _make_scheduler()
        s.add_task(_make_task("new_task"))
        assert any(t.name == "new_task" for t in s._tasks)

    def test_add_duplicate_raises(self):
        s = _make_scheduler(tasks=[_make_task("dupe")])
        with pytest.raises(ValueError, match="already registered"):
            s.add_task(_make_task("dupe"))

    def test_remove_task_disables_it(self):
        s = _make_scheduler(tasks=[_make_task("to_remove")])
        result = s.remove_task("to_remove")
        assert result is True
        assert not any(t.enabled for t in s._tasks if t.name == "to_remove")

    def test_remove_nonexistent_returns_false(self):
        s = _make_scheduler()
        assert s.remove_task("ghost") is False

    def test_list_tasks_returns_all(self):
        tasks = [_make_task("t1"), _make_task("t2")]
        s = _make_scheduler(tasks=tasks)
        listed = s.list_tasks()
        assert len(listed) == 2
        names = {t["name"] for t in listed}
        assert "t1" in names and "t2" in names

    def test_list_tasks_includes_required_keys(self):
        s = _make_scheduler(tasks=[_make_task()])
        item = s.list_tasks()[0]
        for key in ("name", "cron", "skill", "enabled", "next_run_utc", "last_run_succeeded", "last_error"):
            assert key in item


# ─────────────────────────────────────────────────────────────────────────────
# TaskScheduler — _run_task execution paths
# ─────────────────────────────────────────────────────────────────────────────

class TestTaskExecution:

    @pytest.mark.asyncio
    async def test_successful_run_increments_stats(self):
        orc = _make_orchestrator(succeeded=True)
        s = _make_scheduler(orchestrator=orc, tasks=[_make_task()])
        task = s._tasks[0]
        await s._run_task(task)
        assert s.stats.total_runs == 1
        assert s.stats.successful_runs == 1
        assert s.stats.failed_runs == 0

    @pytest.mark.asyncio
    async def test_failed_run_increments_failed_stats(self):
        orc = _make_orchestrator(succeeded=False)
        s = _make_scheduler(orchestrator=orc, tasks=[_make_task()])
        task = s._tasks[0]
        await s._run_task(task)
        assert s.stats.total_runs == 1
        assert s.stats.failed_runs == 1
        assert s.stats.successful_runs == 0

    @pytest.mark.asyncio
    async def test_run_appended_to_history(self):
        s = _make_scheduler(tasks=[_make_task()])
        await s._run_task(s._tasks[0])
        assert len(s.task_history) == 1
        assert s.task_history[0].task_name == "test_task"

    @pytest.mark.asyncio
    async def test_history_capped_at_limit(self):
        orc = _make_orchestrator()
        s = _make_scheduler(orchestrator=orc, tasks=[_make_task()])
        s._history_limit = 5
        task = s._tasks[0]
        for _ in range(8):
            s._task_locks[task.name] = asyncio.Lock()  # reset lock each time
            await s._run_task(task)
        assert len(s.task_history) == 5

    @pytest.mark.asyncio
    async def test_timeout_recorded_as_failure(self):
        # Orchestrator hangs for longer than task timeout
        orc = _make_orchestrator(delay=10.0)
        task = _make_task(timeout_s=0.05)
        s = _make_scheduler(orchestrator=orc, tasks=[task])
        await s._run_task(s._tasks[0])
        assert s.stats.failed_runs == 1
        run = s.task_history[0]
        assert run.succeeded is False
        assert "timeout" in run.error.lower() or "Exceeded" in run.error

    @pytest.mark.asyncio
    async def test_exception_in_orchestrator_recorded_as_failure(self):
        orc = MagicMock()
        async def crash(session, msg):
            raise RuntimeError("LLM exploded")
        orc.run_turn = crash
        task = _make_task()
        s = _make_scheduler(orchestrator=orc, tasks=[task])
        await s._run_task(s._tasks[0])
        assert s.stats.failed_runs == 1
        run = s.task_history[0]
        assert "RuntimeError" in run.error

    @pytest.mark.asyncio
    async def test_overlapping_run_skipped(self):
        """If lock is held (previous run still going), the second trigger is skipped."""
        orc = _make_orchestrator(delay=0.5)
        task = _make_task(timeout_s=5)
        s = _make_scheduler(orchestrator=orc, tasks=[task])
        lock = s._task_locks[task.name]

        # Acquire the lock to simulate a running task
        async with lock:
            await s._run_task(s._tasks[0])  # should be skipped immediately

        assert s.stats.skipped_runs == 1
        assert s.stats.total_runs == 0

    @pytest.mark.asyncio
    async def test_stats_last_run_fields_updated(self):
        s = _make_scheduler(tasks=[_make_task("alpha")])
        await s._run_task(s._tasks[0])
        assert s.stats.last_run_task == "alpha"
        assert s.stats.last_run_at is not None

    @pytest.mark.asyncio
    async def test_persist_run_called_after_execution(self):
        mem = _make_memory()
        s = _make_scheduler(memory=mem, tasks=[_make_task()])
        await s._run_task(s._tasks[0])
        await asyncio.sleep(0.05)  # let persist fire-and-forget settle
        mem.store.assert_called_once()
        call_args = mem.store.call_args
        assert "test_task" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_persist_failure_does_not_crash_scheduler(self):
        mem = MagicMock()
        mem.store = AsyncMock(side_effect=RuntimeError("DB down"))
        s = _make_scheduler(memory=mem, tasks=[_make_task()])
        # Should not raise
        await s._run_task(s._tasks[0])
        await asyncio.sleep(0.05)


# ─────────────────────────────────────────────────────────────────────────────
# TaskScheduler — _make_session
# ─────────────────────────────────────────────────────────────────────────────

class TestSessionFactory:

    def test_make_session_trust_level(self):
        task = _make_task()
        task.trust_level = TrustLevel.HIGH
        s = _make_scheduler(tasks=[task])
        session = s._make_session(task)
        assert session.trust_level == TrustLevel.HIGH

    def test_make_session_grants_required_capabilities(self):
        task = _make_task()
        s = _make_scheduler(tasks=[task])
        session = s._make_session(task)
        expected = {"net:scan", "net:fetch", "fs:read", "fs:write", "shell:run", "data:read", "data:write"}
        assert expected.issubset(session.granted_capabilities)

    def test_make_session_user_id_includes_task_name(self):
        task = _make_task(name="weekly_repo_audit")
        s = _make_scheduler(tasks=[task])
        session = s._make_session(task)
        assert "weekly_repo_audit" in session.id or "weekly_repo_audit" in str(session.user_id if hasattr(session, 'user_id') else "")


# ─────────────────────────────────────────────────────────────────────────────
# Concurrency limiting
# ─────────────────────────────────────────────────────────────────────────────

class TestConcurrencyLimiting:

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrent_runs(self):
        """With max_concurrent=1, two tasks queued at same time should run sequentially."""
        execution_log: list[str] = []

        async def slow_run(session, msg):
            name = session.id  # task name embedded in session id
            execution_log.append(f"start:{name}")
            await asyncio.sleep(0.1)
            execution_log.append(f"end:{name}")
            return _make_turn_result(succeeded=True)

        orc = MagicMock()
        orc.run_turn = slow_run

        task_a = _make_task("a", timeout_s=5)
        task_b = _make_task("b", timeout_s=5)

        s = TaskScheduler(
            orchestrator=orc,
            memory_manager=_make_memory(),
            tasks=[task_a, task_b],
            max_concurrent_tasks=1,
        )

        # Fire both tasks concurrently
        await asyncio.gather(
            s._run_task(task_a),
            s._run_task(task_b),
        )

        # With semaphore=1, they must not overlap:
        # end:a must appear before start:b OR end:b before start:a
        starts = [e for e in execution_log if e.startswith("start:")]
        ends = [e for e in execution_log if e.startswith("end:")]
        assert len(starts) == 2
        assert len(ends) == 2
        # The first task's end must precede the second task's start
        first_end_idx = execution_log.index(ends[0])
        second_start_idx = execution_log.index(starts[1])
        assert first_end_idx < second_start_idx


# ─────────────────────────────────────────────────────────────────────────────
# _next_run_utc helper
# ─────────────────────────────────────────────────────────────────────────────

class TestNextRunUtc:

    def test_returns_future_datetime(self):
        pytest.importorskip("croniter")
        now = datetime.now(timezone.utc)
        nxt = _next_run_utc("0 7 * * *", now)
        assert nxt > now

    def test_result_is_timezone_aware(self):
        pytest.importorskip("croniter")
        now = datetime.now(timezone.utc)
        nxt = _next_run_utc("0 7 * * *", now)
        assert nxt.tzinfo is not None

    def test_raises_import_error_without_croniter(self):
        now = datetime.now(timezone.utc)
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "croniter":
                raise ImportError("No module named croniter")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="croniter"):
                _next_run_utc("0 7 * * *", now)
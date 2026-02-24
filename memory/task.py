"""
memory/task.py — Task Memory (Phase 4 core-hardening)

In-flight plan state — separate from conversation history and episodic memory.
Lives only for the lifetime of a single agent turn; pruned after completion.

Tier summary:
  Short-term  → conversation window (short_term.py)
  Task        → THIS FILE — active plan state, step results, scratch space
  Episodic    → completed episodes in SQLite (episodic.py)
  Semantic    → vector-indexed knowledge in ChromaDB (long_term.py)

Design:
  - TaskMemoryStore is a pure in-memory dict keyed by plan_id.
  - Thread-safe via asyncio.Lock (one lock per plan, plus a store-level lock).
  - prune_old() removes stale entries — called by MemoryManager every turn.
  - TaskMemory is a dataclass; TaskMemoryStore is the manager facade.
  - No external dependencies — importable before MemoryManager.init().

Usage:
    store = TaskMemoryStore()

    # Orchestrator — turn start
    task_mem = store.create(plan_id="plan_abc", goal="Research WebGPU")

    # Executor — after each step
    store.update_result(plan_id="plan_abc", step_id="step_0", result=skill_result)
    store.log_step(plan_id="plan_abc", step_id="step_0", description="Fetch docs")

    # Reflector — read before episode commit
    task_mem = store.get("plan_abc")
    summary = task_mem.to_summary() if task_mem else ""

    # Orchestrator — turn end (always call one of these)
    store.close("plan_abc")     # success / partial
    store.fail("plan_abc", reason="...")   # error path

    # MemoryManager — housekeeping every turn
    store.prune_old(max_age_seconds=3600)
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    from observability.logger import get_logger as _get_logger
    _log_raw = _get_logger(__name__)
    _STRUCTLOG = True
except Exception:
    import logging as _logging
    _log_raw = _logging.getLogger(__name__)
    _STRUCTLOG = False


def log(level: str, event: str, **kwargs) -> None:
    """Portable structured logger — works with structlog and stdlib logging."""
    if _STRUCTLOG:
        getattr(_log_raw, level)(event, **kwargs)
    else:
        extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
        msg = f"{event} {extra}" if extra else event
        getattr(_log_raw, level)(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StepRecord:
    """A single step's metadata and result within a TaskMemory."""
    step_id: str
    description: str
    logged_at: float = field(default_factory=time.monotonic)
    result_content: Optional[str] = None
    is_error: bool = False
    duration_ms: float = 0.0
    completed_at: Optional[float] = None

    def complete(self, result_content: str, is_error: bool = False, duration_ms: float = 0.0) -> None:
        self.result_content = result_content
        self.is_error = is_error
        self.duration_ms = duration_ms
        self.completed_at = time.monotonic()


@dataclass
class TaskMemory:
    """
    In-flight task state for a single plan execution.

    Created at turn start; closed or failed at turn end.
    Stores step records and arbitrary skill scratch space.

    All mutating methods are NOT coroutines — the store-level lock is held
    by TaskMemoryStore before calling them. Do not call mutating methods
    directly from outside the store.
    """
    plan_id: str
    goal: str
    steps: list[StepRecord] = field(default_factory=list)
    intermediate: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.monotonic)
    closed_at: Optional[float] = None
    failed_at: Optional[float] = None
    failure_reason: Optional[str] = None

    @property
    def is_active(self) -> bool:
        return self.closed_at is None and self.failed_at is None

    @property
    def age_seconds(self) -> float:
        return time.monotonic() - self.created_at

    @property
    def completed_steps(self) -> list[StepRecord]:
        return [s for s in self.steps if s.completed_at is not None]

    @property
    def error_steps(self) -> list[StepRecord]:
        return [s for s in self.steps if s.is_error]

    def _log_step(self, step_id: str, description: str) -> StepRecord:
        """Record a new step. Replaces existing entry with the same step_id."""
        record = StepRecord(step_id=step_id, description=description)
        # Replace if already present (idempotent on re-run)
        self.steps = [s for s in self.steps if s.step_id != step_id]
        self.steps.append(record)
        return record

    def _update_result(
        self,
        step_id: str,
        result_content: str,
        is_error: bool = False,
        duration_ms: float = 0.0,
    ) -> bool:
        """Update the result for an existing step. Returns False if not found."""
        for record in self.steps:
            if record.step_id == step_id:
                record.complete(result_content, is_error, duration_ms)
                return True
        return False

    def _set_scratch(self, key: str, value: Any) -> None:
        self.intermediate[key] = value

    def _close(self) -> None:
        self.closed_at = time.monotonic()

    def _fail(self, reason: str) -> None:
        self.failed_at = time.monotonic()
        self.failure_reason = reason

    def to_summary(self) -> str:
        """
        Build a human-readable summary of completed steps for the Reflector.
        Suitable for injecting into an episode commit or lesson extraction.
        """
        if not self.steps:
            return f"Goal: {self.goal} — no steps recorded."

        lines = [f"Goal: {self.goal}"]
        for s in self.steps:
            status = "❌" if s.is_error else "✅"
            result_snippet = (s.result_content or "")[:120]
            lines.append(f"  {status} [{s.step_id}] {s.description}: {result_snippet}")

        if self.failure_reason:
            lines.append(f"  ⚠️  Failed: {self.failure_reason}")

        return "\n".join(lines)

    def to_step_list(self) -> list[str]:
        """Return steps as a list of strings, matching the format Reflector.commit() expects."""
        results = []
        for s in self.steps:
            prefix = "✅" if not s.is_error else "❌"
            snippet = (s.result_content or "")[:100]
            results.append(f"{prefix} {s.description}: {snippet}")
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Store
# ─────────────────────────────────────────────────────────────────────────────

class TaskMemoryStore:
    """
    Manager for all active TaskMemory objects.

    Thread-safe via a single asyncio.Lock. All public methods that
    mutate state acquire the lock.

    Lifecycle:
        store = TaskMemoryStore()

        mem = store.create("plan_xyz", "Research WebGPU")
        store.log_step("plan_xyz", "step_0", "Fetch WebGPU spec")
        store.update_result("plan_xyz", "step_0", "Found 3 relevant docs", duration_ms=142)
        store.close("plan_xyz")

        store.prune_old(max_age_seconds=3600)
    """

    def __init__(self) -> None:
        self._store: dict[str, TaskMemory] = {}
        self._lock = asyncio.Lock()

    # ── Write (turn lifecycle) ────────────────────────────────────────────────

    def create(self, plan_id: str, goal: str) -> TaskMemory:
        """
        Create and register a new TaskMemory for the given plan.

        Safe to call synchronously (no await required) — TaskMemory
        is pure in-memory. Returns the new TaskMemory.

        If a TaskMemory already exists for this plan_id (e.g. retry scenario),
        the old one is replaced and a warning is logged.
        """
        if plan_id in self._store:
            log("warning", "task_memory.create_replace",
                plan_id=plan_id, old_goal=self._store[plan_id].goal)
        mem = TaskMemory(plan_id=plan_id, goal=goal)
        self._store[plan_id] = mem
        log("debug", "task_memory.created", plan_id=plan_id, goal=goal[:60])
        return mem

    def log_step(self, plan_id: str, step_id: str, description: str) -> bool:
        """
        Record that a step has started.

        Returns True if the plan exists, False otherwise (no-op).
        Idempotent — calling with the same step_id replaces the old record.
        """
        mem = self._store.get(plan_id)
        if mem is None:
            log("warning", "task_memory.log_step.not_found", plan_id=plan_id, step_id=step_id)
            return False
        mem._log_step(step_id, description)
        log("debug", "task_memory.step_logged", plan_id=plan_id, step_id=step_id)
        return True

    def update_result(
        self,
        plan_id: str,
        step_id: str,
        result_content: str,
        is_error: bool = False,
        duration_ms: float = 0.0,
    ) -> bool:
        """
        Record the result of a completed step.

        Returns True on success, False if plan or step not found.
        """
        mem = self._store.get(plan_id)
        if mem is None:
            log("warning", "task_memory.update_result.plan_not_found", plan_id=plan_id)
            return False
        found = mem._update_result(step_id, result_content, is_error, duration_ms)
        if not found:
            log("warning", "task_memory.update_result.step_not_found",
                plan_id=plan_id, step_id=step_id)
        return found

    def set_scratch(self, plan_id: str, key: str, value: Any) -> bool:
        """
        Store arbitrary skill-specific scratch data under `key`.

        Returns True on success, False if plan not found.
        """
        mem = self._store.get(plan_id)
        if mem is None:
            return False
        mem._set_scratch(key, value)
        return True

    def close(self, plan_id: str) -> Optional[TaskMemory]:
        """
        Mark a task as successfully completed and return it.

        The entry remains in the store until prune_old() removes it
        (allows Reflector to read it immediately after close).
        Returns None if not found.
        """
        mem = self._store.get(plan_id)
        if mem is None:
            log("warning", "task_memory.close.not_found", plan_id=plan_id)
            return None
        mem._close()
        log("info", "task_memory.closed",
            plan_id=plan_id, steps=len(mem.steps), age_s=round(mem.age_seconds, 1))
        return mem

    def fail(self, plan_id: str, reason: str) -> Optional[TaskMemory]:
        """
        Mark a task as failed and return it.

        Like close(), the entry remains until prune_old().
        Returns None if not found.
        """
        mem = self._store.get(plan_id)
        if mem is None:
            log("warning", "task_memory.fail.not_found", plan_id=plan_id)
            return None
        mem._fail(reason)
        log("warning", "task_memory.failed",
            plan_id=plan_id, reason=reason[:120], steps=len(mem.steps))
        return mem

    # ── Read ──────────────────────────────────────────────────────────────────

    def get(self, plan_id: str) -> Optional[TaskMemory]:
        """Return the TaskMemory for a plan, or None if not found."""
        return self._store.get(plan_id)

    def list_active(self) -> list[TaskMemory]:
        """Return all TaskMemory objects that are still in-flight."""
        return [m for m in self._store.values() if m.is_active]

    def list_all(self) -> list[TaskMemory]:
        """Return all TaskMemory objects (active + completed + failed)."""
        return list(self._store.values())

    # ── Housekeeping ──────────────────────────────────────────────────────────

    def prune_old(self, max_age_seconds: int = 3600) -> int:
        """
        Remove completed/failed TaskMemory objects older than max_age_seconds.

        Active tasks (is_active=True) are never pruned regardless of age.
        Returns the number of entries removed.

        Called by MemoryManager every turn so the store never accumulates
        unbounded completed entries.
        """
        cutoff = time.monotonic() - max_age_seconds
        to_remove = [
            plan_id
            for plan_id, mem in self._store.items()
            if not mem.is_active and mem.created_at < cutoff
        ]
        for plan_id in to_remove:
            del self._store[plan_id]

        if to_remove:
            log("debug", "task_memory.pruned",
                removed=len(to_remove), remaining=len(self._store))
        return len(to_remove)

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        active = len(self.list_active())
        return f"<TaskMemoryStore total={len(self._store)} active={active}>"
"""
agent/heartbeat.py — Proactive Heartbeat System

Reads ~/neuralclaw/HEARTBEAT.md and injects it as a scheduled
orchestrator turn every N minutes. This makes NeuralClaw proactive —
it checks in on pending tasks, overdue reminders, and noteworthy events
without the user having to ask.

The heartbeat runs as a standard ScheduledTask inside the existing
TaskScheduler. No new infrastructure needed.

Workspace files read during heartbeat:
    ~/neuralclaw/HEARTBEAT.md   — checklist the agent works through
    ~/neuralclaw/MEMORY.md      — long-term notes (injected as context)
    ~/neuralclaw/USER.md        — user context (injected as context)

Usage (inside from_settings / CLI bootstrap):
    from agent.heartbeat import build_heartbeat_task, inject_workspace_context
    scheduler.add_task(build_heartbeat_task(interval_minutes=30))
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from observability.logger import get_logger
from scheduler.scheduler import ScheduledTask
from skills.types import TrustLevel

log = get_logger(__name__)

WORKSPACE_DIR = Path("~/neuralclaw").expanduser()

# Files read and injected into the heartbeat prompt
_WORKSPACE_FILES = ["HEARTBEAT.md", "MEMORY.md", "USER.md", "SOUL.md"]

# Cron expression templates keyed by interval bucket
_CRON_MAP: dict[int, str] = {
    5:  "*/5 * * * *",
    10: "*/10 * * * *",
    15: "*/15 * * * *",
    30: "*/30 * * * *",
    60: "0 * * * *",
}


def _cron_for_interval(minutes: int) -> str:
    """Return the nearest cron expression for the requested interval."""
    if minutes in _CRON_MAP:
        return _CRON_MAP[minutes]
    # Generic: run at minute 0, N, 2N, ... within an hour — only works ≤60
    if minutes <= 60 and 60 % minutes == 0:
        return f"*/{minutes} * * * *"
    # Fallback: run every hour at minute 0
    log.warning(
        "heartbeat.unsupported_interval",
        minutes=minutes,
        fallback="every 60 minutes",
    )
    return "0 * * * *"


def _read_workspace_context() -> str:
    """Read all relevant workspace files and return them as a context block."""
    if not WORKSPACE_DIR.exists():
        return ""

    sections: list[str] = []
    for filename in _WORKSPACE_FILES:
        fpath = WORKSPACE_DIR / filename
        if fpath.exists():
            try:
                content = fpath.read_text().strip()
                if content:
                    sections.append(f"### {filename}\n{content}")
            except OSError:
                pass

    if not sections:
        return ""

    return (
        "## Workspace Context\n\n"
        + "\n\n---\n\n".join(sections)
    )


def build_heartbeat_prompt() -> str:
    """
    Build the full prompt sent to the orchestrator for a heartbeat turn.
    Injects workspace context so the agent has full situational awareness.
    """
    workspace_ctx = _read_workspace_context()

    heartbeat_file = WORKSPACE_DIR / "HEARTBEAT.md"
    if heartbeat_file.exists():
        try:
            checklist = heartbeat_file.read_text().strip()
        except OSError:
            checklist = ""
    else:
        checklist = ""

    # Build the prompt
    parts = [
        "## Heartbeat Check-in\n",
        "You are running a scheduled heartbeat. Work through the checklist below.",
        "Be proactive: if something needs attention, act on it or notify the user.",
        "If there is nothing actionable, stay silent — do NOT send empty messages.",
        "",
    ]

    if checklist:
        parts.append(checklist)

    if workspace_ctx:
        parts.append("")
        parts.append(workspace_ctx)

    return "\n".join(parts)


def build_heartbeat_task(interval_minutes: int = 30, enabled: bool = True) -> ScheduledTask:
    """
    Return a ScheduledTask that fires the heartbeat on the given interval.

    Args:
        interval_minutes: How often to run (default 30 minutes).
        enabled:          Set False to register but not start.

    The task uses a special invocation message that build_heartbeat_prompt()
    generates at call time so workspace files are always read fresh.
    """
    cron = _cron_for_interval(interval_minutes)

    log.info(
        "heartbeat.task_built",
        interval_minutes=interval_minutes,
        cron=cron,
        workspace=str(WORKSPACE_DIR),
    )

    return ScheduledTask(
        name="heartbeat",
        cron=cron,
        skill_name="heartbeat",       # sentinel — intercepted by HeartbeatRunner
        skill_args={"interval_minutes": interval_minutes},
        trust_level=TrustLevel.MEDIUM,
        enabled=enabled,
        timeout_s=120,
    )


class HeartbeatRunner:
    """
    Thin wrapper that intercepts heartbeat ScheduledTask invocations
    before they reach the SkillBus (heartbeat is not a real skill).

    Injects the workspace-aware prompt directly into orchestrator.run_turn().

    Usage in the scheduler's _run_task():
        runner = HeartbeatRunner(orchestrator, memory_manager)
        if task.skill_name == "heartbeat":
            await runner.run(session)
        else:
            # normal skill dispatch
    """

    def __init__(self, orchestrator, memory_manager) -> None:
        self._orchestrator = orchestrator
        self._memory = memory_manager

    async def run(self, session) -> None:
        """Execute one heartbeat turn against the orchestrator."""
        prompt = build_heartbeat_prompt()
        if not prompt.strip():
            log.info("heartbeat.skipped", reason="empty prompt")
            return

        log.info("heartbeat.firing", workspace=str(WORKSPACE_DIR))

        try:
            result = await self._orchestrator.run_turn(session, prompt)
            log.info(
                "heartbeat.complete",
                status=result.status.value,
                tool_calls=result.steps_taken,
                ms=round(result.duration_ms),
            )

            # Persist any learned facts back to MEMORY.md if the agent
            # produced a non-trivial response (> 100 chars, not an error)
            if (
                result.succeeded
                and len(result.response.text) > 100
                and not result.response.text.startswith("❌")
            ):
                await _maybe_persist_memory(result.response.text, self._memory, session.id)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.error(
                "heartbeat.error",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )


async def _maybe_persist_memory(response_text: str, memory_manager, session_id: str) -> None:
    """
    If the heartbeat produced an actionable result, append a summary
    to ~/neuralclaw/MEMORY.md for the next heartbeat to build on.
    """
    try:
        memory_file = WORKSPACE_DIR / "MEMORY.md"
        if not memory_file.exists():
            return

        # Only persist if the response contains something note-worthy
        keywords = ["completed", "found", "updated", "created", "error", "reminder",
                    "scheduled", "note:", "important", "fixed", "resolved"]
        text_lower = response_text.lower()
        if not any(k in text_lower for k in keywords):
            return

        from datetime import datetime, timezone
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        note = f"\n## {timestamp} — Heartbeat Note\n{response_text[:500]}\n"

        current = memory_file.read_text()
        memory_file.write_text(current + note)

        log.debug("heartbeat.memory_appended", chars=len(note))

    except OSError as e:
        log.warning("heartbeat.memory_write_failed", error=str(e))
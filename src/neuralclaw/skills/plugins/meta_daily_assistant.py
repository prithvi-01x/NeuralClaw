"""
skills/plugins/meta_daily_assistant.py — Meta-Skill: Daily Assistant

Fixed multi-step morning briefing workflow. Orchestrates:
    personal_weather_fetch → personal_calendar_read → personal_news_digest →
    personal_task_manager → personal_reminder_set (optional) → automation_report_render

Produces a single Markdown briefing file that the scheduler runs at 07:00 daily.
All steps logged to TaskMemory. The orchestrator sees this as one plan step.

Risk: LOW — data:read + net:fetch + fs:write capabilities.
No confirmation required (all low-risk sub-skills).

Phase C meta-skill. No kernel changes required.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, ClassVar

from neuralclaw.skills.base import SkillBase
from neuralclaw.skills.types import RiskLevel, SkillManifest, SkillResult, SkillValidationError


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers (duplicated from meta_recon_pipeline to keep skills self-contained)
# ─────────────────────────────────────────────────────────────────────────────

def _truncate(text: Any, max_len: int = 2000) -> str:
    s = str(text) if not isinstance(text, str) else text
    return s[:max_len] + "…" if len(s) > max_len else s


def _result_summary(result: SkillResult) -> str:
    if not result.success:
        return f"❌ FAILED — {result.error}"
    output = result.output
    if isinstance(output, str):
        return f"✅ {_truncate(output, 120)}"
    if isinstance(output, dict):
        return f"✅ OK — {list(output.keys())}"
    return "✅ OK"


def _skill_call(name: str, args: dict) -> Any:
    from neuralclaw.skills.types import SkillCall
    return SkillCall(
        id=f"meta-daily-{uuid.uuid4().hex[:8]}",
        skill_name=name,
        arguments=args,
    )


async def _dispatch(bus: Any, call: Any, session: Any) -> SkillResult:
    return await bus.dispatch(
        call,
        trust_level=session.trust_level if session else None,
        granted_capabilities=session.granted_capabilities if session else frozenset(),
    )


def _output_to_md(output: Any) -> str:
    """Convert a sub-skill output dict/str to a readable Markdown block."""
    if isinstance(output, str):
        return output
    import json
    try:
        return f"```json\n{json.dumps(output, indent=2, default=str)[:3000]}\n```"
    except Exception:
        return str(output)[:3000]


# ─────────────────────────────────────────────────────────────────────────────
# Meta-Skill
# ─────────────────────────────────────────────────────────────────────────────

class MetaDailyAssistantSkill(SkillBase):
    manifest: ClassVar[SkillManifest] = SkillManifest(
        name="meta_daily_assistant",
        version="1.0.0",
        description=(
            "Morning briefing pipeline. Collects weather, calendar events, top news, "
            "and pending tasks, then renders a consolidated Markdown report. "
            "Designed for daily scheduler execution at 07:00. "
            "Optionally sets a reminder for any time-sensitive calendar events found."
        ),
        category="personal",
        risk_level=RiskLevel.LOW,
        capabilities=frozenset({"data:read", "net:fetch", "fs:write"}),
        requires_confirmation=False,
        timeout_seconds=120,
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City or location for weather fetch (e.g. 'London, UK').",
                    "default": "",
                },
                "news_topics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of news topics/keywords to include in the digest.",
                    "default": [],
                },
                "report_path": {
                    "type": "string",
                    "description": "Output path for the briefing Markdown file.",
                    "default": "~/neuralclaw/reports/daily_briefing.md",
                },
                "skip_steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Steps to skip: weather, calendar, news, tasks, reminders.",
                    "default": [],
                },
                "plan_id": {
                    "type": "string",
                    "description": "TaskMemory plan ID (injected by orchestrator).",
                    "default": "",
                },
            },
            "required": [],
        },
    )

    async def execute(
        self,
        location: str = "",
        news_topics: list[str] | None = None,
        report_path: str = "~/neuralclaw/reports/daily_briefing.md",
        skip_steps: list[str] | None = None,
        plan_id: str = "",
        **kwargs,
    ) -> SkillResult:
        call_id = kwargs.get("_skill_call_id", "")
        t_start = time.monotonic()
        skip = set(skip_steps or [])
        plan_id = plan_id or f"daily-{uuid.uuid4().hex[:8]}"

        bus = kwargs.get("_bus")
        session = kwargs.get("_session")
        task_store = kwargs.get("_task_store")

        if not bus:
            return SkillResult.fail(
                self.manifest.name, call_id,
                "meta_daily_assistant requires '_bus' injected. "
                "Ensure the orchestrator passes _bus to meta-skill execute() calls.",
                "MetaSkillConfigError",
            )

        if task_store:
            task_store.create(plan_id=plan_id, goal="Daily morning briefing")

        def log_step(step_id: str, desc: str) -> None:
            if task_store:
                task_store.log_step(plan_id, step_id, desc)

        def log_result(step_id: str, result: SkillResult) -> None:
            if task_store:
                task_store.update_result(
                    plan_id, step_id,
                    result_content=_result_summary(result),
                    is_error=not result.success,
                    duration_ms=result.duration_ms,
                )

        step_results: dict[str, SkillResult] = {}

        # Step 1: Weather
        if "weather" not in skip:
            log_step("weather", f"Fetching weather{' for ' + location if location else ''}")
            args: dict[str, Any] = {}
            if location:
                args["location"] = location
            r = await _dispatch(bus, _skill_call("personal_weather_fetch", args), session)
            log_result("weather", r)
            step_results["weather"] = r

        # Step 2: Calendar
        if "calendar" not in skip:
            log_step("calendar", "Fetching today's calendar events")
            r = await _dispatch(bus, _skill_call("personal_calendar_read", {"days_ahead": 1}), session)
            log_result("calendar", r)
            step_results["calendar"] = r

        # Step 3: News
        if "news" not in skip:
            log_step("news", "Fetching news digest")
            news_args: dict[str, Any] = {}
            if news_topics:
                news_args["topics"] = news_topics
            r = await _dispatch(bus, _skill_call("personal_news_digest", news_args), session)
            log_result("news", r)
            step_results["news"] = r

        # Step 4: Tasks
        if "tasks" not in skip:
            log_step("tasks", "Fetching pending tasks")
            r = await _dispatch(bus, _skill_call("personal_task_manager", {"action": "list", "filter": "pending"}), session)
            log_result("tasks", r)
            step_results["tasks"] = r

        # Step 5: Reminders (optional — only if calendar returned urgent events)
        if "reminders" not in skip:
            cal_result = step_results.get("calendar")
            if cal_result and cal_result.success:
                events = []
                if isinstance(cal_result.output, dict):
                    events = cal_result.output.get("events", [])
                if events:
                    # Set a reminder for the first event of the day
                    first = events[0]
                    reminder_text = first.get("title", "Calendar event") if isinstance(first, dict) else str(first)
                    log_step("reminders", f"Setting reminder: {reminder_text}")
                    r = await _dispatch(bus, _skill_call("personal_reminder_set", {
                        "message": reminder_text,
                        "remind_at": first.get("start", "") if isinstance(first, dict) else "",
                    }), session)
                    log_result("reminders", r)
                    step_results["reminders"] = r

        # ── Build report sections ────────────────────────────────────────────
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%A, %d %B %Y")

        sections = [
            {"heading": "Overview", "content": f"Daily briefing for **{today}**.", "level": 2},
        ]

        step_labels = {
            "weather": "🌤 Weather",
            "calendar": "📅 Calendar",
            "news": "📰 News Digest",
            "tasks": "✅ Pending Tasks",
            "reminders": "⏰ Reminders Set",
        }
        for step_id, label in step_labels.items():
            if step_id in skip:
                continue
            r = step_results.get(step_id)
            if r is None:
                continue
            if not r.success:
                content = f"**Status:** ❌ Failed\n\n**Error:** `{r.error}`"
            else:
                content = _output_to_md(r.output)
            sections.append({"heading": label, "content": content, "level": 2})

        # ── Write report ─────────────────────────────────────────────────────
        log_step("report_render", "Writing daily briefing report")
        report_result = await _dispatch(bus, _skill_call("automation_report_render", {
            "title": f"Daily Briefing — {today}",
            "sections": sections,
            "output_path": report_path,
        }), session)
        log_result("report_render", report_result)

        if task_store:
            task_store.close(plan_id)

        duration_ms = (time.monotonic() - t_start) * 1000
        success_count = sum(1 for r in step_results.values() if r.success)

        return SkillResult.ok(
            skill_name=self.manifest.name,
            skill_call_id=call_id,
            output={
                "date": today,
                "steps_completed": success_count,
                "steps_total": len(step_results),
                "steps_skipped": list(skip),
                "report_path": report_path,
                "report_written": report_result.success,
                "summary": {k: _result_summary(v) for k, v in step_results.items()},
            },
            duration_ms=duration_ms,
        )
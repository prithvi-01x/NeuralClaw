"""
skills/plugins/personal_calendar_read.py — Personal: Calendar Read

Reads calendar events from a local iCalendar (.ics) file or a directory of .ics files.
No external API required — works with any calendar exported to .ics format
(Google Calendar, Nextcloud, Apple Calendar, Thunderbird, etc.).

Risk: LOW — fs:read capability required.
"""

from __future__ import annotations

import asyncio
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import ClassVar, Optional

from neuralclaw.skills.base import SkillBase
from neuralclaw.skills.types import RiskLevel, SkillManifest, SkillResult, SkillValidationError


class PersonalCalendarReadSkill(SkillBase):
    manifest: ClassVar[SkillManifest] = SkillManifest(
        name="personal_calendar_read",
        version="1.0.0",
        description=(
            "Read upcoming calendar events from a local .ics file or directory. "
            "Returns events sorted by start time within the requested date range. "
            "Supports iCalendar format (Google, Apple, Nextcloud, Thunderbird exports)."
        ),
        category="personal",
        risk_level=RiskLevel.LOW,
        capabilities=frozenset({"fs:read"}),
        requires_confirmation=False,
        timeout_seconds=15,
        parameters={
            "type": "object",
            "properties": {
                "calendar_path": {
                    "type": "string",
                    "description": "Path to .ics file or directory containing .ics files.",
                },
                "days_ahead": {
                    "type": "integer",
                    "description": "How many days ahead to look for events (default 7, max 90).",
                    "default": 7,
                },
                "days_back": {
                    "type": "integer",
                    "description": "How many days back to include (default 0 = today onwards).",
                    "default": 0,
                },
                "max_events": {
                    "type": "integer",
                    "description": "Maximum number of events to return (default 20).",
                    "default": 20,
                },
            },
            "required": ["calendar_path"],
        },
    )

    async def validate(self, calendar_path: str, days_ahead: int = 7, **_) -> None:
        p = Path(calendar_path).expanduser()
        if not p.exists():
            raise SkillValidationError(f"Calendar path does not exist: '{calendar_path}'")
        if p.is_file() and p.suffix.lower() != ".ics":
            raise SkillValidationError(f"File must be a .ics file, got: '{p.suffix}'")
        if days_ahead > 90:
            raise SkillValidationError("days_ahead must be ≤ 90.")

    async def execute(
        self,
        calendar_path: str,
        days_ahead: int = 7,
        days_back: int = 0,
        max_events: int = 20,
        **kwargs,
    ) -> SkillResult:
        call_id = kwargs.get("_skill_call_id", "")
        t_start = time.monotonic()
        days_ahead = min(int(days_ahead), 90)
        max_events = min(int(max_events), 100)

        try:
            import icalendar  # type: ignore
        except ImportError:
            return SkillResult.fail(
                self.manifest.name, call_id,
                "icalendar package not installed. Run: pip install icalendar",
                "ImportError",
            )

        try:
            p = Path(calendar_path).expanduser()
            ics_files: list[Path] = []
            if p.is_dir():
                ics_files = sorted(p.glob("**/*.ics"))
            else:
                ics_files = [p]

            if not ics_files:
                return SkillResult.fail(
                    self.manifest.name, call_id,
                    f"No .ics files found in '{calendar_path}'", "FileNotFoundError",
                )

            now_utc = datetime.now(tz=timezone.utc)
            range_start = now_utc - timedelta(days=days_back)
            range_end = now_utc + timedelta(days=days_ahead)

            events: list[dict] = []

            def _parse_files() -> None:
                for ics_path in ics_files:
                    try:
                        cal = icalendar.Calendar.from_ical(ics_path.read_bytes())
                        for component in cal.walk():
                            if component.name != "VEVENT":
                                continue
                            dtstart = component.get("DTSTART")
                            if dtstart is None:
                                continue
                            start = dtstart.dt
                            # Normalize to datetime with tz
                            if isinstance(start, date) and not isinstance(start, datetime):
                                start = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
                            elif isinstance(start, datetime) and start.tzinfo is None:
                                start = start.replace(tzinfo=timezone.utc)
                            if not (range_start <= start <= range_end):
                                continue
                            dtend = component.get("DTEND")
                            end = None
                            if dtend:
                                end = dtend.dt
                                if isinstance(end, date) and not isinstance(end, datetime):
                                    end = datetime(end.year, end.month, end.day, tzinfo=timezone.utc)
                                elif isinstance(end, datetime) and end.tzinfo is None:
                                    end = end.replace(tzinfo=timezone.utc)
                            events.append({
                                "summary": str(component.get("SUMMARY", "No title")),
                                "start": start.isoformat(),
                                "end": end.isoformat() if end else None,
                                "location": str(component.get("LOCATION", "")),
                                "description": str(component.get("DESCRIPTION", ""))[:300],
                                "all_day": not isinstance(dtstart.dt, datetime),
                                "calendar_file": ics_path.name,
                            })
                    except (ValueError, KeyError, AttributeError):
                        continue

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _parse_files)

            events.sort(key=lambda e: e["start"])
            events = events[:max_events]

            duration_ms = (time.monotonic() - t_start) * 1000
            return SkillResult.ok(
                skill_name=self.manifest.name,
                skill_call_id=call_id,
                output={
                    "range_start": range_start.isoformat(),
                    "range_end": range_end.isoformat(),
                    "event_count": len(events),
                    "events": events,
                    "calendars_read": len(ics_files),
                },
                duration_ms=duration_ms,
            )

        except (OSError, ValueError) as e:
            return SkillResult.fail(
                self.manifest.name, call_id,
                f"{type(e).__name__}: {e}", type(e).__name__,
                duration_ms=(time.monotonic() - t_start) * 1000,
            )
        except BaseException as e:
            return SkillResult.fail(
                self.manifest.name, call_id,
                f"{type(e).__name__}: {e}", type(e).__name__,
                duration_ms=(time.monotonic() - t_start) * 1000,
            )
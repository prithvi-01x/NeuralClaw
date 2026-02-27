"""
skills/plugins/personal_reminder_set.py — Personal: Reminder Set

Stores reminders in ~/neuralclaw/reminders.json.
Supports set, list, delete, and check (returns due reminders).

Risk: LOW — data:write
"""
from __future__ import annotations
import asyncio, json, time, uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar
from neuralclaw.skills.base import SkillBase
from neuralclaw.skills.types import RiskLevel, SkillManifest, SkillResult, SkillValidationError

_REMINDERS_FILE = Path("~/neuralclaw/reminders.json").expanduser()

class PersonalReminderSetSkill(SkillBase):
    manifest: ClassVar[SkillManifest] = SkillManifest(
        name="personal_reminder_set",
        version="1.0.0",
        description="Set, list, delete, and check reminders. Stored locally in ~/neuralclaw/reminders.json.",
        category="personal",
        risk_level=RiskLevel.LOW,
        capabilities=frozenset({"data:write"}),
        timeout_seconds=10,
        parameters={"type":"object","properties":{
            "action":{"type":"string","enum":["set","list","delete","check"],"description":"Action to perform."},
            "message":{"type":"string","description":"Reminder message (required for set).","default":""},
            "due_at":{"type":"string","description":"ISO datetime when reminder is due (required for set, e.g. '2025-03-01T09:00:00').","default":""},
            "reminder_id":{"type":"string","description":"Reminder ID (required for delete).","default":""},
        },"required":["action"]},
    )

    async def validate(self, action: str, message: str="", due_at: str="", reminder_id: str="", **_) -> None:
        if action == "set":
            if not message.strip():
                raise SkillValidationError("message is required for action='set'.")
            if not due_at.strip():
                raise SkillValidationError("due_at is required for action='set'.")
            try: datetime.fromisoformat(due_at)
            except ValueError: raise SkillValidationError(f"due_at '{due_at}' is not a valid ISO datetime.")
        if action == "delete" and not reminder_id.strip():
            raise SkillValidationError("reminder_id is required for action='delete'.")

    async def execute(self, action: str, message: str="", due_at: str="", reminder_id: str="", **kwargs) -> SkillResult:
        call_id = kwargs.get("_skill_call_id","")
        t_start = time.monotonic()

        def _load() -> list[dict]:
            if _REMINDERS_FILE.exists():
                try: return json.loads(_REMINDERS_FILE.read_text())
                except Exception: return []
            return []

        def _save(reminders: list[dict]) -> None:
            _REMINDERS_FILE.parent.mkdir(parents=True, exist_ok=True)
            _REMINDERS_FILE.write_text(json.dumps(reminders, indent=2))

        try:
            loop = asyncio.get_event_loop()
            reminders = await loop.run_in_executor(None, _load)

            if action == "set":
                r = {"id": str(uuid.uuid4())[:8], "message": message.strip(), "due_at": due_at,
                     "created": datetime.now(tz=timezone.utc).isoformat(), "triggered": False}
                reminders.append(r)
                await loop.run_in_executor(None, _save, reminders)
                return SkillResult.ok(self.manifest.name, call_id, {"set": r})

            elif action == "list":
                return SkillResult.ok(self.manifest.name, call_id, {"count":len(reminders),"reminders":reminders})

            elif action == "delete":
                before = len(reminders)
                reminders = [r for r in reminders if r["id"] != reminder_id]
                if len(reminders) == before:
                    return SkillResult.fail(self.manifest.name, call_id, f"Reminder '{reminder_id}' not found.")
                await loop.run_in_executor(None, _save, reminders)
                return SkillResult.ok(self.manifest.name, call_id, {"deleted_id": reminder_id})

            elif action == "check":
                now = datetime.now(tz=timezone.utc).isoformat()
                due = [r for r in reminders if not r.get("triggered") and r["due_at"] <= now]
                for r in due: r["triggered"] = True
                if due: await loop.run_in_executor(None, _save, reminders)
                return SkillResult.ok(self.manifest.name, call_id, {"due_count":len(due),"due":due})

            return SkillResult.fail(self.manifest.name, call_id, f"Unknown action: {action}")
        except BaseException as e:
            return SkillResult.fail(self.manifest.name, call_id, f"{type(e).__name__}: {e}", type(e).__name__,
                                    duration_ms=(time.monotonic()-t_start)*1000)

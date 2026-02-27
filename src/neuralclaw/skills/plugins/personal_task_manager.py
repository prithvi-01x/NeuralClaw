"""
skills/plugins/personal_task_manager.py — Personal: Task Manager

Simple local task list stored as JSON in ~/neuralclaw/tasks.json.
Supports add, list, complete, delete, search.

Risk: LOW — data:read, data:write
"""
from __future__ import annotations
import asyncio, json, time, uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar
from neuralclaw.skills.base import SkillBase
from neuralclaw.skills.types import RiskLevel, SkillManifest, SkillResult, SkillValidationError

_TASKS_FILE = Path("~/neuralclaw/tasks.json").expanduser()

class PersonalTaskManagerSkill(SkillBase):
    manifest: ClassVar[SkillManifest] = SkillManifest(
        name="personal_task_manager",
        version="1.0.0",
        description="Manage a local task list (add/list/complete/delete/search). Tasks stored in ~/neuralclaw/tasks.json.",
        category="personal",
        risk_level=RiskLevel.LOW,
        capabilities=frozenset({"data:read","data:write"}),
        timeout_seconds=10,
        parameters={"type":"object","properties":{
            "action":{"type":"string","enum":["add","list","complete","delete","search"],"description":"Action to perform."},
            "title":{"type":"string","description":"Task title (required for add).","default":""},
            "task_id":{"type":"string","description":"Task ID (required for complete/delete).","default":""},
            "query":{"type":"string","description":"Search query (for search action).","default":""},
            "priority":{"type":"string","enum":["low","medium","high"],"description":"Task priority (for add).","default":"medium"},
            "due":{"type":"string","description":"Due date ISO string (for add, optional).","default":""},
            "show_completed":{"type":"boolean","description":"Include completed tasks in list (default false).","default":False},
        },"required":["action"]},
    )

    async def validate(self, action: str, title: str="", task_id: str="", **_) -> None:
        if action == "add" and not title.strip():
            raise SkillValidationError("title is required for action='add'.")
        if action in ("complete","delete") and not task_id.strip():
            raise SkillValidationError(f"task_id is required for action='{action}'.")

    async def execute(self, action: str, title: str="", task_id: str="", query: str="",
                      priority: str="medium", due: str="", show_completed: bool=False, **kwargs) -> SkillResult:
        call_id = kwargs.get("_skill_call_id","")
        t_start = time.monotonic()

        def _load() -> list[dict]:
            if _TASKS_FILE.exists():
                try: return json.loads(_TASKS_FILE.read_text())
                except Exception: return []
            return []

        def _save(tasks: list[dict]) -> None:
            _TASKS_FILE.parent.mkdir(parents=True, exist_ok=True)
            _TASKS_FILE.write_text(json.dumps(tasks, indent=2))

        try:
            loop = asyncio.get_event_loop()
            tasks: list[dict] = await loop.run_in_executor(None, _load)

            if action == "add":
                task = {"id": str(uuid.uuid4())[:8], "title": title.strip(), "priority": priority,
                        "due": due, "completed": False, "created": datetime.now(tz=timezone.utc).isoformat()}
                tasks.append(task)
                await loop.run_in_executor(None, _save, tasks)
                return SkillResult.ok(self.manifest.name, call_id, {"added": task})

            elif action == "list":
                visible = [t for t in tasks if show_completed or not t.get("completed")]
                return SkillResult.ok(self.manifest.name, call_id, {"count": len(visible), "tasks": visible})

            elif action == "complete":
                for t in tasks:
                    if t["id"] == task_id:
                        t["completed"] = True
                        t["completed_at"] = datetime.now(tz=timezone.utc).isoformat()
                        await loop.run_in_executor(None, _save, tasks)
                        return SkillResult.ok(self.manifest.name, call_id, {"completed": t})
                return SkillResult.fail(self.manifest.name, call_id, f"Task '{task_id}' not found.")

            elif action == "delete":
                before = len(tasks)
                tasks = [t for t in tasks if t["id"] != task_id]
                if len(tasks) == before:
                    return SkillResult.fail(self.manifest.name, call_id, f"Task '{task_id}' not found.")
                await loop.run_in_executor(None, _save, tasks)
                return SkillResult.ok(self.manifest.name, call_id, {"deleted_id": task_id})

            elif action == "search":
                q = query.lower()
                matches = [t for t in tasks if q in t.get("title","").lower()]
                return SkillResult.ok(self.manifest.name, call_id, {"count": len(matches), "tasks": matches})

            return SkillResult.fail(self.manifest.name, call_id, f"Unknown action: {action}")
        except BaseException as e:
            return SkillResult.fail(self.manifest.name, call_id, f"{type(e).__name__}: {e}", type(e).__name__,
                                    duration_ms=(time.monotonic()-t_start)*1000)

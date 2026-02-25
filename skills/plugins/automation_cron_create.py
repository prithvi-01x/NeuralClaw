"""
skills/plugins/automation_cron_create.py — Automation: Cron Create

Adds, lists, or removes crontab entries using the system crontab command.
requires_confirmation=True — always prompts before modifying crontab.

Risk: HIGH — system:cron
"""
from __future__ import annotations
import asyncio, re, time
from typing import ClassVar
from skills.base import SkillBase
from skills.types import RiskLevel, SkillManifest, SkillResult, SkillValidationError

_CRON_RE = re.compile(r'^(\*|[0-9,\-/]+)\s+(\*|[0-9,\-/]+)\s+(\*|[0-9,\-/]+)\s+(\*|[0-9,\-/]+)\s+(\*|[0-9,\-/]+)\s+.+')

class AutomationCronCreateSkill(SkillBase):
    manifest: ClassVar[SkillManifest] = SkillManifest(
        name="automation_cron_create",
        version="1.0.0",
        description="Add, list, or remove crontab entries. Requires confirmation before any modification. Validates cron expression syntax before writing.",
        category="automation",
        risk_level=RiskLevel.HIGH,
        capabilities=frozenset({"system:cron"}),
        requires_confirmation=True,
        timeout_seconds=15,
        parameters={"type":"object","properties":{
            "action":{"type":"string","enum":["add","list","remove"],"description":"Action to perform."},
            "schedule":{"type":"string","description":"Cron schedule (e.g. '0 7 * * *'). Required for add.","default":""},
            "command":{"type":"string","description":"Command to run. Required for add.","default":""},
            "comment":{"type":"string","description":"Comment/label for the entry (helps identify it for removal).","default":""},
        },"required":["action"]},
    )

    async def validate(self, action: str, schedule: str="", command: str="", **_) -> None:
        if action == "add":
            if not schedule.strip(): raise SkillValidationError("schedule is required for action='add'.")
            if not command.strip(): raise SkillValidationError("command is required for action='add'.")
            test_line = f"{schedule} {command}"
            if not _CRON_RE.match(test_line):
                raise SkillValidationError(f"Invalid cron expression: '{schedule}'. Expected format: 'min hour dom mon dow'.")

    async def execute(self, action: str, schedule: str="", command: str="", comment: str="", **kwargs) -> SkillResult:
        call_id = kwargs.get("_skill_call_id","")
        t_start = time.monotonic()

        async def _run(cmd): 
            proc = await asyncio.create_subprocess_exec(*cmd,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
            out,err = await asyncio.wait_for(proc.communicate(),timeout=10)
            return out.decode(errors="replace"), err.decode(errors="replace"), proc.returncode

        try:
            if action == "list":
                out, err, rc = await _run(["crontab","-l"])
                if rc != 0 and "no crontab" in err.lower():
                    return SkillResult.ok(self.manifest.name, call_id, {"entries":[],"raw":""})
                entries = [l for l in out.splitlines() if l.strip() and not l.startswith("#")]
                return SkillResult.ok(self.manifest.name, call_id, {"count":len(entries),"entries":entries,"raw":out})

            elif action == "add":
                out, err, rc = await _run(["crontab","-l"])
                existing = "" if (rc != 0 and "no crontab" in err.lower()) else out
                label = f"# neuralclaw: {comment}" if comment else "# neuralclaw"
                new_entry = f"\n{label}\n{schedule} {command}\n"
                new_crontab = existing.rstrip() + new_entry
                proc = await asyncio.create_subprocess_exec("crontab","-",stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
                _, err_b = await asyncio.wait_for(proc.communicate(input=new_crontab.encode()),timeout=10)
                if proc.returncode != 0:
                    return SkillResult.fail(self.manifest.name, call_id, f"crontab write failed: {err_b.decode(errors='replace')}")
                return SkillResult.ok(self.manifest.name, call_id, {"added":f"{schedule} {command}","comment":comment})

            elif action == "remove":
                if not comment: return SkillResult.fail(self.manifest.name, call_id, "comment is required to identify which entry to remove.")
                out, err, rc = await _run(["crontab","-l"])
                if rc != 0: return SkillResult.fail(self.manifest.name, call_id, "No crontab found.")
                lines = out.splitlines()
                filtered, removed = [], 0
                skip_next = False
                for line in lines:
                    if comment.lower() in line.lower() and line.startswith("#"):
                        skip_next = True; removed += 1; continue
                    if skip_next:
                        skip_next = False; continue
                    filtered.append(line)
                if removed == 0:
                    return SkillResult.fail(self.manifest.name, call_id, f"No entry found with comment '{comment}'.")
                new_crontab = "\n".join(filtered) + "\n"
                proc = await asyncio.create_subprocess_exec("crontab","-",stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
                await asyncio.wait_for(proc.communicate(input=new_crontab.encode()),timeout=10)
                return SkillResult.ok(self.manifest.name, call_id, {"removed_count":removed,"comment":comment})

            return SkillResult.fail(self.manifest.name, call_id, f"Unknown action: {action}")
        except BaseException as e:
            return SkillResult.fail(self.manifest.name, call_id, f"{type(e).__name__}: {e}", type(e).__name__,
                                    duration_ms=(time.monotonic()-t_start)*1000)

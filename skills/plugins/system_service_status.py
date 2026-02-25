"""
skills/plugins/system_service_status.py — System: Service Status

Checks the status of systemd services (or launchd on macOS).

Risk: LOW — fs:read
"""
from __future__ import annotations
import asyncio, sys, time
from typing import ClassVar
from skills.base import SkillBase
from skills.types import RiskLevel, SkillManifest, SkillResult, SkillValidationError

class SystemServiceStatusSkill(SkillBase):
    manifest: ClassVar[SkillManifest] = SkillManifest(
        name="system_service_status",
        version="1.0.0",
        description="Check the status of one or more system services. Uses systemctl on Linux, launchctl on macOS. Returns active/inactive/failed status.",
        category="system",
        risk_level=RiskLevel.LOW,
        capabilities=frozenset({"fs:read"}),
        timeout_seconds=20,
        parameters={"type":"object","properties":{
            "services":{"type":"array","items":{"type":"string"},"description":"Service names to check (e.g. ['nginx','postgresql']). Leave empty to list all failed services.","default":[]},
            "show_all":{"type":"boolean","description":"List all running services (Linux only, default false).","default":False},
        },"required":[]},
    )

    async def execute(self, services: list|None=None, show_all: bool=False, **kwargs) -> SkillResult:
        call_id = kwargs.get("_skill_call_id","")
        t_start = time.monotonic()
        services = services or []

        async def _run(cmd):
            proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            out, err = await asyncio.wait_for(proc.communicate(), timeout=15)
            return out.decode(errors="replace"), proc.returncode

        try:
            results = []
            if sys.platform.startswith("linux"):
                if show_all or not services:
                    out, _ = await _run(["systemctl","list-units","--type=service","--state=failed","--no-pager","--plain","--no-legend"])
                    failed = [l.split()[0] for l in out.splitlines() if l.strip() and ".service" in l]
                    if show_all:
                        out2, _ = await _run(["systemctl","list-units","--type=service","--state=active","--no-pager","--plain","--no-legend"])
                        active = [l.split()[0] for l in out2.splitlines() if l.strip() and ".service" in l]
                        return SkillResult.ok(self.manifest.name, call_id, {"active_count":len(active),"failed_count":len(failed),"active":active[:50],"failed":failed})
                    return SkillResult.ok(self.manifest.name, call_id, {"failed_count":len(failed),"failed":failed})
                for svc in services:
                    out, rc = await _run(["systemctl","is-active",svc])
                    status = out.strip()
                    out2, _ = await _run(["systemctl","show",svc,"--property=ActiveState,SubState,MainPID,ExecMainStartTimestamp","--no-pager"])
                    props = {l.split("=",1)[0]:l.split("=",1)[1] for l in out2.splitlines() if "=" in l}
                    results.append({"service":svc,"active":status=="active","status":status,
                                    "pid":props.get("MainPID",""),"started":props.get("ExecMainStartTimestamp","")})
            elif sys.platform == "darwin":
                for svc in services:
                    out, rc = await _run(["launchctl","list",svc])
                    running = rc == 0 and "PID" in out
                    results.append({"service":svc,"active":running,"status":"active" if running else "inactive","raw":out[:200]})
            else:
                return SkillResult.fail(self.manifest.name, call_id, f"Unsupported platform: {sys.platform}")

            duration_ms = (time.monotonic()-t_start)*1000
            return SkillResult.ok(self.manifest.name, call_id,
                {"service_count":len(results),"services":results}, duration_ms=duration_ms)
        except BaseException as e:
            return SkillResult.fail(self.manifest.name, call_id, f"{type(e).__name__}: {e}", type(e).__name__,
                                    duration_ms=(time.monotonic()-t_start)*1000)

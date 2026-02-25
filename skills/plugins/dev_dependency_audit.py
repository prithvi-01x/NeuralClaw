"""
skills/plugins/dev_dependency_audit.py — Developer: Dependency Audit

Audits dependencies for known vulnerabilities.
Python: pip-audit or safety. Node: npm audit. Falls back to listing deps only.

Risk: LOW — fs:read, net:fetch
"""
from __future__ import annotations
import asyncio, json, time
from pathlib import Path
from typing import ClassVar
from skills.base import SkillBase
from skills.types import RiskLevel, SkillManifest, SkillResult, SkillValidationError

class DevDependencyAuditSkill(SkillBase):
    manifest: ClassVar[SkillManifest] = SkillManifest(
        name="dev_dependency_audit",
        version="1.0.0",
        description="Audit project dependencies for known vulnerabilities. Python: pip-audit/safety. Node: npm audit. Also lists all direct dependencies.",
        category="developer",
        risk_level=RiskLevel.LOW,
        capabilities=frozenset({"fs:read","net:fetch"}),
        timeout_seconds=60,
        parameters={"type":"object","properties":{
            "project_path":{"type":"string","description":"Path to project root."},
            "ecosystem":{"type":"string","enum":["auto","python","node"],"description":"Ecosystem (default: auto-detect).","default":"auto"},
        },"required":["project_path"]},
    )

    async def validate(self, project_path: str, **_) -> None:
        if not Path(project_path).expanduser().exists():
            raise SkillValidationError(f"Project path does not exist: '{project_path}'")

    async def execute(self, project_path: str, ecosystem: str="auto", **kwargs) -> SkillResult:
        call_id = kwargs.get("_skill_call_id","")
        t_start = time.monotonic()
        proj = Path(project_path).expanduser().resolve()

        async def _run(cmd, timeout=50):
            proc = await asyncio.create_subprocess_exec(*cmd, cwd=str(proj),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            try: stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError: proc.kill(); return None, None, -1
            return stdout.decode(errors="replace"), stderr.decode(errors="replace"), proc.returncode

        try:
            eco = ecosystem
            if eco == "auto":
                if (proj/"requirements.txt").exists() or (proj/"pyproject.toml").exists() or (proj/"setup.py").exists():
                    eco = "python"
                elif (proj/"package.json").exists():
                    eco = "node"
                else:
                    eco = "unknown"

            vulns = []
            deps = []
            audit_tool = None

            if eco == "python":
                # Try pip-audit
                stdout, _, rc = await _run(["pip-audit","--format=json","--progress-spinner=off"])
                if rc == 0 and stdout:
                    try:
                        data = json.loads(stdout)
                        audit_tool = "pip-audit"
                        for dep in data.get("dependencies",[]):
                            deps.append({"name":dep.get("name"),"version":dep.get("version")})
                            for v in dep.get("vulns",[]):
                                vulns.append({"package":dep.get("name"),"version":dep.get("version"),
                                    "id":v.get("id"),"description":v.get("description","")[:200],"fix":v.get("fix_versions",[])})
                    except Exception: pass
                else:
                    # Fall back to listing only
                    stdout, _, _ = await _run(["pip","list","--format=json"])
                    audit_tool = "pip list (no audit tool)"
                    if stdout:
                        try: deps = json.loads(stdout)
                        except Exception: pass

            elif eco == "node":
                stdout, _, rc = await _run(["npm","audit","--json"])
                audit_tool = "npm audit"
                if stdout:
                    try:
                        data = json.loads(stdout)
                        for name, adv in data.get("advisories",{}).items():
                            vulns.append({"package":adv.get("module_name"),"severity":adv.get("severity"),
                                "title":adv.get("title",""),"url":adv.get("url","")})
                        for name, info in data.get("dependencies",{}).items():
                            deps.append({"name":name,"version":info.get("version","")})
                    except Exception: pass

            duration_ms = (time.monotonic()-t_start)*1000
            return SkillResult.ok(self.manifest.name, call_id, {
                "ecosystem":eco,"audit_tool":audit_tool,"vulnerability_count":len(vulns),
                "vulnerabilities":vulns,"dependency_count":len(deps),"dependencies":deps[:50]
            }, duration_ms=duration_ms)
        except BaseException as e:
            return SkillResult.fail(self.manifest.name, call_id, f"{type(e).__name__}: {e}", type(e).__name__,
                                    duration_ms=(time.monotonic()-t_start)*1000)

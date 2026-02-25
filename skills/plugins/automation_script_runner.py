"""
skills/plugins/automation_script_runner.py — Automation: Script Runner

Executes a shell script or command, but ONLY if the script path is within allowed_paths.
Path validation runs in validate() before execute() is ever called.
requires_confirmation=True — always prompts.

Risk: HIGH — shell:run
"""
from __future__ import annotations
import asyncio, os, time
from pathlib import Path
from typing import ClassVar
from skills.base import SkillBase
from skills.types import RiskLevel, SkillManifest, SkillResult, SkillValidationError

# Paths that are always blocked regardless of allowed_paths
_ALWAYS_BLOCKED = ["/etc","/sys","/proc","/boot","/dev","/root","~/.ssh","~/.gnupg"]

class AutomationScriptRunnerSkill(SkillBase):
    manifest: ClassVar[SkillManifest] = SkillManifest(
        name="automation_script_runner",
        version="1.0.0",
        description="Execute a shell script file. Script path MUST be within allowed_paths. Requires confirmation. Never executes scripts in /etc, /sys, /proc, or system directories.",
        category="automation",
        risk_level=RiskLevel.HIGH,
        capabilities=frozenset({"shell:run"}),
        requires_confirmation=True,
        timeout_seconds=120,
        parameters={"type":"object","properties":{
            "script_path":{"type":"string","description":"Absolute path to the script file to execute."},
            "args":{"type":"array","items":{"type":"string"},"description":"Arguments to pass to the script.","default":[]},
            "timeout":{"type":"integer","description":"Max seconds to wait (default 60, max 110).","default":60},
            "allowed_paths":{"type":"array","items":{"type":"string"},"description":"Paths the script is allowed to be in. Defaults to home directory.","default":[]},
            "env_vars":{"type":"object","description":"Extra environment variables to set.","default":{}},
        },"required":["script_path"]},
    )

    async def validate(self, script_path: str, allowed_paths: list|None=None, **_) -> None:
        p = Path(script_path).expanduser().resolve()
        if not p.exists(): raise SkillValidationError(f"Script does not exist: '{script_path}'")
        if not p.is_file(): raise SkillValidationError(f"'{script_path}' is not a file.")
        # Block system directories
        for blocked in _ALWAYS_BLOCKED:
            blocked_p = Path(blocked).expanduser().resolve()
            try:
                p.relative_to(blocked_p)
                raise SkillValidationError(f"Script path '{script_path}' is in a blocked system directory: {blocked}")
            except ValueError: pass
        # Check allowed_paths
        if allowed_paths:
            allowed = [Path(a).expanduser().resolve() for a in allowed_paths]
            if not any(_is_subpath(p, a) for a in allowed):
                raise SkillValidationError(
                    f"Script '{p}' is not within allowed_paths: {[str(a) for a in allowed]}")
        else:
            home = Path.home().resolve()
            if not _is_subpath(p, home):
                raise SkillValidationError(f"Script '{p}' must be within your home directory when no allowed_paths specified.")

    async def execute(self, script_path: str, args: list|None=None, timeout: int=60,
                      allowed_paths: list|None=None, env_vars: dict|None=None, **kwargs) -> SkillResult:
        call_id = kwargs.get("_skill_call_id","")
        t_start = time.monotonic()
        timeout = min(int(timeout), 110)
        p = Path(script_path).expanduser().resolve()
        args = args or []
        env = {**os.environ, **(env_vars or {})}

        try:
            cmd = [str(p)] + [str(a) for a in args]
            proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, env=env, cwd=str(p.parent))
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                return SkillResult.fail(self.manifest.name, call_id, f"Script timed out after {timeout}s", "TimeoutError",
                                        duration_ms=(time.monotonic()-t_start)*1000)

            out = stdout.decode(errors="replace")
            err = stderr.decode(errors="replace")
            duration_ms = (time.monotonic()-t_start)*1000
            return SkillResult.ok(self.manifest.name, call_id, {
                "script":str(p),"return_code":proc.returncode,"stdout":out[-5000:],
                "stderr":err[-2000:],"success":proc.returncode==0
            }, duration_ms=duration_ms)
        except BaseException as e:
            return SkillResult.fail(self.manifest.name, call_id, f"{type(e).__name__}: {e}", type(e).__name__,
                                    duration_ms=(time.monotonic()-t_start)*1000)

def _is_subpath(child: Path, parent: Path) -> bool:
    try: child.relative_to(parent); return True
    except ValueError: return False

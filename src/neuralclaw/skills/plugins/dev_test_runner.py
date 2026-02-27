"""
skills/plugins/dev_test_runner.py — Developer: Test Runner

Runs pytest, unittest, or jest and returns structured results.

Risk: MED — fs:read, shell:run
"""
from __future__ import annotations
import asyncio, json, re, time
from pathlib import Path
from typing import ClassVar
from neuralclaw.skills.base import SkillBase
from neuralclaw.skills.types import RiskLevel, SkillManifest, SkillResult, SkillValidationError

class DevTestRunnerSkill(SkillBase):
    manifest: ClassVar[SkillManifest] = SkillManifest(
        name="dev_test_runner",
        version="1.0.0",
        description="Run the test suite for a project using pytest, unittest, or jest. Returns pass/fail counts and failed test details.",
        category="developer",
        risk_level=RiskLevel.MEDIUM,
        capabilities=frozenset({"fs:read","shell:run"}),
        timeout_seconds=120,
        parameters={"type":"object","properties":{
            "project_path":{"type":"string","description":"Path to project root."},
            "runner":{"type":"string","enum":["auto","pytest","unittest","jest"],"description":"Test runner (default: auto).","default":"auto"},
            "test_path":{"type":"string","description":"Specific test file or directory (optional).","default":""},
            "extra_args":{"type":"string","description":"Extra CLI args to pass to the runner.","default":""},
            "timeout":{"type":"integer","description":"Max seconds to wait for tests (default 90).","default":90},
        },"required":["project_path"]},
    )

    async def validate(self, project_path: str, **_) -> None:
        if not Path(project_path).expanduser().exists():
            raise SkillValidationError(f"Project path does not exist: '{project_path}'")

    async def execute(self, project_path: str, runner: str="auto", test_path: str="",
                      extra_args: str="", timeout: int=90, **kwargs) -> SkillResult:
        call_id = kwargs.get("_skill_call_id","")
        t_start = time.monotonic()
        timeout = min(int(timeout), 110)
        proj = Path(project_path).expanduser().resolve()

        async def _try_runner(name: str) -> bool:
            try:
                p = await asyncio.create_subprocess_exec(name,"--version" if name!="python" else "-c","import pytest",
                    stdout=asyncio.subprocess.DEVNULL,stderr=asyncio.subprocess.DEVNULL)
                await p.wait()
                return p.returncode == 0
            except (OSError, FileNotFoundError): return False

        try:
            # Determine runner
            chosen = runner
            if chosen == "auto":
                if await _try_runner("pytest"): chosen = "pytest"
                elif (proj/"package.json").exists() and await _try_runner("jest"): chosen = "jest"
                else: chosen = "unittest"

            if chosen == "pytest":
                cmd = ["pytest","--tb=short","-q","--no-header"]
                try:
                    import pytest; cmd.append("--json-report") if False else None
                except: pass
                if test_path: cmd.append(test_path)
                if extra_args: cmd.extend(extra_args.split())
            elif chosen == "unittest":
                cmd = ["python","-m","unittest","discover","-v"]
                if test_path: cmd.extend(["-s",test_path])
                if extra_args: cmd.extend(extra_args.split())
            elif chosen == "jest":
                cmd = ["jest","--no-coverage","--passWithNoTests"]
                if test_path: cmd.append(test_path)
                if extra_args: cmd.extend(extra_args.split())
            else:
                return SkillResult.fail(self.manifest.name, call_id, f"Unknown runner: {chosen}")

            proc = await asyncio.create_subprocess_exec(*cmd, cwd=str(proj),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                return SkillResult.fail(self.manifest.name, call_id, f"Test run timed out after {timeout}s", "TimeoutError")

            combined = (stdout+stderr).decode(errors="replace")
            passed = failed = errors = 0

            # Parse pytest summary line
            m = re.search(r"(\d+) passed", combined)
            if m: passed = int(m.group(1))
            m = re.search(r"(\d+) failed", combined)
            if m: failed = int(m.group(1))
            m = re.search(r"(\d+) error", combined)
            if m: errors = int(m.group(1))

            # Extract FAILED lines
            failed_tests = [l.strip() for l in combined.splitlines() if l.strip().startswith("FAILED")]

            duration_ms = (time.monotonic()-t_start)*1000
            return SkillResult.ok(self.manifest.name, call_id, {
                "runner":chosen,"return_code":proc.returncode,"passed":passed,"failed":failed,"errors":errors,
                "failed_tests":failed_tests,"clean":proc.returncode==0,"output":combined[-3000:]
            }, duration_ms=duration_ms)
        except BaseException as e:
            return SkillResult.fail(self.manifest.name, call_id, f"{type(e).__name__}: {e}", type(e).__name__,
                                    duration_ms=(time.monotonic()-t_start)*1000)

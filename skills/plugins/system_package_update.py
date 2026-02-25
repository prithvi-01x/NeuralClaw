"""
skills/plugins/system_package_update.py — System: Package Update

Lists available updates or runs package upgrade. CRITICAL risk — always confirms.
Supports apt, dnf/yum, brew, pacman.

Risk: CRITICAL — requires_confirmation=True
"""
from __future__ import annotations
import asyncio, sys, time
from typing import ClassVar
from skills.base import SkillBase
from skills.types import RiskLevel, SkillManifest, SkillResult, SkillValidationError

class SystemPackageUpdateSkill(SkillBase):
    manifest: ClassVar[SkillManifest] = SkillManifest(
        name="system_package_update",
        version="1.0.0",
        description="List available system package updates or perform an upgrade. CRITICAL risk — always requires confirmation. Supports apt, dnf, yum, brew, pacman.",
        category="system",
        risk_level=RiskLevel.CRITICAL,
        capabilities=frozenset({"shell:run"}),
        requires_confirmation=True,
        timeout_seconds=180,
        parameters={"type":"object","properties":{
            "action":{"type":"string","enum":["list","upgrade"],"description":"'list' shows available updates. 'upgrade' installs them (requires confirmation)."},
            "packages":{"type":"array","items":{"type":"string"},"description":"Specific packages to update (empty = all).","default":[]},
            "dry_run":{"type":"boolean","description":"Simulate upgrade without installing (default false).","default":False},
        },"required":["action"]},
    )

    async def execute(self, action: str, packages: list|None=None, dry_run: bool=False, **kwargs) -> SkillResult:
        call_id = kwargs.get("_skill_call_id","")
        t_start = time.monotonic()
        packages = packages or []

        async def _run(cmd, timeout=150):
            proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            try: out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError: proc.kill(); return "", "timeout", -1
            return out.decode(errors="replace"), err.decode(errors="replace"), proc.returncode

        async def _detect_pm():
            for pm in ["apt","dnf","yum","brew","pacman"]:
                p = await asyncio.create_subprocess_exec("which",pm,stdout=asyncio.subprocess.DEVNULL,stderr=asyncio.subprocess.DEVNULL)
                await p.wait()
                if p.returncode == 0: return pm
            return None

        try:
            pm = await _detect_pm()
            if not pm: return SkillResult.fail(self.manifest.name, call_id, "No supported package manager found (apt/dnf/yum/brew/pacman).")

            if action == "list":
                if pm == "apt":
                    await _run(["apt","update","-qq"])
                    out, _, _ = await _run(["apt","list","--upgradable","--quiet=2"])
                elif pm in ("dnf","yum"):
                    out, _, _ = await _run([pm,"check-update"])
                elif pm == "brew":
                    out, _, _ = await _run(["brew","outdated"])
                elif pm == "pacman":
                    out, _, _ = await _run(["pacman","-Qu"])
                else: out = ""
                lines = [l for l in out.splitlines() if l.strip()]
                return SkillResult.ok(self.manifest.name, call_id,
                    {"package_manager":pm,"available_updates":len(lines),"updates":lines})

            elif action == "upgrade":
                if pm == "apt":
                    cmd = ["apt","upgrade","-y"] + (["--dry-run"] if dry_run else []) + packages
                elif pm in ("dnf","yum"):
                    cmd = [pm,"upgrade","-y"] + (["--assumeno"] if dry_run else []) + packages
                elif pm == "brew":
                    cmd = ["brew","upgrade"] + packages
                elif pm == "pacman":
                    cmd = ["pacman","-Syu","--noconfirm"] + packages
                else: return SkillResult.fail(self.manifest.name, call_id, f"Upgrade not supported for {pm}")
                out, err, rc = await _run(cmd, timeout=150)
                duration_ms = (time.monotonic()-t_start)*1000
                return SkillResult.ok(self.manifest.name, call_id,
                    {"package_manager":pm,"return_code":rc,"dry_run":dry_run,"output":out[-3000:],"error":err[-1000:]}, duration_ms=duration_ms)

            return SkillResult.fail(self.manifest.name, call_id, f"Unknown action: {action}")
        except BaseException as e:
            return SkillResult.fail(self.manifest.name, call_id, f"{type(e).__name__}: {e}", type(e).__name__,
                                    duration_ms=(time.monotonic()-t_start)*1000)

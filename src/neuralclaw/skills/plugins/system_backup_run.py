"""
skills/plugins/system_backup_run.py — System: Backup Run

Creates a compressed archive of source paths to a destination.
CRITICAL risk — requires confirmation. Uses tar + gzip/bzip2/xz.

Risk: CRITICAL — requires_confirmation=True
"""
from __future__ import annotations
import asyncio, time
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar
from neuralclaw.skills.base import SkillBase
from neuralclaw.skills.types import RiskLevel, SkillManifest, SkillResult, SkillValidationError

_BLOCKED_SOURCES = ["/etc","/sys","/proc","/dev","/boot"]

class SystemBackupRunSkill(SkillBase):
    manifest: ClassVar[SkillManifest] = SkillManifest(
        name="system_backup_run",
        version="1.0.0",
        description="Create a compressed tar archive of specified paths. CRITICAL risk — always requires confirmation. Supports gzip, bzip2, xz compression.",
        category="system",
        risk_level=RiskLevel.CRITICAL,
        capabilities=frozenset({"fs:read","fs:write"}),
        requires_confirmation=True,
        timeout_seconds=300,
        parameters={"type":"object","properties":{
            "source_paths":{"type":"array","items":{"type":"string"},"description":"List of files/directories to back up."},
            "destination":{"type":"string","description":"Destination directory for the archive."},
            "compression":{"type":"string","enum":["gz","bz2","xz","none"],"description":"Compression type (default gz).","default":"gz"},
            "archive_name":{"type":"string","description":"Archive filename prefix (default: 'backup').","default":"backup"},
            "exclude_patterns":{"type":"array","items":{"type":"string"},"description":"Glob patterns to exclude (e.g. ['*.pyc','__pycache__']).","default":[]},
        },"required":["source_paths","destination"]},
    )

    async def validate(self, source_paths: list, destination: str, **_) -> None:
        if not source_paths: raise SkillValidationError("source_paths must be non-empty.")
        for sp in source_paths:
            p = Path(sp).expanduser().resolve()
            if not p.exists(): raise SkillValidationError(f"Source path does not exist: '{sp}'")
            for blocked in _BLOCKED_SOURCES:
                try: p.relative_to(Path(blocked).resolve()); raise SkillValidationError(f"'{sp}' is in a blocked system directory: {blocked}")
                except ValueError: pass
        dest = Path(destination).expanduser()
        if dest.is_file(): raise SkillValidationError(f"destination '{destination}' is a file, must be a directory.")

    async def execute(self, source_paths: list, destination: str, compression: str="gz",
                      archive_name: str="backup", exclude_patterns: list|None=None, **kwargs) -> SkillResult:
        call_id = kwargs.get("_skill_call_id","")
        t_start = time.monotonic()
        exclude_patterns = exclude_patterns or []
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        ext = {"gz":"tar.gz","bz2":"tar.bz2","xz":"tar.xz","none":"tar"}.get(compression,"tar.gz")
        dest_dir = Path(destination).expanduser().resolve()

        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            archive_path = dest_dir / f"{archive_name}_{ts}.{ext}"

            compress_flag = {"gz":"-z","bz2":"-j","xz":"-J","none":""}.get(compression,"")
            cmd = ["tar","-c"]
            if compress_flag: cmd.append(compress_flag)
            for pat in exclude_patterns: cmd.extend(["--exclude",pat])
            cmd.extend(["-f",str(archive_path)])
            for sp in source_paths: cmd.append(str(Path(sp).expanduser().resolve()))

            proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            try: _, err = await asyncio.wait_for(proc.communicate(), timeout=280)
            except asyncio.TimeoutError: proc.kill(); return SkillResult.fail(self.manifest.name, call_id, "Backup timed out after 280s", "TimeoutError")

            if proc.returncode != 0:
                return SkillResult.fail(self.manifest.name, call_id, f"tar failed: {err.decode(errors='replace')[:500]}", "BackupError")

            size = archive_path.stat().st_size
            duration_ms = (time.monotonic()-t_start)*1000
            return SkillResult.ok(self.manifest.name, call_id, {
                "archive":str(archive_path),"size_bytes":size,"size_human":_hr(size),
                "sources":source_paths,"compression":compression,"duration_ms":round(duration_ms)
            }, duration_ms=duration_ms)
        except BaseException as e:
            return SkillResult.fail(self.manifest.name, call_id, f"{type(e).__name__}: {e}", type(e).__name__,
                                    duration_ms=(time.monotonic()-t_start)*1000)

def _hr(n):
    for u in ["B","KB","MB","GB"]:
        if n<1024: return f"{n:.1f} {u}"
        n/=1024
    return f"{n:.1f} TB"

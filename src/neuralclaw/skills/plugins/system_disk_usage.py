"""
skills/plugins/system_disk_usage.py — System: Disk Usage

Reports disk usage for mounted filesystems and optionally for specific paths.
Uses Python's shutil and os — no external dependencies.

Risk: LOW — fs:read capability required.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time
from pathlib import Path
from typing import ClassVar

from neuralclaw.skills.base import SkillBase
from neuralclaw.skills.types import RiskLevel, SkillManifest, SkillResult, SkillValidationError

_WARN_THRESHOLD = 80   # % used before marking as warning
_CRIT_THRESHOLD = 90   # % used before marking as critical


class SystemDiskUsageSkill(SkillBase):
    manifest: ClassVar[SkillManifest] = SkillManifest(
        name="system_disk_usage",
        version="1.0.0",
        description=(
            "Report disk usage for all mounted filesystems and optionally drill into "
            "specific directories. Flags filesystems above 80% (warning) and 90% (critical) usage."
        ),
        category="system",
        risk_level=RiskLevel.LOW,
        capabilities=frozenset({"fs:read"}),
        requires_confirmation=False,
        timeout_seconds=30,
        parameters={
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of specific paths to report usage for.",
                    "default": [],
                },
                "show_all_mounts": {
                    "type": "boolean",
                    "description": "Include all mounted filesystems (default true).",
                    "default": True,
                },
                "human_readable": {
                    "type": "boolean",
                    "description": "Format sizes in human-readable format GB/MB (default true).",
                    "default": True,
                },
            },
            "required": [],
        },
    )

    async def execute(
        self,
        paths: list[str] | None = None,
        show_all_mounts: bool = True,
        human_readable: bool = True,
        **kwargs,
    ) -> SkillResult:
        call_id = kwargs.get("_skill_call_id", "")
        t_start = time.monotonic()

        def _hr(n: int) -> str:
            if not human_readable:
                return str(n)
            for unit in ["B", "KB", "MB", "GB", "TB"]:
                if n < 1024:
                    return f"{n:.1f} {unit}"
                n /= 1024
            return f"{n:.1f} PB"

        try:
            result: dict = {}

            # ── All mounted filesystems ──────────────────────────────────────
            if show_all_mounts:
                mounts = []
                seen_devices: set[str] = set()

                def _get_mounts() -> list[dict]:
                    entries = []
                    try:
                        # Read /proc/mounts on Linux
                        with open("/proc/mounts") as f:
                            for line in f:
                                parts = line.split()
                                if len(parts) < 2:
                                    continue
                                device, mountpoint = parts[0], parts[1]
                                # Skip pseudo-filesystems
                                if device in ("proc", "sysfs", "devtmpfs", "tmpfs", "devpts",
                                              "cgroup", "cgroup2", "pstore", "bpf", "tracefs",
                                              "debugfs", "securityfs", "fusectl", "mqueue",
                                              "hugetlbfs", "none", "udev", "run"):
                                    continue
                                if device.startswith(("proc", "sys", "dev", "run", "tmpfs")):
                                    continue
                                if device in seen_devices:
                                    continue
                                seen_devices.add(device)
                                try:
                                    usage = shutil.disk_usage(mountpoint)
                                    pct = (usage.used / usage.total * 100) if usage.total > 0 else 0
                                    status = (
                                        "critical" if pct >= _CRIT_THRESHOLD
                                        else "warning" if pct >= _WARN_THRESHOLD
                                        else "ok"
                                    )
                                    entries.append({
                                        "mountpoint": mountpoint,
                                        "device": device,
                                        "total": _hr(usage.total),
                                        "used": _hr(usage.used),
                                        "free": _hr(usage.free),
                                        "used_pct": round(pct, 1),
                                        "status": status,
                                    })
                                except (OSError, PermissionError):
                                    pass
                    except (OSError, FileNotFoundError):
                        # macOS fallback
                        try:
                            usage = shutil.disk_usage("/")
                            pct = usage.used / usage.total * 100
                            entries.append({
                                "mountpoint": "/",
                                "device": "root",
                                "total": _hr(usage.total),
                                "used": _hr(usage.used),
                                "free": _hr(usage.free),
                                "used_pct": round(pct, 1),
                                "status": "critical" if pct >= _CRIT_THRESHOLD else "warning" if pct >= _WARN_THRESHOLD else "ok",
                            })
                        except OSError:
                            pass
                    return entries

                loop = asyncio.get_event_loop()
                mounts = await loop.run_in_executor(None, _get_mounts)
                result["filesystems"] = sorted(mounts, key=lambda x: x["mountpoint"])
                result["alerts"] = [m for m in mounts if m["status"] != "ok"]

            # ── Specific paths ───────────────────────────────────────────────
            if paths:
                path_results = []
                for p_str in paths:
                    p = Path(p_str).expanduser().resolve()
                    if not p.exists():
                        path_results.append({"path": p_str, "error": "path does not exist"})
                        continue
                    try:
                        usage = shutil.disk_usage(str(p))
                        pct = usage.used / usage.total * 100 if usage.total > 0 else 0

                        # du-style size of the directory itself
                        dir_bytes = 0
                        if p.is_dir():
                            def _du(directory: Path) -> int:
                                total = 0
                                try:
                                    for entry in os.scandir(directory):
                                        try:
                                            if entry.is_file(follow_symlinks=False):
                                                total += entry.stat(follow_symlinks=False).st_size
                                            elif entry.is_dir(follow_symlinks=False):
                                                total += _du(Path(entry.path))
                                        except (OSError, PermissionError):
                                            pass
                                except (OSError, PermissionError):
                                    pass
                                return total
                            loop = asyncio.get_event_loop()
                            dir_bytes = await loop.run_in_executor(None, _du, p)

                        path_results.append({
                            "path": str(p),
                            "filesystem_total": _hr(usage.total),
                            "filesystem_used": _hr(usage.used),
                            "filesystem_free": _hr(usage.free),
                            "filesystem_used_pct": round(pct, 1),
                            "directory_size": _hr(dir_bytes) if p.is_dir() else _hr(p.stat().st_size),
                        })
                    except (OSError, PermissionError) as e:
                        path_results.append({"path": p_str, "error": str(e)})
                result["paths"] = path_results

            duration_ms = (time.monotonic() - t_start) * 1000
            return SkillResult.ok(
                skill_name=self.manifest.name,
                skill_call_id=call_id,
                output=result,
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
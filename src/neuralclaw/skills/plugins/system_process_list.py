"""
skills/plugins/system_process_list.py — System: Process List

Returns running processes with PID, name, CPU%, memory%, and status.
Uses psutil (preferred) or /proc fallback on Linux.

Risk: LOW — fs:read
"""
from __future__ import annotations
import asyncio, time
from typing import ClassVar
from neuralclaw.skills.base import SkillBase
from neuralclaw.skills.types import RiskLevel, SkillManifest, SkillResult

class SystemProcessListSkill(SkillBase):
    manifest: ClassVar[SkillManifest] = SkillManifest(
        name="system_process_list",
        version="1.0.0",
        description="List running processes with PID, name, CPU%, memory%, user, and status. Filter by name or user. Requires psutil (pip install psutil).",
        category="system",
        risk_level=RiskLevel.LOW,
        capabilities=frozenset({"fs:read"}),
        timeout_seconds=15,
        parameters={"type":"object","properties":{
            "filter_name":{"type":"string","description":"Filter processes by name (partial match, case-insensitive).","default":""},
            "filter_user":{"type":"string","description":"Filter by username.","default":""},
            "sort_by":{"type":"string","enum":["cpu","memory","pid","name"],"description":"Sort by this field (default: cpu).","default":"cpu"},
            "limit":{"type":"integer","description":"Max processes to return (default 30).","default":30},
            "include_threads":{"type":"boolean","description":"Include thread count (default false).","default":False},
        },"required":[]},
    )

    async def execute(self, filter_name: str="", filter_user: str="", sort_by: str="cpu",
                      limit: int=30, include_threads: bool=False, **kwargs) -> SkillResult:
        call_id = kwargs.get("_skill_call_id","")
        t_start = time.monotonic()
        limit = min(int(limit), 200)

        def _list_processes():
            try:
                import psutil
                procs = []
                for p in psutil.process_iter(["pid","name","username","status","cpu_percent","memory_percent","num_threads"]):
                    try:
                        info = p.info
                        if filter_name and filter_name.lower() not in (info.get("name") or "").lower(): continue
                        if filter_user and filter_user.lower() not in (info.get("username") or "").lower(): continue
                        entry = {"pid":info["pid"],"name":info.get("name",""),"user":info.get("username",""),
                                 "status":info.get("status",""),"cpu_pct":round(info.get("cpu_percent") or 0,2),
                                 "mem_pct":round(info.get("memory_percent") or 0,2)}
                        if include_threads: entry["threads"] = info.get("num_threads",0)
                        procs.append(entry)
                    except (psutil.NoSuchProcess, psutil.AccessDenied): pass
                key_map = {"cpu":"cpu_pct","memory":"mem_pct","pid":"pid","name":"name"}
                procs.sort(key=lambda x: x.get(key_map.get(sort_by,"cpu_pct"),0), reverse=sort_by in ("cpu","memory"))
                return procs[:limit], "psutil"
            except ImportError:
                # /proc fallback (Linux only)
                import os, re
                procs = []
                try:
                    for pid_str in os.listdir("/proc"):
                        if not pid_str.isdigit(): continue
                        try:
                            name = open(f"/proc/{pid_str}/comm").read().strip()
                            if filter_name and filter_name.lower() not in name.lower(): continue
                            procs.append({"pid":int(pid_str),"name":name,"user":"","status":"","cpu_pct":0,"mem_pct":0})
                        except Exception: pass
                except Exception: pass
                return procs[:limit], "proc_fallback"

        try:
            loop = asyncio.get_event_loop()
            procs, source = await loop.run_in_executor(None, _list_processes)
            duration_ms = (time.monotonic()-t_start)*1000
            return SkillResult.ok(self.manifest.name, call_id,
                {"process_count":len(procs),"source":source,"processes":procs}, duration_ms=duration_ms)
        except BaseException as e:
            return SkillResult.fail(self.manifest.name, call_id, f"{type(e).__name__}: {e}", type(e).__name__,
                                    duration_ms=(time.monotonic()-t_start)*1000)

"""
skills/plugins/automation_file_watcher.py — Automation: File Watcher

Takes a snapshot of a directory (files, sizes, mtimes) and optionally
compares against a previous snapshot to detect changes. No background daemon —
call it twice and compare, or use as a pre/post check.

Risk: MED — fs:read
"""
from __future__ import annotations
import asyncio, hashlib, json, time
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar
from neuralclaw.skills.base import SkillBase
from neuralclaw.skills.types import RiskLevel, SkillManifest, SkillResult, SkillValidationError

_SNAPSHOT_DIR = Path("~/neuralclaw/snapshots").expanduser()

class AutomationFileWatcherSkill(SkillBase):
    manifest: ClassVar[SkillManifest] = SkillManifest(
        name="automation_file_watcher",
        version="1.0.0",
        description="Snapshot a directory's file tree (names, sizes, mtimes). Compare two snapshots to detect added, removed, or modified files. Useful as a pre/post change detector.",
        category="automation",
        risk_level=RiskLevel.MEDIUM,
        capabilities=frozenset({"fs:read"}),
        timeout_seconds=30,
        parameters={"type":"object","properties":{
            "action":{"type":"string","enum":["snapshot","compare","list_snapshots"],"description":"Action to perform."},
            "watch_path":{"type":"string","description":"Directory to snapshot (required for snapshot/compare).","default":""},
            "snapshot_name":{"type":"string","description":"Name to save/load snapshot as.","default":""},
            "compare_with":{"type":"string","description":"Name of older snapshot to compare against (for compare action).","default":""},
            "max_files":{"type":"integer","description":"Max files to include in snapshot (default 1000).","default":1000},
        },"required":["action"]},
    )

    async def validate(self, action: str, watch_path: str="", snapshot_name: str="", **_) -> None:
        if action in ("snapshot","compare") and not watch_path:
            raise SkillValidationError(f"watch_path is required for action='{action}'.")
        if action in ("snapshot","compare") and not snapshot_name:
            raise SkillValidationError(f"snapshot_name is required for action='{action}'.")
        if action in ("snapshot","compare") and watch_path and not Path(watch_path).expanduser().exists():
            raise SkillValidationError(f"watch_path does not exist: '{watch_path}'")

    async def execute(self, action: str, watch_path: str="", snapshot_name: str="",
                      compare_with: str="", max_files: int=1000, **kwargs) -> SkillResult:
        call_id = kwargs.get("_skill_call_id","")
        t_start = time.monotonic()
        max_files = min(int(max_files), 5000)

        def _make_snapshot(path: Path) -> dict:
            files = {}
            for f in sorted(path.rglob("*")):
                if not f.is_file(): continue
                try:
                    st = f.stat()
                    files[str(f.relative_to(path))] = {"size":st.st_size,"mtime":st.st_mtime}
                except OSError: pass
                if len(files) >= max_files: break
            return {"path":str(path),"file_count":len(files),"taken_at":datetime.now(tz=timezone.utc).isoformat(),"files":files}

        def _load_snapshot(name: str) -> dict|None:
            p = _SNAPSHOT_DIR / f"{name}.json"
            if not p.exists(): return None
            try: return json.loads(p.read_text())
            except Exception: return None

        def _save_snapshot(name: str, snap: dict) -> None:
            _SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
            (_SNAPSHOT_DIR / f"{name}.json").write_text(json.dumps(snap, indent=2))

        try:
            loop = asyncio.get_event_loop()

            if action == "list_snapshots":
                def _list():
                    if not _SNAPSHOT_DIR.exists(): return []
                    return [{"name":f.stem,"size":f.stat().st_size} for f in sorted(_SNAPSHOT_DIR.glob("*.json"))]
                snaps = await loop.run_in_executor(None, _list)
                return SkillResult.ok(self.manifest.name, call_id, {"count":len(snaps),"snapshots":snaps})

            watch = Path(watch_path).expanduser().resolve()
            snap = await loop.run_in_executor(None, _make_snapshot, watch)
            await loop.run_in_executor(None, _save_snapshot, snapshot_name, snap)

            if action == "snapshot":
                duration_ms = (time.monotonic()-t_start)*1000
                return SkillResult.ok(self.manifest.name, call_id,
                    {"snapshot_name":snapshot_name,"file_count":snap["file_count"],"taken_at":snap["taken_at"]}, duration_ms=duration_ms)

            elif action == "compare":
                old = await loop.run_in_executor(None, _load_snapshot, compare_with)
                if not old: return SkillResult.fail(self.manifest.name, call_id, f"Snapshot '{compare_with}' not found.")
                old_files = old.get("files",{})
                new_files = snap.get("files",{})
                added   = [f for f in new_files if f not in old_files]
                removed = [f for f in old_files if f not in new_files]
                modified= [f for f in new_files if f in old_files and new_files[f]["mtime"] != old_files[f]["mtime"]]
                duration_ms = (time.monotonic()-t_start)*1000
                return SkillResult.ok(self.manifest.name, call_id, {
                    "added":added,"removed":removed,"modified":modified,
                    "added_count":len(added),"removed_count":len(removed),"modified_count":len(modified),
                    "has_changes": bool(added or removed or modified)
                }, duration_ms=duration_ms)

            return SkillResult.fail(self.manifest.name, call_id, f"Unknown action: {action}")
        except BaseException as e:
            return SkillResult.fail(self.manifest.name, call_id, f"{type(e).__name__}: {e}", type(e).__name__,
                                    duration_ms=(time.monotonic()-t_start)*1000)

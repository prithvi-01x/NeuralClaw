"""
skills/plugins/dev_code_search.py — Developer: Code Search

Searches for a pattern in source files using ripgrep (preferred) or stdlib grep fallback.

Risk: LOW — fs:read
"""
from __future__ import annotations
import asyncio, time
from pathlib import Path
from typing import ClassVar
from skills.base import SkillBase
from skills.types import RiskLevel, SkillManifest, SkillResult, SkillValidationError

class DevCodeSearchSkill(SkillBase):
    manifest: ClassVar[SkillManifest] = SkillManifest(
        name="dev_code_search",
        version="1.0.0",
        description="Search source files for a text pattern or regex. Uses ripgrep if available, falls back to Python glob+regex. Returns matching lines with file and line number.",
        category="developer",
        risk_level=RiskLevel.LOW,
        capabilities=frozenset({"fs:read"}),
        timeout_seconds=30,
        parameters={"type":"object","properties":{
            "path":{"type":"string","description":"Directory or file to search."},
            "pattern":{"type":"string","description":"Search pattern (string or regex)."},
            "file_glob":{"type":"string","description":"File glob filter (e.g. '*.py', '*.ts'). Default: all files.","default":""},
            "case_sensitive":{"type":"boolean","description":"Case-sensitive search (default false).","default":False},
            "max_results":{"type":"integer","description":"Max results to return (default 50).","default":50},
            "context_lines":{"type":"integer","description":"Lines of context around each match (default 0).","default":0},
        },"required":["path","pattern"]},
    )

    async def validate(self, path: str, pattern: str, **_) -> None:
        if not Path(path).expanduser().exists():
            raise SkillValidationError(f"Path does not exist: '{path}'")
        if not pattern.strip():
            raise SkillValidationError("pattern must be non-empty.")

    async def execute(self, path: str, pattern: str, file_glob: str="", case_sensitive: bool=False,
                      max_results: int=50, context_lines: int=0, **kwargs) -> SkillResult:
        call_id = kwargs.get("_skill_call_id","")
        t_start = time.monotonic()
        max_results = min(int(max_results), 200)
        search_path = Path(path).expanduser().resolve()

        try:
            # Try ripgrep first
            rg_available = False
            try:
                tp = await asyncio.create_subprocess_exec("rg","--version",stdout=asyncio.subprocess.DEVNULL,stderr=asyncio.subprocess.DEVNULL)
                await tp.wait()
                rg_available = tp.returncode == 0
            except (OSError, FileNotFoundError):
                pass

            matches = []

            if rg_available:
                cmd = ["rg","--line-number","--no-heading","--max-count","1"]
                if not case_sensitive: cmd.append("--ignore-case")
                if context_lines > 0: cmd.extend(["-C",str(context_lines)])
                if file_glob: cmd.extend(["--glob",file_glob])
                cmd.extend([pattern, str(search_path)])

                proc = await asyncio.create_subprocess_exec(*cmd,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=25)
                for line in stdout.decode(errors="replace").splitlines()[:max_results]:
                    parts = line.split(":",2)
                    if len(parts) >= 3:
                        matches.append({"file":parts[0],"line":parts[1],"content":parts[2]})
                    else:
                        matches.append({"file":"","line":"","content":line})
            else:
                import re
                flags = 0 if case_sensitive else re.IGNORECASE
                try: rx = re.compile(pattern, flags)
                except re.error: rx = re.compile(re.escape(pattern), flags)
                glob = file_glob or "**/*"
                def _search():
                    results = []
                    for fp in search_path.rglob(glob) if search_path.is_dir() else [search_path]:
                        if not fp.is_file(): continue
                        try:
                            for i, line in enumerate(fp.read_text(errors="replace").splitlines(), 1):
                                if rx.search(line):
                                    results.append({"file":str(fp),"line":str(i),"content":line.strip()})
                                    if len(results) >= max_results: return results
                        except Exception: pass
                    return results
                loop = asyncio.get_event_loop()
                matches = await loop.run_in_executor(None, _search)

            duration_ms = (time.monotonic()-t_start)*1000
            return SkillResult.ok(self.manifest.name, call_id,
                {"pattern":pattern,"path":str(search_path),"match_count":len(matches),"matches":matches}, duration_ms=duration_ms)
        except BaseException as e:
            return SkillResult.fail(self.manifest.name, call_id, f"{type(e).__name__}: {e}", type(e).__name__,
                                    duration_ms=(time.monotonic()-t_start)*1000)

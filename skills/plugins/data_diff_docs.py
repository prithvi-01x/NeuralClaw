"""
skills/plugins/data_diff_docs.py — Data: Diff Documents

Compares two text files line by line and returns a unified diff.
Works on any plain text: code, configs, markdown, CSV rows, etc.

Risk: LOW — fs:read
"""
from __future__ import annotations
import asyncio, difflib, time
from pathlib import Path
from typing import ClassVar
from skills.base import SkillBase
from skills.types import RiskLevel, SkillManifest, SkillResult, SkillValidationError

class DataDiffDocsSkill(SkillBase):
    manifest: ClassVar[SkillManifest] = SkillManifest(
        name="data_diff_docs",
        version="1.0.0",
        description="Compare two text files and return a unified diff. Works on code, configs, markdown, or any plain text format.",
        category="data",
        risk_level=RiskLevel.LOW,
        capabilities=frozenset({"fs:read"}),
        timeout_seconds=20,
        parameters={"type":"object","properties":{
            "file_a":{"type":"string","description":"Path to first file (original)."},
            "file_b":{"type":"string","description":"Path to second file (modified)."},
            "context_lines":{"type":"integer","description":"Lines of context around changes (default 3).","default":3},
            "max_chars":{"type":"integer","description":"Max characters of diff output (default 10000).","default":10000},
        },"required":["file_a","file_b"]},
    )

    async def validate(self, file_a: str, file_b: str, **_) -> None:
        for path, label in [(file_a,"file_a"),(file_b,"file_b")]:
            p = Path(path).expanduser()
            if not p.exists(): raise SkillValidationError(f"{label} does not exist: '{path}'")
            if not p.is_file(): raise SkillValidationError(f"{label} is not a file: '{path}'")

    async def execute(self, file_a: str, file_b: str, context_lines: int=3, max_chars: int=10000, **kwargs) -> SkillResult:
        call_id = kwargs.get("_skill_call_id","")
        t_start = time.monotonic()
        max_chars = min(int(max_chars), 50_000)

        def _diff():
            pa = Path(file_a).expanduser().resolve()
            pb = Path(file_b).expanduser().resolve()
            lines_a = pa.read_text(errors="replace").splitlines(keepends=True)
            lines_b = pb.read_text(errors="replace").splitlines(keepends=True)
            diff = list(difflib.unified_diff(lines_a, lines_b, fromfile=str(pa), tofile=str(pb), n=context_lines))
            diff_text = "".join(diff)
            added   = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
            removed = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))
            truncated = len(diff_text) > max_chars
            return {"file_a":str(pa),"file_b":str(pb),"lines_added":added,"lines_removed":removed,
                    "has_changes":bool(diff),"truncated":truncated,"diff":diff_text[:max_chars]}

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, _diff)
            duration_ms = (time.monotonic()-t_start)*1000
            return SkillResult.ok(self.manifest.name, call_id, result, duration_ms=duration_ms)
        except BaseException as e:
            return SkillResult.fail(self.manifest.name, call_id, f"{type(e).__name__}: {e}", type(e).__name__,
                                    duration_ms=(time.monotonic()-t_start)*1000)

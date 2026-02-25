"""
skills/plugins/dev_git_blame.py — Developer: Git Blame

Returns git blame output for a file showing who last modified each line.

Risk: LOW — fs:read, shell:run
"""
from __future__ import annotations
import asyncio, time
from pathlib import Path
from typing import ClassVar
from skills.base import SkillBase
from skills.types import RiskLevel, SkillManifest, SkillResult, SkillValidationError

class DevGitBlameSkill(SkillBase):
    manifest: ClassVar[SkillManifest] = SkillManifest(
        name="dev_git_blame",
        version="1.0.0",
        description="Show git blame for a file — who last modified each line, with commit hash, author, and date.",
        category="developer",
        risk_level=RiskLevel.LOW,
        capabilities=frozenset({"fs:read","shell:run"}),
        timeout_seconds=30,
        parameters={"type":"object","properties":{
            "repo_path":{"type":"string","description":"Path to git repository."},
            "file_path":{"type":"string","description":"File to blame (relative to repo root)."},
            "line_start":{"type":"integer","description":"Start line number (1-based, optional).","default":0},
            "line_end":{"type":"integer","description":"End line number (optional).","default":0},
            "ref":{"type":"string","description":"Blame at this commit/ref (default: HEAD).","default":""},
        },"required":["repo_path","file_path"]},
    )

    async def validate(self, repo_path: str, file_path: str, **_) -> None:
        p = Path(repo_path).expanduser().resolve()
        if not p.exists(): raise SkillValidationError(f"Repo does not exist: '{repo_path}'")
        if not (p/".git").exists(): raise SkillValidationError(f"Not a git repo: '{repo_path}'")

    async def execute(self, repo_path: str, file_path: str, line_start: int=0, line_end: int=0, ref: str="", **kwargs) -> SkillResult:
        call_id = kwargs.get("_skill_call_id","")
        t_start = time.monotonic()
        repo = Path(repo_path).expanduser().resolve()

        try:
            cmd = ["git","-C",str(repo),"blame","--line-porcelain"]
            if line_start > 0 and line_end > 0:
                cmd.extend(["-L",f"{line_start},{line_end}"])
            elif line_start > 0:
                cmd.extend(["-L",f"{line_start},+50"])
            if ref: cmd.append(ref)
            cmd.extend(["--",file_path])

            proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=25)
            if proc.returncode != 0:
                return SkillResult.fail(self.manifest.name, call_id, stderr.decode(errors="replace").strip(), "GitError")

            raw = stdout.decode(errors="replace")
            # Parse porcelain format into structured records
            lines_out = []
            blocks = raw.strip().split("\n\t")
            for block in blocks:
                parts = block.split("\n")
                if not parts: continue
                header = parts[0].split()
                commit_hash = header[0] if header else ""
                meta = {p.split(" ",1)[0]: p.split(" ",1)[1] if " " in p else ""
                        for p in parts[1:] if not p.startswith("\t")}
                line_content = parts[-1].lstrip("\t") if parts[-1].startswith("\t") else ""
                lines_out.append({
                    "commit": commit_hash[:8],
                    "author": meta.get("author",""),
                    "date":   meta.get("author-time",""),
                    "summary": meta.get("summary",""),
                    "line": line_content,
                })

            duration_ms = (time.monotonic()-t_start)*1000
            return SkillResult.ok(self.manifest.name, call_id,
                {"file":file_path,"repo":str(repo),"line_count":len(lines_out),"blame":lines_out}, duration_ms=duration_ms)
        except BaseException as e:
            return SkillResult.fail(self.manifest.name, call_id, f"{type(e).__name__}: {e}", type(e).__name__,
                                    duration_ms=(time.monotonic()-t_start)*1000)

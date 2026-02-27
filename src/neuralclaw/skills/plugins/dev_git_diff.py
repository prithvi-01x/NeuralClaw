"""
skills/plugins/dev_git_diff.py — Developer: Git Diff

Returns the diff for a commit, between two refs, or for the working tree.

Risk: LOW — fs:read, shell:run
"""
from __future__ import annotations
import asyncio, time
from pathlib import Path
from typing import ClassVar
from neuralclaw.skills.base import SkillBase
from neuralclaw.skills.types import RiskLevel, SkillManifest, SkillResult, SkillValidationError

class DevGitDiffSkill(SkillBase):
    manifest: ClassVar[SkillManifest] = SkillManifest(
        name="dev_git_diff",
        version="1.0.0",
        description="Show git diff for working tree, staged changes, a specific commit, or between two refs. Returns structured patch output.",
        category="developer",
        risk_level=RiskLevel.LOW,
        capabilities=frozenset({"fs:read","shell:run"}),
        timeout_seconds=30,
        parameters={"type":"object","properties":{
            "repo_path":{"type":"string","description":"Path to git repository."},
            "ref_from":{"type":"string","description":"Base ref/commit (leave empty for working tree diff).","default":""},
            "ref_to":{"type":"string","description":"Target ref/commit (leave empty for HEAD).","default":""},
            "staged":{"type":"boolean","description":"Show staged (index) diff instead of working tree.","default":False},
            "path_filter":{"type":"string","description":"Limit diff to this file or directory.","default":""},
            "stat_only":{"type":"boolean","description":"Return only diffstat summary, not full patch.","default":False},
            "max_chars":{"type":"integer","description":"Max characters of diff output (default 20000).","default":20000},
        },"required":["repo_path"]},
    )

    async def validate(self, repo_path: str, **_) -> None:
        p = Path(repo_path).expanduser().resolve()
        if not p.exists(): raise SkillValidationError(f"Path does not exist: '{repo_path}'")
        if not (p/".git").exists(): raise SkillValidationError(f"Not a git repo: '{repo_path}'")

    async def execute(self, repo_path: str, ref_from: str="", ref_to: str="", staged: bool=False,
                      path_filter: str="", stat_only: bool=False, max_chars: int=20000, **kwargs) -> SkillResult:
        call_id = kwargs.get("_skill_call_id","")
        t_start = time.monotonic()
        max_chars = min(int(max_chars), 100_000)
        repo = Path(repo_path).expanduser().resolve()

        try:
            cmd = ["git","-C",str(repo),"diff"]
            if stat_only: cmd.append("--stat")
            if staged: cmd.append("--staged")
            if ref_from: cmd.append(ref_from)
            if ref_to: cmd.append(ref_to)
            if path_filter: cmd.extend(["--",path_filter])

            proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=25)
            if proc.returncode != 0:
                return SkillResult.fail(self.manifest.name, call_id, stderr.decode(errors="replace").strip(), "GitError")

            diff_text = stdout.decode(errors="replace")
            truncated = len(diff_text) > max_chars
            if truncated: diff_text = diff_text[:max_chars] + f"\n\n[truncated — {len(diff_text):,} total chars]"

            # Count files changed
            files_changed = len([l for l in diff_text.splitlines() if l.startswith("diff --git")])
            duration_ms = (time.monotonic()-t_start)*1000
            return SkillResult.ok(self.manifest.name, call_id,
                {"repo":str(repo),"files_changed":files_changed,"truncated":truncated,"diff":diff_text}, duration_ms=duration_ms)
        except BaseException as e:
            return SkillResult.fail(self.manifest.name, call_id, f"{type(e).__name__}: {e}", type(e).__name__,
                                    duration_ms=(time.monotonic()-t_start)*1000)

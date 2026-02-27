"""
skills/plugins/dev_git_log.py — Developer: Git Log

Returns structured git log output for a repository: commits, authors, dates,
messages, and changed files. Runs git as a subprocess — no gitpython dependency.

Risk: LOW — fs:read + shell:run capabilities required.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import ClassVar

from neuralclaw.skills.base import SkillBase
from neuralclaw.skills.types import RiskLevel, SkillManifest, SkillResult, SkillValidationError


class DevGitLogSkill(SkillBase):
    manifest: ClassVar[SkillManifest] = SkillManifest(
        name="dev_git_log",
        version="1.0.0",
        description=(
            "Read git commit history for a repository. Returns commits with author, "
            "date, message, and changed files. Optionally filter by branch or author."
        ),
        category="developer",
        risk_level=RiskLevel.LOW,
        capabilities=frozenset({"fs:read", "shell:run"}),
        requires_confirmation=False,
        timeout_seconds=30,
        parameters={
            "type": "object",
            "properties": {
                "repo_path": {
                    "type": "string",
                    "description": "Absolute or relative path to the git repository root.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of commits to return (default 20, max 100).",
                    "default": 20,
                },
                "branch": {
                    "type": "string",
                    "description": "Branch or ref to log (default: current HEAD).",
                    "default": "",
                },
                "author": {
                    "type": "string",
                    "description": "Filter commits by author name or email (partial match).",
                    "default": "",
                },
                "since": {
                    "type": "string",
                    "description": "Only show commits after this date (e.g. '2 weeks ago', '2024-01-01').",
                    "default": "",
                },
                "show_files": {
                    "type": "boolean",
                    "description": "Include list of changed files per commit (default false).",
                    "default": False,
                },
            },
            "required": ["repo_path"],
        },
    )

    async def validate(self, repo_path: str, **_) -> None:
        p = Path(repo_path).expanduser().resolve()
        if not p.exists():
            raise SkillValidationError(f"Repository path does not exist: '{repo_path}'")
        if not (p / ".git").exists():
            raise SkillValidationError(
                f"'{repo_path}' is not a git repository (no .git directory found)."
            )

    async def execute(
        self,
        repo_path: str,
        limit: int = 20,
        branch: str = "",
        author: str = "",
        since: str = "",
        show_files: bool = False,
        **kwargs,
    ) -> SkillResult:
        call_id = kwargs.get("_skill_call_id", "")
        t_start = time.monotonic()
        limit = min(int(limit), 100)
        repo = Path(repo_path).expanduser().resolve()

        try:
            # Separator that won't appear in commit messages
            sep = "|||NEURALCLAW|||"
            fmt = f"%H{sep}%h{sep}%an{sep}%ae{sep}%ai{sep}%s{sep}%b"

            cmd = ["git", "-C", str(repo), "log", f"--pretty=format:{fmt}", f"-n{limit}"]
            if branch:
                cmd.append(branch)
            if author:
                cmd.extend(["--author", author])
            if since:
                cmd.extend(["--since", since])

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=20)

            if proc.returncode != 0:
                err = stderr.decode("utf-8", errors="replace").strip()
                return SkillResult.fail(
                    self.manifest.name, call_id,
                    f"git log failed: {err}", "GitError",
                )

            raw = stdout.decode("utf-8", errors="replace")
            commits = []

            for line in raw.strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.split(sep)
                if len(parts) < 6:
                    continue
                commit: dict = {
                    "hash": parts[0],
                    "short_hash": parts[1],
                    "author": parts[2],
                    "email": parts[3],
                    "date": parts[4],
                    "subject": parts[5],
                    "body": parts[6].strip() if len(parts) > 6 else "",
                }

                if show_files:
                    files_cmd = [
                        "git", "-C", str(repo), "diff-tree",
                        "--no-commit-id", "-r", "--name-status", parts[0],
                    ]
                    fp = await asyncio.create_subprocess_exec(
                        *files_cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    fstdout, _ = await asyncio.wait_for(fp.communicate(), timeout=5)
                    file_lines = fstdout.decode("utf-8", errors="replace").strip().split("\n")
                    commit["files"] = [f for f in file_lines if f.strip()]

                commits.append(commit)

            # Get current branch name
            branch_proc = await asyncio.create_subprocess_exec(
                "git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            b_out, _ = await branch_proc.communicate()
            current_branch = b_out.decode().strip()

            duration_ms = (time.monotonic() - t_start) * 1000
            return SkillResult.ok(
                skill_name=self.manifest.name,
                skill_call_id=call_id,
                output={
                    "repo": str(repo),
                    "current_branch": current_branch,
                    "commit_count": len(commits),
                    "commits": commits,
                },
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
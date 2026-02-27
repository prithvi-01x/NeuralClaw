"""
skills/builtin/terminal.py — Terminal Execution Skill

Migrated from tools/terminal.py to the SkillBase contract.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Optional

from skills.base import SkillBase
from skills.types import RiskLevel, SkillManifest, SkillResult, SkillValidationError

MAX_OUTPUT_BYTES = 50_000

_SECRET_ENV_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"API[_-]?KEY", r"SECRET", r"PASSWORD", r"PASSWD", r"TOKEN",
        r"AUTH", r"CREDENTIAL", r"PRIVATE[_-]?KEY", r"ACCESS[_-]?KEY",
        r"TELEGRAM", r"DISCORD", r"SLACK", r"OPENAI", r"ANTHROPIC",
        r"GEMINI", r"BYTEZ", r"DATABASE[_-]?URL", r"DB[_-]?(PASS|USER|HOST|URL)",
        r"REDIS[_-]?URL", r"MONGO.*URI", r"PGPASSWORD", r"AWS[_-]", r"GCP[_-]", r"AZURE[_-]",
    ]
]


def _safe_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items()
            if not any(p.search(k) for p in _SECRET_ENV_PATTERNS)}


class TerminalExecSkill(SkillBase):
    manifest = SkillManifest(
        name="terminal_exec",
        version="1.0.0",
        description=(
            "Execute a whitelisted shell command and return stdout/stderr. "
            "Only commands on the safety whitelist are permitted. "
            "Use for: file inspection, git status, running tests, grep, etc."
        ),
        category="terminal",
        risk_level=RiskLevel.HIGH,
        capabilities=frozenset({"shell:run"}),
        requires_confirmation=True,
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
                "working_dir": {"type": "string", "description": "Working directory", "default": "./data/agent_files"},
                "timeout_seconds": {"type": "integer", "description": "Timeout in seconds", "default": 30},
            },
            "required": ["command"],
        },
        timeout_seconds=60,
    )

    async def validate(self, command: str, working_dir: str = "./data/agent_files",
                        _whitelist_extra: Optional[list] = None, **_) -> None:
        from safety.whitelist import check_command
        extra = _whitelist_extra or []
        allowed, reason, _ = check_command(command, extra_allowed=extra)
        if not allowed:
            raise SkillValidationError(f"Command not allowed: {reason}")

    async def execute(self, command: str, working_dir: str = "./data/agent_files",
                       timeout_seconds: int = 30,
                       _allowed_paths: Optional[list] = None,
                       _whitelist_extra: Optional[list] = None, **_) -> SkillResult:
        call_id = _.get("_skill_call_id", "")
        resolved_dir = Path(working_dir).expanduser().resolve()

        # Validate working_dir is within an allowed path
        from safety.whitelist import check_path
        allowed = _allowed_paths or []
        ok, reason = check_path(str(resolved_dir), allowed, operation="read")
        if not ok:
            return SkillResult.fail(
                self.manifest.name, call_id,
                f"working_dir blocked by safety policy: {reason}",
                "SafetyBlockedError",
            )

        resolved_dir.mkdir(parents=True, exist_ok=True)

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(resolved_dir),
                env={**_safe_env(), "PYTHONUNBUFFERED": "1"},
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                return SkillResult.fail(
                    self.manifest.name, call_id,
                    f"Command timed out after {timeout_seconds}s",
                    "SkillTimeoutError",
                )
        except (OSError, RuntimeError) as e:
            return SkillResult.fail(self.manifest.name, call_id, f"Failed to start: {e}", type(e).__name__)

        stdout = stdout_b.decode("utf-8", errors="replace")
        stderr = stderr_b.decode("utf-8", errors="replace")
        if len(stdout) > MAX_OUTPUT_BYTES:
            stdout = stdout[:MAX_OUTPUT_BYTES] + f"\n[stdout truncated — {len(stdout)} total bytes]"
        if len(stderr) > MAX_OUTPUT_BYTES:
            stderr = stderr[:MAX_OUTPUT_BYTES] + f"\n[stderr truncated — {len(stderr)} total bytes]"

        return SkillResult.ok(
            skill_name=self.manifest.name, skill_call_id=call_id,
            output={
                "command": command,
                "working_dir": str(resolved_dir),
                "exit_code": proc.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "success": proc.returncode == 0,
            },
        )
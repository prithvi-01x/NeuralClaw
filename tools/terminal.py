"""
tools/terminal.py — Terminal Execution Tool

Allows the agent to execute whitelisted shell commands.
All commands pass through the Safety Kernel's whitelist check
before execution. Output is captured and returned as a string.

Registered tools:
  - terminal_exec → run a shell command, capture output
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
from pathlib import Path

from tools.tool_registry import registry
from tools.types import RiskLevel

# Absolute cap on output fed back to LLM
MAX_OUTPUT_BYTES = 50_000
DEFAULT_TIMEOUT = 30

# Env-var name patterns that indicate secrets.  Any variable whose name
# matches one of these patterns is stripped from the subprocess environment
# before execution so that commands like `printenv` cannot exfiltrate them.
_SECRET_ENV_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"API[_-]?KEY",
        r"SECRET",
        r"PASSWORD",
        r"PASSWD",
        r"TOKEN",
        r"AUTH",
        r"CREDENTIAL",
        r"PRIVATE[_-]?KEY",
        r"ACCESS[_-]?KEY",
        r"TELEGRAM",
        r"DISCORD",
        r"SLACK",
        r"OPENAI",
        r"ANTHROPIC",
        r"GEMINI",
        r"BYTEZ",
        r"DATABASE[_-]?URL",
        r"DB[_-]?(PASS|USER|HOST|URL)",
        r"REDIS[_-]?URL",
        r"MONGO.*URI",
        r"PGPASSWORD",
        r"AWS[_-]",
        r"GCP[_-]",
        r"AZURE[_-]",
    ]
]


def _safe_env() -> dict[str, str]:
    """
    Return a copy of os.environ with secret variables removed.

    This prevents the subprocess from inheriting API keys, tokens, and other
    credentials that the agent process has loaded from .env / the environment.
    Commands like `printenv` or `env` will only see non-sensitive variables.
    """
    return {
        key: value
        for key, value in os.environ.items()
        if not any(pat.search(key) for pat in _SECRET_ENV_PATTERNS)
    }


@registry.register(
    name="terminal_exec",
    description=(
        "Execute a shell command and return its stdout and stderr output. "
        "Only whitelisted commands are permitted. "
        "Working directory defaults to ~/agent_files. "
        "Use for: file inspection, git status, running tests, grep, etc."
    ),
    category="terminal",
    risk_level=RiskLevel.HIGH,
    requires_confirmation=True,
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute (e.g. 'ls -la', 'grep -r TODO .')",
            },
            "working_dir": {
                "type": "string",
                "description": "Working directory for the command (default: ~/agent_files)",
                "default": "~/agent_files",
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Max seconds to wait for command completion (default: 30)",
                "default": 30,
            },
        },
        "required": ["command"],
    },
)
async def terminal_exec(
    command: str,
    working_dir: str = "~/agent_files",
    timeout_seconds: int = DEFAULT_TIMEOUT,
) -> str:
    """
    Execute a shell command asynchronously and return captured output.

    Returns a JSON string with keys: stdout, stderr, exit_code, command.
    """
    resolved_dir = Path(working_dir).expanduser().resolve()

    # Validate working_dir is within an allowed path before using it as cwd.
    # The safety kernel only checks the command argument; this closes the gap.
    from safety.whitelist import check_path
    from config.settings import get_settings
    _allowed = get_settings().tools.filesystem.allowed_paths
    _ok, _reason = check_path(str(resolved_dir), _allowed, operation="read")
    if not _ok:
        return json.dumps({
            "command": command,
            "exit_code": -1,
            "stdout": "",
            "stderr": f"working_dir blocked by safety policy: {_reason}",
            "error": True,
        })

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
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return json.dumps({
                "command": command,
                "exit_code": -1,
                "stdout": "",
                "stderr": f"Command timed out after {timeout_seconds} seconds",
                "timed_out": True,
            })

    except Exception as e:
        return json.dumps({
            "command": command,
            "exit_code": -1,
            "stdout": "",
            "stderr": f"Failed to start process: {e}",
            "error": True,
        })

    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")

    # Truncate if too large
    if len(stdout) > MAX_OUTPUT_BYTES:
        stdout = stdout[:MAX_OUTPUT_BYTES] + f"\n[stdout truncated — {len(stdout)} total bytes]"
    if len(stderr) > MAX_OUTPUT_BYTES:
        stderr = stderr[:MAX_OUTPUT_BYTES] + f"\n[stderr truncated — {len(stderr)} total bytes]"

    return json.dumps({
        "command": command,
        "working_dir": str(resolved_dir),
        "exit_code": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "success": proc.returncode == 0,
    }, indent=2)
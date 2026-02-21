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
import shlex
from pathlib import Path

from tools.tool_registry import registry
from tools.types import RiskLevel

# Absolute cap on output fed back to LLM
MAX_OUTPUT_BYTES = 50_000
DEFAULT_TIMEOUT = 30


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
    resolved_dir.mkdir(parents=True, exist_ok=True)

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(resolved_dir),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
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
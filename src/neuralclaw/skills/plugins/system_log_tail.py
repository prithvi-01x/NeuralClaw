"""
skills/plugins/system_log_tail.py — System: Log Tail

Reads the last N lines from a log file or journald. Supports filtering by
keyword, log level, and time window. No external dependencies for file-based logs.

OS-aware auto-resolution when source='system':
  Linux  → tries journald first, falls back to /var/log/syslog, then /var/log/messages
  macOS  → /var/log/system.log
  Other  → /var/log/messages

Risk: LOW — fs:read capability required.
"""

from __future__ import annotations

import asyncio
import re
import sys
import time
from pathlib import Path
from typing import ClassVar

from neuralclaw.skills.base import SkillBase
from neuralclaw.skills.types import RiskLevel, SkillManifest, SkillResult, SkillValidationError

_MAX_LINES = 500
_MAX_FILE_SIZE_READ = 10 * 1024 * 1024  # 10 MB tail read limit

# Common log level patterns
_LEVEL_PATTERN = re.compile(
    r"\b(DEBUG|INFO|NOTICE|WARNING|WARN|ERROR|CRITICAL|FATAL|EMERGENCY|ALERT)\b",
    re.IGNORECASE,
)

# OS-aware fallback candidates for source='system'
_SYSTEM_LOG_CANDIDATES: dict = {
    "darwin": ["/var/log/system.log"],
    "linux":  ["/var/log/syslog", "/var/log/messages", "/var/log/kern.log"],
}


def _resolve_source(source: str) -> str:
    """
    Resolve the special 'system' alias to the correct log source for this OS.
    Linux  → 'journald' (preferred), or first existing /var/log/* file
    macOS  → /var/log/system.log
    Returns source unchanged for anything else.
    """
    if source.strip().lower() != "system":
        return source
    if sys.platform.startswith("linux"):
        return "journald"
    candidates = _SYSTEM_LOG_CANDIDATES.get(sys.platform, ["/var/log/messages"])
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return candidates[0]  # best guess — will produce a clear FileNotFoundError


class SystemLogTailSkill(SkillBase):
    manifest: ClassVar[SkillManifest] = SkillManifest(
        name="system_log_tail",
        version="1.1.0",
        description=(
            "Read the last N lines of a log file or systemd journal. "
            "Use source='system' to auto-detect the correct log for the current OS "
            "(journald on Linux, /var/log/system.log on macOS). "
            "Or pass an explicit path like '/var/log/system.log' or 'journald:nginx'. "
            "Filter by keyword or log level."
        ),
        category="system",
        risk_level=RiskLevel.LOW,
        capabilities=frozenset({"fs:read"}),
        requires_confirmation=False,
        timeout_seconds=20,
        parameters={
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": (
                        "'system' (auto-detect for current OS), "
                        "a log file path (e.g. '/var/log/system.log'), "
                        "'journald' for systemd journal, "
                        "or 'journald:<service>' (e.g. 'journald:nginx')."
                    ),
                    "default": "system",
                },
                "lines": {
                    "type": "integer",
                    "description": "Number of lines to return (default 50, max 500).",
                    "default": 50,
                },
                "filter_keyword": {
                    "type": "string",
                    "description": "Only return lines containing this string (case-insensitive).",
                    "default": "",
                },
                "filter_level": {
                    "type": "string",
                    "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", ""],
                    "description": "Only return lines at or above this log level.",
                    "default": "",
                },
                "since": {
                    "type": "string",
                    "description": "For journald: time window, e.g. '1 hour ago', '2024-01-01'.",
                    "default": "",
                },
            },
            "required": [],
        },
    )

    async def validate(self, source: str = "system", lines: int = 50, **_) -> None:
        if not source or not source.strip():
            raise SkillValidationError("source must be a non-empty string.")
        if lines > _MAX_LINES:
            raise SkillValidationError(f"lines must be <= {_MAX_LINES}.")
        # Resolve 'system' alias before checking — don't reject it
        resolved = _resolve_source(source)
        if not resolved.startswith("journald"):
            p = Path(resolved).expanduser()
            if not p.exists():
                raise SkillValidationError(
                    f"Log file does not exist: '{resolved}'. "
                    f"On macOS try '/var/log/system.log'; on Linux try 'journald' or '/var/log/syslog'."
                )
            if not p.is_file():
                raise SkillValidationError(f"'{resolved}' is not a file.")

    async def execute(
        self,
        source: str = "system",
        lines: int = 50,
        filter_keyword: str = "",
        filter_level: str = "",
        since: str = "",
        **kwargs,
    ) -> SkillResult:
        call_id = kwargs.get("_skill_call_id", "")
        t_start = time.monotonic()
        lines = min(int(lines), _MAX_LINES)
        keyword = filter_keyword.lower() if filter_keyword else ""
        level_order = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        min_level_idx = level_order.index(filter_level.upper()) if filter_level.upper() in level_order else -1

        # Resolve 'system' alias
        resolved = _resolve_source(source)

        try:
            raw_lines: list[str] = []

            # ── journald ────────────────────────────────────────────────────
            if resolved.startswith("journald"):
                service = resolved[len("journald:"):].strip() if ":" in resolved else ""
                cmd = ["journalctl", f"-n{lines}", "--no-pager", "--output=short-iso"]
                if service:
                    cmd.extend(["-u", service])
                if since:
                    cmd.extend(["--since", since])

                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
                except asyncio.TimeoutError:
                    proc.kill()
                    return SkillResult.fail(
                        self.manifest.name, call_id,
                        "journalctl timed out", "TimeoutError",
                    )

                if proc.returncode != 0:
                    err = stderr.decode("utf-8", errors="replace").strip()
                    if "command not found" in err.lower() or proc.returncode == 127:
                        return SkillResult.fail(
                            self.manifest.name, call_id,
                            "journalctl not found — system does not use systemd.",
                            "CommandNotFoundError",
                        )
                    return SkillResult.fail(
                        self.manifest.name, call_id,
                        f"journalctl failed: {err}", "JournaldError",
                    )

                raw_lines = stdout.decode("utf-8", errors="replace").splitlines()

            # ── File ────────────────────────────────────────────────────────
            else:
                p = Path(resolved).expanduser().resolve()
                file_size = p.stat().st_size

                def _read_tail() -> list[str]:
                    with open(p, "rb") as f:
                        if file_size > _MAX_FILE_SIZE_READ:
                            f.seek(-_MAX_FILE_SIZE_READ, 2)
                        data = f.read()
                    return data.decode("utf-8", errors="replace").splitlines()

                loop = asyncio.get_event_loop()
                all_lines = await loop.run_in_executor(None, _read_tail)
                raw_lines = all_lines[-lines:]

            # ── Filter ──────────────────────────────────────────────────────
            filtered: list[dict] = []
            for raw in raw_lines:
                if keyword and keyword not in raw.lower():
                    continue
                if min_level_idx >= 0:
                    m = _LEVEL_PATTERN.search(raw)
                    if m:
                        found_level = m.group(1).upper()
                        if found_level == "WARN":
                            found_level = "WARNING"
                        found_idx = level_order.index(found_level) if found_level in level_order else 0
                        if found_idx < min_level_idx:
                            continue
                    else:
                        continue

                level_match = _LEVEL_PATTERN.search(raw)
                detected_level = level_match.group(1).upper() if level_match else ""
                if detected_level == "WARN":
                    detected_level = "WARNING"
                filtered.append({
                    "line": raw,
                    "level": detected_level,
                })

            duration_ms = (time.monotonic() - t_start) * 1000
            return SkillResult.ok(
                skill_name=self.manifest.name,
                skill_call_id=call_id,
                output={
                    "source": source,
                    "resolved_source": resolved,
                    "lines_read": len(raw_lines),
                    "lines_returned": len(filtered),
                    "filter_keyword": filter_keyword,
                    "filter_level": filter_level,
                    "entries": filtered,
                },
                duration_ms=duration_ms,
            )

        except (OSError, PermissionError) as e:
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
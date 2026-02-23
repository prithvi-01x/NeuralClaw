"""
tools/filesystem.py — Filesystem Tools

Provides the agent with safe, path-restricted file system access.
All operations are confined to the allowed_paths defined in config.

Registered tools:
  - file_read      → read a text file
  - file_write     → write/overwrite a text file
  - file_append    → append to a text file
  - list_dir       → list directory contents
  - file_exists    → check if a path exists
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from tools.tool_registry import registry
from tools.types import RiskLevel

# Hard cap: refuse to load files larger than this into the LLM context
_MAX_READ_BYTES = 10 * 1024 * 1024  # 10 MB


def _revalidate_path(resolved: "Path", allowed_paths: list[str] | None = None) -> None:
    """
    Re-check the resolved path against the safety whitelist immediately before
    I/O to close the TOCTOU window between SafetyKernel.evaluate() and actual
    file access.  If the path is now outside the allowed set (e.g. a symlink
    was swapped between check and use), raise PermissionError.

    This is an in-process defence layer; it does not replace the SafetyKernel
    check — it complements it.
    """
    from safety.whitelist import check_path
    from config.settings import get_settings
    _allowed = allowed_paths or get_settings().tools.filesystem.allowed_paths
    ok, reason = check_path(str(resolved), _allowed, operation="read")
    if not ok:
        raise PermissionError(
            f"Path validation failed at execution time (possible symlink swap): {reason}"
        )


@registry.register(
    name="file_read",
    description=(
        "Read the contents of a text file. "
        "Returns the file content as a string. "
        "Only files within allowed paths can be read."
    ),
    category="filesystem",
    risk_level=RiskLevel.LOW,
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or ~ relative path to the file to read",
            },
            "encoding": {
                "type": "string",
                "description": "File encoding (default: utf-8)",
                "default": "utf-8",
            },
        },
        "required": ["path"],
    },
)
async def file_read(path: str, encoding: str = "utf-8") -> str:
    resolved = Path(path).expanduser().resolve()
    _revalidate_path(resolved)          # TOCTOU defence: re-check after resolve
    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"Path is not a file: {resolved}")
    size = resolved.stat().st_size
    if size > _MAX_READ_BYTES:
        return (
            f"[File too large to read directly: {size:,} bytes "
            f"(limit {_MAX_READ_BYTES:,} bytes). "
            f"Use terminal_exec with 'head', 'tail', or 'grep' to inspect it.]"
        )
    return resolved.read_text(encoding=encoding)


@registry.register(
    name="file_write",
    description=(
        "Write content to a file, creating it if it doesn't exist "
        "or overwriting if it does. "
        "Only files within allowed paths can be written."
    ),
    category="filesystem",
    risk_level=RiskLevel.MEDIUM,
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or ~ relative path to write to",
            },
            "content": {
                "type": "string",
                "description": "Text content to write",
            },
            "encoding": {
                "type": "string",
                "description": "File encoding (default: utf-8)",
                "default": "utf-8",
            },
            "create_dirs": {
                "type": "boolean",
                "description": "Create parent directories if they don't exist (default: true)",
                "default": True,
            },
        },
        "required": ["path", "content"],
    },
)
async def file_write(
    path: str,
    content: str,
    encoding: str = "utf-8",
    create_dirs: bool = True,
) -> str:
    resolved = Path(path).expanduser().resolve()
    _revalidate_path(resolved)          # TOCTOU defence: re-check after resolve
    if create_dirs:
        resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding=encoding)
    return f"Written {len(content)} characters to {resolved}"


@registry.register(
    name="file_append",
    description="Append content to an existing file (or create it if it doesn't exist).",
    category="filesystem",
    risk_level=RiskLevel.MEDIUM,
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file"},
            "content": {"type": "string", "description": "Content to append"},
        },
        "required": ["path", "content"],
    },
)
async def file_append(path: str, content: str) -> str:
    resolved = Path(path).expanduser().resolve()
    _revalidate_path(resolved)          # TOCTOU defence: re-check after resolve
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("a", encoding="utf-8") as f:
        f.write(content)
    return f"Appended {len(content)} characters to {resolved}"


@registry.register(
    name="list_dir",
    description=(
        "List the contents of a directory. "
        "Returns file names, sizes, and whether each entry is a file or directory."
    ),
    category="filesystem",
    risk_level=RiskLevel.LOW,
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path to list (default: ~/agent_files)",
                "default": "~/agent_files",
            },
            "show_hidden": {
                "type": "boolean",
                "description": "Include hidden files (starting with .)",
                "default": False,
            },
        },
        "required": [],
    },
)
async def list_dir(
    path: str = "~/agent_files",
    show_hidden: bool = False,
) -> str:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Directory not found: {resolved}")
    if not resolved.is_dir():
        raise ValueError(f"Path is not a directory: {resolved}")

    entries = []
    for entry in sorted(resolved.iterdir()):
        if not show_hidden and entry.name.startswith("."):
            continue
        try:
            stat = entry.stat()
            size = stat.st_size
            kind = "dir" if entry.is_dir() else "file"
            entries.append({
                "name": entry.name,
                "type": kind,
                "size_bytes": size if kind == "file" else None,
            })
        except PermissionError:
            entries.append({"name": entry.name, "type": "unknown", "error": "permission denied"})

    return json.dumps({"path": str(resolved), "entries": entries, "count": len(entries)}, indent=2)


@registry.register(
    name="file_exists",
    description="Check whether a file or directory exists at the given path.",
    category="filesystem",
    risk_level=RiskLevel.LOW,
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to check"},
        },
        "required": ["path"],
    },
)
async def file_exists(path: str) -> str:
    resolved = Path(path).expanduser().resolve()
    exists = resolved.exists()
    kind = "file" if resolved.is_file() else "directory" if resolved.is_dir() else "unknown"
    return json.dumps({
        "path": str(resolved),
        "exists": exists,
        "type": kind if exists else None,
    })
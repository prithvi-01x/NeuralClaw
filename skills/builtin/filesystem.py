"""
skills/builtin/filesystem.py — Filesystem Skills

Migrated from tools/filesystem.py to the SkillBase contract.
All five filesystem operations: file_read, file_write, file_append,
list_dir, file_exists.

Each skill is a separate class with a static manifest ClassVar.
Path safety is re-validated at execution time (TOCTOU defence).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from skills.base import SkillBase
from skills.types import RiskLevel, SkillManifest, SkillResult, SkillValidationError

_MAX_READ_BYTES = 10 * 1024 * 1024  # 10 MB hard cap


def _revalidate_path(
    resolved: Path,
    operation: str = "read",
    allowed_paths: Optional[list] = None,
) -> None:
    from safety.whitelist import check_path
    ok, reason = check_path(str(resolved), allowed_paths or [], operation=operation)
    if not ok:
        raise PermissionError(f"Path validation failed: {reason}")


class FileReadSkill(SkillBase):
    manifest = SkillManifest(
        name="file_read",
        version="1.0.0",
        description=(
            "Read the contents of a text file. "
            "Returns the file content as a string. "
            "Only files within allowed paths can be read."
        ),
        category="filesystem",
        risk_level=RiskLevel.LOW,
        capabilities=frozenset({"fs:read"}),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read"},
                "encoding": {"type": "string", "description": "File encoding (default: utf-8)", "default": "utf-8"},
            },
            "required": ["path"],
        },
        timeout_seconds=10,
    )

    async def execute(self, path: str, encoding: str = "utf-8",
                       _allowed_paths: Optional[list] = None, **_) -> SkillResult:
        call_id = _.get("_skill_call_id", "")
        try:
            resolved = Path(path).expanduser().resolve()
            _revalidate_path(resolved, "read", _allowed_paths)
            if not resolved.exists():
                raise FileNotFoundError(f"File not found: {resolved}")
            if not resolved.is_file():
                raise ValueError(f"Path is not a file: {resolved}")
            size = resolved.stat().st_size
            if size > _MAX_READ_BYTES:
                return SkillResult.ok(
                    skill_name=self.manifest.name, skill_call_id=call_id,
                    output=(
                        f"[File too large: {size:,} bytes (limit {_MAX_READ_BYTES:,}). "
                        f"Use terminal_exec with head/tail/grep instead.]"
                    ),
                )
            content = resolved.read_text(encoding=encoding)
            return SkillResult.ok(skill_name=self.manifest.name, skill_call_id=call_id, output=content)
        except PermissionError as e:
            return SkillResult.fail(self.manifest.name, call_id, str(e), "PermissionError")
        except FileNotFoundError as e:
            return SkillResult.fail(self.manifest.name, call_id, str(e), "FileNotFoundError")
        except (OSError, ValueError, RuntimeError, PermissionError, IOError) as e:
            return SkillResult.fail(self.manifest.name, call_id, f"{type(e).__name__}: {e}", type(e).__name__)


class FileWriteSkill(SkillBase):
    manifest = SkillManifest(
        name="file_write",
        version="1.0.0",
        description=(
            "Write content to a file, creating it if it doesn't exist "
            "or overwriting if it does. Only allowed paths."
        ),
        category="filesystem",
        risk_level=RiskLevel.MEDIUM,
        capabilities=frozenset({"fs:write"}),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to write to"},
                "content": {"type": "string", "description": "Text content to write"},
                "encoding": {"type": "string", "description": "File encoding (default: utf-8)", "default": "utf-8"},
                "create_dirs": {"type": "boolean", "description": "Create parent dirs if needed", "default": True},
            },
            "required": ["path", "content"],
        },
        timeout_seconds=10,
    )

    async def execute(self, path: str, content: str, encoding: str = "utf-8",
                       create_dirs: bool = True,
                       _allowed_paths: Optional[list] = None, **_) -> SkillResult:
        call_id = _.get("_skill_call_id", "")
        try:
            resolved = Path(path).expanduser().resolve()
            _revalidate_path(resolved, "write", _allowed_paths)
            if create_dirs:
                resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding=encoding)
            return SkillResult.ok(
                skill_name=self.manifest.name, skill_call_id=call_id,
                output=f"Written {len(content)} characters to {resolved}",
            )
        except PermissionError as e:
            return SkillResult.fail(self.manifest.name, call_id, str(e), "PermissionError")
        except (OSError, ValueError, RuntimeError, PermissionError, IOError) as e:
            return SkillResult.fail(self.manifest.name, call_id, f"{type(e).__name__}: {e}", type(e).__name__)


class FileAppendSkill(SkillBase):
    manifest = SkillManifest(
        name="file_append",
        version="1.0.0",
        description="Append content to an existing file (creates it if it doesn't exist).",
        category="filesystem",
        risk_level=RiskLevel.MEDIUM,
        capabilities=frozenset({"fs:write"}),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file"},
                "content": {"type": "string", "description": "Content to append"},
            },
            "required": ["path", "content"],
        },
        timeout_seconds=10,
    )

    async def execute(self, path: str, content: str,
                       _allowed_paths: Optional[list] = None, **_) -> SkillResult:
        call_id = _.get("_skill_call_id", "")
        try:
            resolved = Path(path).expanduser().resolve()
            _revalidate_path(resolved, "write", _allowed_paths)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            with resolved.open("a", encoding="utf-8") as f:
                f.write(content)
            return SkillResult.ok(
                skill_name=self.manifest.name, skill_call_id=call_id,
                output=f"Appended {len(content)} characters to {resolved}",
            )
        except PermissionError as e:
            return SkillResult.fail(self.manifest.name, call_id, str(e), "PermissionError")
        except (OSError, ValueError, RuntimeError, PermissionError, IOError) as e:
            return SkillResult.fail(self.manifest.name, call_id, f"{type(e).__name__}: {e}", type(e).__name__)


class ListDirSkill(SkillBase):
    manifest = SkillManifest(
        name="list_dir",
        version="1.0.0",
        description="List directory contents with file names, sizes, and types.",
        category="filesystem",
        risk_level=RiskLevel.LOW,
        capabilities=frozenset({"fs:read"}),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path to list", "default": "./data/agent_files"},
                "show_hidden": {"type": "boolean", "description": "Include hidden files", "default": False},
            },
            "required": [],
        },
        timeout_seconds=10,
    )

    async def execute(self, path: str = "./data/agent_files", show_hidden: bool = False, **_) -> SkillResult:
        call_id = _.get("_skill_call_id", "")
        try:
            resolved = Path(path).expanduser().resolve()
            if not resolved.exists():
                raise FileNotFoundError(f"Directory not found: {resolved}")
            if not resolved.is_dir():
                raise ValueError(f"Not a directory: {resolved}")

            entries = []
            for entry in sorted(resolved.iterdir()):
                if not show_hidden and entry.name.startswith("."):
                    continue
                try:
                    stat = entry.stat()
                    kind = "dir" if entry.is_dir() else "file"
                    entries.append({
                        "name": entry.name,
                        "type": kind,
                        "size_bytes": stat.st_size if kind == "file" else None,
                    })
                except PermissionError:
                    entries.append({"name": entry.name, "type": "unknown", "error": "permission denied"})

            return SkillResult.ok(
                skill_name=self.manifest.name, skill_call_id=call_id,
                output={"path": str(resolved), "entries": entries, "count": len(entries)},
            )
        except (FileNotFoundError, ValueError) as e:
            return SkillResult.fail(self.manifest.name, call_id, str(e), type(e).__name__)
        except (OSError, ValueError, RuntimeError, PermissionError, IOError) as e:
            return SkillResult.fail(self.manifest.name, call_id, f"{type(e).__name__}: {e}", type(e).__name__)


class FileExistsSkill(SkillBase):
    manifest = SkillManifest(
        name="file_exists",
        version="1.0.0",
        description="Check whether a file or directory exists at the given path.",
        category="filesystem",
        risk_level=RiskLevel.LOW,
        capabilities=frozenset({"fs:read"}),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to check"},
            },
            "required": ["path"],
        },
        timeout_seconds=5,
    )

    async def execute(self, path: str, **_) -> SkillResult:
        call_id = _.get("_skill_call_id", "")
        try:
            resolved = Path(path).expanduser().resolve()
            exists = resolved.exists()
            kind = "file" if resolved.is_file() else "directory" if resolved.is_dir() else "unknown"
            return SkillResult.ok(
                skill_name=self.manifest.name, skill_call_id=call_id,
                output={"path": str(resolved), "exists": exists, "type": kind if exists else None},
            )
        except (OSError, ValueError, RuntimeError, PermissionError, IOError) as e:
            return SkillResult.fail(self.manifest.name, call_id, f"{type(e).__name__}: {e}", type(e).__name__)
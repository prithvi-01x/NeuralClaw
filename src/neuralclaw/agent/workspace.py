"""
agent/workspace.py — Workspace File Loader

Reads ~/neuralclaw/*.md workspace files and injects them into the
LLM system prompt so the agent always knows:
    - who it is (SOUL.md)
    - who the user is (USER.md)
    - what it should remember (MEMORY.md)

Files are cached per workspace_dir and invalidated whenever the file's
mtime changes, so edits are picked up on the next turn without restarting.

Usage (inside ContextBuilder or Orchestrator):
    loader = WorkspaceLoader()
    extra = loader.build_context_block()
    # Pass `extra` as extra_system to ContextBuilder.build()
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

try:
    from neuralclaw.observability.logger import get_logger
    _log = get_logger(__name__)
except ImportError:
    import logging
    _log = logging.getLogger(__name__)

WORKSPACE_DIR = Path("~/neuralclaw").expanduser()

# Order and headers for each file injected into the system prompt
_FILE_ORDER: list[tuple[str, str]] = [
    ("SOUL.md",   "## Assistant Identity"),
    ("USER.md",   "## About the User"),
    ("MEMORY.md", "## Long-Term Memory"),
]

# Soft cap on total chars injected (keeps context lean)
_MAX_CHARS = 4_000


class WorkspaceLoader:
    """
    Reads workspace markdown files and builds a system-prompt block.

    Caches by file mtime — zero re-read cost if files haven't changed.
    Thread-safe for read-only use (single asyncio event loop assumed).
    """

    def __init__(self, workspace_dir: Optional[Path] = None) -> None:
        self._dir = (workspace_dir or WORKSPACE_DIR).expanduser()
        # Cache: filename → (mtime, content)
        self._cache: dict[str, tuple[float, str]] = {}

    def build_context_block(self) -> str:
        """
        Return a formatted context block for injection into the system prompt.
        Returns empty string if workspace dir doesn't exist or all files are empty.
        """
        if not self._dir.exists():
            return ""

        sections: list[str] = []
        total_chars = 0

        for filename, header in _FILE_ORDER:
            content = self._read_cached(filename)
            if not content:
                continue

            # Trim if we're approaching the budget
            remaining = _MAX_CHARS - total_chars
            if remaining <= 100:
                break

            if len(content) > remaining:
                content = content[:remaining] + "\n\n[...truncated]"

            sections.append(f"{header}\n\n{content}")
            total_chars += len(content)

        if not sections:
            return ""

        return (
            "---\n"
            "# Workspace Context\n\n"
            + "\n\n---\n\n".join(sections)
            + "\n---"
        )

    def invalidate(self, filename: Optional[str] = None) -> None:
        """
        Invalidate the cache for a specific file or all files.
        Call after the onboard wizard writes new workspace files.
        """
        if filename:
            self._cache.pop(filename, None)
        else:
            self._cache.clear()

    def is_configured(self) -> bool:
        """Return True if the workspace directory exists and has at least SOUL.md."""
        return (self._dir / "SOUL.md").exists()

    def agent_name(self) -> Optional[str]:
        """
        Extract the agent name from SOUL.md for display in the CLI banner.
        Returns None if SOUL.md doesn't exist or has no name line.
        """
        content = self._read_cached("SOUL.md")
        if not content:
            return None
        for line in content.splitlines():
            stripped = line.strip()
            # Look for a line that follows "## Name" header
            if stripped and not stripped.startswith("#") and not stripped.startswith("-"):
                # Skip lines that are clearly not a name
                if len(stripped) < 50 and "\n" not in stripped:
                    return stripped
        return None

    def soul_personality(self) -> Optional[str]:
        """Extract the personality line from SOUL.md."""
        content = self._read_cached("SOUL.md")
        if not content:
            return None
        in_personality = False
        for line in content.splitlines():
            if line.strip() == "## Personality":
                in_personality = True
                continue
            if in_personality:
                stripped = line.strip()
                if stripped.startswith("##"):
                    break
                if stripped:
                    return stripped
        return None

    # ── Private ───────────────────────────────────────────────────────────────

    def _read_cached(self, filename: str) -> str:
        """Return file content from cache if mtime unchanged, else re-read."""
        fpath = self._dir / filename
        if not fpath.exists():
            return ""

        try:
            mtime = fpath.stat().st_mtime
        except OSError:
            return ""

        cached = self._cache.get(filename)
        if cached and cached[0] == mtime:
            return cached[1]

        try:
            content = fpath.read_text().strip()
            self._cache[filename] = (mtime, content)
            return content
        except OSError as e:
            _log.warning("workspace.read_error", file=filename, error=str(e))
            return ""


# ── Module-level singleton (shared across orchestrator and context builder) ──

_default_loader: Optional[WorkspaceLoader] = None


def get_workspace_loader(workspace_dir: Optional[Path] = None) -> WorkspaceLoader:
    """Return the shared WorkspaceLoader instance, creating it if needed."""
    global _default_loader
    if _default_loader is None or (workspace_dir and workspace_dir != _default_loader._dir):
        _default_loader = WorkspaceLoader(workspace_dir)
    return _default_loader
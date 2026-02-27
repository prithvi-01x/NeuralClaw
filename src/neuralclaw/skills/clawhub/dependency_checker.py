"""
skills/clawhub/dependency_checker.py — Binary & Env Dependency Checks

Checks whether required CLI binaries and environment variables are available
for ClawHub skills. Optionally runs install directives to satisfy missing deps.
"""

from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuralclaw.skills.clawhub.bridge_parser import ClawhubInstallDirective, ClawhubRequires

try:
    from neuralclaw.observability.logger import get_logger as _get_logger
    _log_raw = _get_logger(__name__)
    _STRUCTLOG = True
except ImportError:
    import logging as _logging
    _log_raw = _logging.getLogger(__name__)
    _STRUCTLOG = False


def _log(level: str, event: str, **kwargs) -> None:
    if _STRUCTLOG:
        getattr(_log_raw, level)(event, **kwargs)
    else:
        extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
        getattr(_log_raw, level)("%s %s", event, extra)


# ─────────────────────────────────────────────────────────────────────────────
# Binary checks
# ─────────────────────────────────────────────────────────────────────────────

def check_bins(requires: "ClawhubRequires") -> tuple[bool, list[str]]:
    """
    Check that all required binaries are on PATH.

    Returns:
        (all_ok, list_of_missing_bins)
        For anyBins, passes if at least one is found.
    """
    missing: list[str] = []

    # All required bins must exist
    for b in requires.bins:
        if shutil.which(b) is None:
            missing.append(b)

    # At least one of anyBins must exist
    if requires.any_bins:
        found_any = any(shutil.which(b) is not None for b in requires.any_bins)
        if not found_any:
            missing.append(f"one of: {', '.join(requires.any_bins)}")

    return (len(missing) == 0, missing)


def check_env(requires: "ClawhubRequires") -> tuple[bool, list[str]]:
    """
    Check that all required environment variables are set.

    Returns:
        (all_ok, list_of_missing_vars)
    """
    missing = [v for v in requires.env if not os.environ.get(v)]
    return (len(missing) == 0, missing)


# ─────────────────────────────────────────────────────────────────────────────
# Install directives
# ─────────────────────────────────────────────────────────────────────────────

# Maps install directive kind → shell command template
_INSTALL_COMMANDS = {
    "brew": lambda d: f"brew install {d.formula}" if d.formula else None,
    "node": lambda d: f"npm install -g {d.package}" if d.package else None,
    "go":   lambda d: f"go install {d.package}" if d.package else None,
    "uv":   lambda d: f"uv tool install {d.package}" if d.package else None,
    "pip":  lambda d: f"pip install {d.package}" if d.package else None,
}


def build_install_command(directive: "ClawhubInstallDirective") -> str | None:
    """
    Build the shell command string for an install directive.
    Returns None if the kind is unknown or the directive is incomplete.
    """
    builder = _INSTALL_COMMANDS.get(directive.kind)
    if builder is None:
        _log("warning", "clawhub_deps.unknown_install_kind", kind=directive.kind)
        return None
    return builder(directive)


async def run_install_directive(
    directive: "ClawhubInstallDirective",
    skill_bus=None,
) -> bool:
    """
    Run a ClawHub install directive via terminal_exec if available,
    or directly via subprocess.

    Returns True if install succeeded.
    """
    cmd = build_install_command(directive)
    if cmd is None:
        return False

    _log("info", "clawhub_deps.installing",
         kind=directive.kind, command=cmd)

    # Try via SkillBus terminal_exec if available
    if skill_bus is not None:
        try:
            from neuralclaw.skills.types import SkillCall
            call = SkillCall(
                id="_clawhub_install",
                skill_name="terminal_exec",
                arguments={"command": cmd},
            )
            result = await skill_bus.dispatch(call)
            if result.success:
                _log("info", "clawhub_deps.install_ok", command=cmd)
                return True
            else:
                _log("warning", "clawhub_deps.install_failed",
                     command=cmd, error=result.error)
                return False
        except Exception as e:
            _log("warning", "clawhub_deps.install_error",
                 command=cmd, error=str(e))
            return False

    # Fallback: direct subprocess
    import asyncio
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        if proc.returncode == 0:
            _log("info", "clawhub_deps.install_ok", command=cmd)
            return True
        else:
            _log("warning", "clawhub_deps.install_failed",
                 command=cmd, stderr=stderr.decode(errors="replace")[:200])
            return False
    except Exception as e:
        _log("warning", "clawhub_deps.install_error",
             command=cmd, error=str(e))
        return False

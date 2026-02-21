"""
safety/whitelist.py — Command and Path Whitelists

Defines what terminal commands and filesystem paths the agent
is allowed to access. This is the first line of defense in the
Safety Kernel — deny everything not explicitly permitted.

Philosophy: deny-first. The agent can only do what's in the whitelist.
Extra commands can be added via config.yaml (tools.terminal.whitelist_extra).
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Terminal command whitelist
# ─────────────────────────────────────────────────────────────────────────────

# Commands the agent may run without elevated trust
# Format: exact binary names (we check argv[0])
ALLOWED_COMMANDS: frozenset[str] = frozenset([
    # File inspection
    "ls", "ll", "la", "cat", "head", "tail", "wc", "file",
    "find", "locate", "which", "whereis", "stat",
    # Text search / processing
    "grep", "rg", "awk", "sed", "cut", "sort", "uniq", "diff",
    "tr", "xargs", "jq",
    # Archive / compression
    "tar", "gzip", "gunzip", "zip", "unzip", "zcat",
    # Networking (read-only)
    "curl", "wget", "ping", "dig", "nslookup", "netstat", "ss",
    # Python / package management
    "python", "python3", "pip", "pip3", "uv",
    # Git (read operations)
    "git",
    # System info
    "echo", "printf", "date", "uptime", "df", "du", "free",
    "ps", "top", "htop", "env", "printenv", "pwd",
    # Text editors (output only, no interactive)
    "cat", "less", "more",
    # Build / test
    "make", "pytest", "ruff", "mypy",
    # Node / JS
    "node", "npm", "npx",
    # Misc utilities
    "sleep", "true", "false", "test", "expr", "bc",
    "md5sum", "sha256sum", "base64",
])

# Commands that are HIGH risk — allowed but require confirmation
HIGH_RISK_COMMANDS: frozenset[str] = frozenset([
    "git",      # git push, git reset --hard
    "pip", "pip3", "uv",   # modifies environment
    "npm",      # installs packages
    "make",     # could run arbitrary recipes
    "curl", "wget",        # network exfiltration risk
])

# Patterns that are ALWAYS blocked regardless of whitelist
# These are checked against the full command string
BLOCKED_PATTERNS: list[re.Pattern] = [
    re.compile(r"\brm\s+(-\w*r\w*|-\w*f\w*|--recursive|--force)", re.IGNORECASE),
    re.compile(r"\brm\b.*\s+/"),                    # rm targeting root paths
    re.compile(r"\brmdir\b"),
    re.compile(r"\bmkfs\b"),                        # format filesystem
    re.compile(r"\bdd\b.*\bof=/dev"),               # write to device
    re.compile(r"\b(shutdown|reboot|halt|poweroff)\b"),
    re.compile(r"\bkill\s+-9\s+1\b"),               # kill init
    re.compile(r"\b(su|sudo)\b"),                   # privilege escalation
    re.compile(r"\bchmod\s+.*777"),                 # world-writable
    re.compile(r"\b>\s*/dev/(sd|hd|nvme|vd)"),     # overwrite block device
    re.compile(r"\beval\b.*\$\("),                  # eval with subshell
    re.compile(r":()\{.*\}.*;:"),                   # fork bomb
    re.compile(r"\bbase64\b.*\|\s*bash"),           # encoded payload execution
    re.compile(r"\bcurl\b.*\|\s*(bash|sh|python)"), # remote code execution
    re.compile(r"\bwget\b.*-O.*\|\s*(bash|sh)"),    # remote code execution
]

# Git subcommands that require confirmation
HIGH_RISK_GIT_SUBCOMMANDS: frozenset[str] = frozenset([
    "push", "reset", "rebase", "merge", "force", "clean", "gc",
    "remote", "config", "stash drop",
])


# ─────────────────────────────────────────────────────────────────────────────
# Filesystem path allowlist
# ─────────────────────────────────────────────────────────────────────────────

# Paths the agent can always read from (system info, not sensitive)
ALWAYS_READABLE_PATHS: frozenset[str] = frozenset([
    "/tmp",
    "/var/tmp",
])

# Paths that are ALWAYS blocked for read and write
BLOCKED_PATHS: frozenset[str] = frozenset([
    "/etc/passwd",
    "/etc/shadow",
    "/etc/sudoers",
    "/root",
    "/proc",
    "/sys",
    "/dev",
    "/boot",
    "/private/etc",     # macOS
])


# ─────────────────────────────────────────────────────────────────────────────
# Whitelist checker functions
# ─────────────────────────────────────────────────────────────────────────────


def check_command(
    command: str,
    extra_allowed: Optional[list[str]] = None,
) -> tuple[bool, str, bool]:
    """
    Check if a terminal command is permitted.

    Args:
        command:       The full command string to check.
        extra_allowed: Additional allowed commands from config.

    Returns:
        (allowed, reason, is_high_risk)
        - allowed:      True if command may proceed
        - reason:       Human-readable explanation
        - is_high_risk: True if command needs confirmation even if allowed
    """
    command = command.strip()
    if not command:
        return False, "Empty command", False

    # 1. Check blocked patterns first (always wins)
    for pattern in BLOCKED_PATTERNS:
        if pattern.search(command):
            return False, f"Blocked pattern matched: {pattern.pattern}", False

    # 2. Parse the command to get argv[0]
    try:
        parts = shlex.split(command)
    except ValueError as e:
        return False, f"Could not parse command: {e}", False

    if not parts:
        return False, "Empty command after parsing", False

    binary = Path(parts[0]).name  # strip any path prefix (e.g. /usr/bin/python → python)

    # 3. Build effective allowlist
    effective_allowed = set(ALLOWED_COMMANDS)
    if extra_allowed:
        effective_allowed.update(extra_allowed)

    # 4. Check if binary is allowed
    if binary not in effective_allowed:
        return False, f"Command '{binary}' is not in the allowed commands list", False

    # 5. Determine if it's high risk
    is_high_risk = binary in HIGH_RISK_COMMANDS

    # 6. Special case: git subcommand check
    if binary == "git" and len(parts) > 1:
        subcommand = parts[1].lower()
        if subcommand in HIGH_RISK_GIT_SUBCOMMANDS:
            is_high_risk = True

    return True, f"Command '{binary}' is permitted", is_high_risk


def check_path(
    path: str | Path,
    allowed_paths: list[str],
    operation: str = "read",
) -> tuple[bool, str]:
    """
    Check if a filesystem path is within allowed bounds.

    Args:
        path:          The path to check (will be resolved to absolute).
        allowed_paths: List of allowed base directories from config.
        operation:     "read" or "write" (write is stricter).

    Returns:
        (allowed, reason)
    """
    try:
        resolved = Path(path).expanduser().resolve()
    except Exception as e:
        return False, f"Could not resolve path: {e}"

    path_str = str(resolved)

    # 1. Check always-blocked paths
    for blocked in BLOCKED_PATHS:
        if path_str.startswith(blocked):
            return False, f"Path '{path_str}' is in the blocked paths list"

    # 2. Check allowed paths
    expanded_allowed = [
        str(Path(p).expanduser().resolve()) for p in allowed_paths
    ]

    for allowed in expanded_allowed:
        if path_str.startswith(allowed):
            return True, f"Path is within allowed directory: {allowed}"

    # 3. Allow reads from always-readable paths
    if operation == "read":
        for readable in ALWAYS_READABLE_PATHS:
            if path_str.startswith(readable):
                return True, f"Path is in always-readable location: {readable}"

    return (
        False,
        f"Path '{path_str}' is outside allowed directories: {expanded_allowed}",
    )
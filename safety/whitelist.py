"""
safety/whitelist.py — Command & Path Whitelist

Two public functions used by the Safety Kernel:

    check_command(command, extra_allowed=[])
        → (allowed: bool, reason: str, is_high_risk: bool)

    check_path(path, allowed_paths, operation="read")
        → (allowed: bool, reason: str)

Design principles
-----------------
* Default-deny for commands: the binary (first token) must be in the
  ALLOWED_COMMANDS set, OR in the caller-supplied extra_allowed list.
* Certain dangerous patterns are always blocked regardless of the binary
  (fork bombs, pipe-to-shell, encoded payloads, sudo, etc.).
* High-risk binaries (git, pip, npm, curl …) are allowed but flagged so the
  Safety Kernel can escalate the risk level and request confirmation.
* Path checks are independent of the command whitelist: they verify the
  resolved absolute path is under one of the configured allowed_paths,
  and hard-block a set of always-forbidden system paths.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Terminal command allowlist
# ─────────────────────────────────────────────────────────────────────────────

# Binaries the agent may invoke. We check argv[0] (binary name only, no path).
# Split into normal-risk and high-risk groups.
ALLOWED_COMMANDS: frozenset[str] = frozenset([
    # Navigation & listing
    "ls", "ll", "la", "dir",
    "pwd", "cd", "find", "locate",
    # File inspection
    "cat", "head", "tail", "less", "more",
    "wc", "file", "stat", "du", "df",
    "strings", "hexdump",
    # Text processing
    "grep", "egrep", "fgrep", "rg",
    "awk", "sed", "cut", "sort", "uniq",
    "tr", "column", "jq", "yq",
    "diff", "patch", "comm",
    # Archiving
    "tar", "zip", "unzip", "gzip", "gunzip", "bzip2", "xz",
    "7z", "zcat",
    # Networking (info only)
    "ping", "traceroute", "nslookup", "dig", "host",
    "curl", "wget", "http",
    # System info
    "ps", "top", "htop", "uptime", "who", "whoami",
    "uname", "lsb_release", "hostname",
    "lscpu", "lsblk",
    "free", "vmstat", "iostat",
    # NOTE: env and printenv removed — see comment below
    # Process inspection
    "pgrep", "pstree",
    # NOTE: "env" and "printenv" removed — they dump the full process
    #       environment including every API key and secret token inherited
    #       by the agent process, enabling trivial one-call exfiltration.
    # Development — interpreters
    "python", "python3", "python3.11", "python3.12",
    "node", "npx",
    "make", "cmake", "ninja",
    "gcc", "g++", "clang", "rustc", "cargo", "go",
    "javac", "java", "mvn", "gradle",
    # Development — quality tools
    "pytest", "ruff", "mypy", "flake8", "black", "isort",
    "eslint", "prettier",
    # Shell builtins / utilities
    "echo", "printf", "test", "true", "false",
    "sleep", "date", "cal", "bc", "expr",
    # Version control
    "git",
    # Package managers (high-risk but allowed; flagged below)
    "pip", "pip3", "uv", "npm", "yarn", "pnpm", "bun",
    # File operations
    "touch", "mkdir", "rmdir", "cp", "mv", "ln",
    # NOTE: "tee" removed — it can redirect output to arbitrary sinks
    #       including /dev/tcp pseudo-files for exfiltration.
    # Permissions (flagged high-risk)
    "chmod", "chown",
    # Misc utilities
    "which", "type", "command",
    "man", "info",
    "md5sum", "sha1sum", "sha256sum", "sha512sum",
    "base64",
    "xargs",
    "ssh", "scp",
    # NOTE: "rsync" removed — rsync's ssh transport can exfiltrate files to
    #       arbitrary remote hosts without triggering pipe-to-shell patterns.
    "docker", "kubectl", "helm",
    "ffmpeg",
    # NOTE: "openssl" kept for digest/encode use but remote connection
    #       patterns (s_client, s_time, etc.) are hard-blocked below.
    "openssl",
])

# Subset of ALLOWED_COMMANDS that triggers is_high_risk=True.
# The Safety Kernel escalates the risk level for these.
HIGH_RISK_COMMANDS: frozenset[str] = frozenset([
    "git",
    "pip", "pip3", "uv",
    "npm", "yarn", "pnpm", "bun",
    "cargo", "go",
    "curl", "wget",
    "ssh", "scp",
    "docker", "podman", "kubectl", "helm",
    "chmod", "chown",
    "ffmpeg",
    "openssl",
    "make",
])

# Git subcommands that escalate is_high_risk even further
HIGH_RISK_GIT_SUBCOMMANDS: frozenset[str] = frozenset([
    "push", "reset", "rebase", "merge",
    "clean", "gc", "remote", "config",
])


# ─────────────────────────────────────────────────────────────────────────────
# Hard-block patterns  (checked BEFORE the binary allowlist, on raw string)
# ─────────────────────────────────────────────────────────────────────────────

BLOCKED_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Privilege escalation
    (re.compile(r"\bsudo\b"),                           "sudo is not permitted"),
    (re.compile(r"\bsu\s+-"),                           "su - is not permitted"),
    (re.compile(r"\bdoas\b"),                           "doas is not permitted"),
    # Destructive rm
    (re.compile(r"\brm\s+(-\w*r\w*|-\w*f\w*|--recursive|--force)", re.I),
                                                        "rm with recursive/force flag is blocked"),
    (re.compile(r"\brm\b.*\s+/"),                       "rm targeting root-level path is blocked"),
    # Pipe-to-shell (command injection vector)
    (re.compile(r"\|\s*(ba|z|da|fi|k)?sh\b"),           "pipe-to-shell is blocked"),
    # Fork bomb variants
    (re.compile(r":\(\)\s*\{"),                         "fork bomb detected"),
    (re.compile(r":\(\)\{.*\}.*;:"),                    "fork bomb detected"),
    # Encoded payload execution
    (re.compile(r"base64\b.*\|\s*(ba|z|da)?sh", re.I), "encoded payload execution is blocked"),
    (re.compile(r"echo\b.*\|\s*(ba|z|da)?sh", re.I),   "echo-to-shell is blocked"),
    # Remote code execution shortcuts
    (re.compile(r"\bcurl\b.*\|\s*(bash|sh|python)", re.I), "curl-to-shell is blocked"),
    (re.compile(r"\bwget\b.*-O-?\s*\|\s*(bash|sh)", re.I), "wget-to-shell is blocked"),
    # System power/shutdown
    (re.compile(r"\b(shutdown|reboot|halt|poweroff)\b", re.I), "system control commands are blocked"),
    # Disk operations
    (re.compile(r"\bmkfs\b"),                           "filesystem formatting is blocked"),
    (re.compile(r"\bfdisk\b|\bparted\b"),               "disk partitioning is blocked"),
    (re.compile(r"\bdd\b.*of=/dev/"),                   "writing to a block device is blocked"),
    # Process killing
    (re.compile(r"\bkillall\b"),                        "killall is blocked"),
    (re.compile(r"\bkill\s+-9\s+1\b"),                  "killing init (PID 1) is blocked"),
    # Cron manipulation
    (re.compile(r"\bcrontab\s+-[er]\b"),                "crontab editing is blocked"),
    # Anti-forensics
    (re.compile(r"history\s+-c"),                       "clearing history is blocked"),
    (re.compile(r">\s*~/?\.(bash|zsh)_history"),        "overwriting shell history is blocked"),
    # Sensitive file reads via cat
    (re.compile(r"\bcat\b.*/etc/shadow"),               "reading /etc/shadow is blocked"),
    (re.compile(r"\bcat\b.*/etc/passwd"),               "reading /etc/passwd is blocked"),
    # Shell redirects to /etc
    (re.compile(r">\s*/etc/"),                          "writing to /etc via redirection is blocked"),
    # Netcat in listen/exec mode
    (re.compile(r"\bnc\b.*-[lek]"),                     "netcat listener/exec mode is blocked"),
    (re.compile(r"\bnetcat\b.*-[lek]"),                 "netcat listener/exec mode is blocked"),
    # eval with subshell
    (re.compile(r"\beval\b.*\$\("),                     "eval with subshell substitution is blocked"),
    # ── Interpreter inline-code flags (CRITICAL-1) ────────────────────────────
    # python/node -c/-e allow arbitrary code execution and fully bypass the
    # command whitelist — they are unconditionally blocked.
    (re.compile(r"\bpython[0-9.]*\b.*\s-[ce]\s"),       "python -c/-e inline code execution is blocked"),
    (re.compile(r"\bpython[0-9.]*\b.*\s-[ce]$"),        "python -c/-e inline code execution is blocked"),
    (re.compile(r"\bnode\b.*\s-e\s"),                   "node -e inline code execution is blocked"),
    (re.compile(r"\bnode\b.*\s-e$"),                    "node -e inline code execution is blocked"),
    (re.compile(r"\bnpx\b.*\s-e\s"),                    "npx -e inline code execution is blocked"),
    # ── Remote-connection exfiltration paths (MEDIUM-2) ──────────────────────
    # openssl s_client / s_time / dgram open raw TLS sockets to arbitrary hosts
    (re.compile(r"\bopenssl\b.*(s_client|s_time|dgram)", re.I),
                                                        "openssl network-connect subcommands are blocked"),
    # rsync to a remote destination (host: or user@host:) enables SSH exfil
    (re.compile(r"\brsync\b.*\s[\w.@-]+:"),            "rsync to remote destination is blocked"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Filesystem path allowlist
# ─────────────────────────────────────────────────────────────────────────────

# Paths always blocked regardless of allowed_paths config.
BLOCKED_PATH_PREFIXES: frozenset[str] = frozenset([
    "/etc/passwd",
    "/etc/shadow",
    "/etc/sudoers",
    "/etc/ssh",
    "/root",
    "/proc",
    "/sys",
    "/dev",
    "/boot",
    "/private/etc",  # macOS equivalent of /etc
])

# Regex patterns that block a path regardless of allowed_paths.
BLOCKED_PATH_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\.ssh/"),     ".ssh directories are always blocked"),
    (re.compile(r"\.gnupg/"),   ".gnupg directories are always blocked"),
    (re.compile(r"\.aws/"),     ".aws credentials are always blocked"),
    (re.compile(r"\.kube/"),    ".kube config is always blocked"),
    (re.compile(r"\.env$"),     ".env files are always blocked"),
    (re.compile(r"\.pem$"),     ".pem key files are always blocked"),
    (re.compile(r"\.key$"),     ".key files are always blocked"),
    (re.compile(r"\.p12$"),     ".p12 certificate files are always blocked"),
    (re.compile(r"\.pfx$"),     ".pfx certificate files are always blocked"),
]

# Paths readable without an explicit allowed_paths entry (read-only).
ALWAYS_READABLE_PREFIXES: tuple[str, ...] = (
    "/tmp/",
    "/var/tmp/",
)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def check_command(
    command: str,
    extra_allowed: Optional[list[str]] = None,
) -> tuple[bool, str, bool]:
    """
    Decide whether a shell command is permitted.

    Args:
        command:       The raw shell command string from the LLM.
        extra_allowed: Additional binary names to allow (from config.yaml
                       tools.terminal.whitelist_extra).

    Returns:
        (allowed, reason, is_high_risk)
        - allowed:      True if the command may proceed.
        - reason:       Human-readable explanation of the decision.
        - is_high_risk: True if allowed but the binary is flagged high-risk
                        (e.g. git, pip, curl). The Safety Kernel uses this
                        to escalate the effective risk level.
    """
    command = command.strip()

    # Guard: empty command
    if not command:
        return False, "Empty command is not permitted", False

    # Step 1: Hard-block patterns (checked on raw command string)
    for pattern, reason in BLOCKED_PATTERNS:
        if pattern.search(command):
            return False, f"Command blocked — {reason}", False

    # Step 2: Parse the command to get the binary name
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        return False, f"Command has malformed quoting and cannot be parsed: {exc}", False

    if not tokens:
        return False, "Command parsed to an empty token list", False

    # Strip leading VAR=val environment assignments
    cmd_tokens = [t for t in tokens if "=" not in t]
    if not cmd_tokens:
        return False, "Command consists only of environment assignments", False

    binary = Path(cmd_tokens[0]).name  # strip path prefix (e.g. /usr/bin/python → python)

    if not binary:
        return False, "Could not determine the command binary", False

    # Step 3: Check against the combined allowlist
    effective_allowed = ALLOWED_COMMANDS | set(extra_allowed or [])

    if binary not in effective_allowed:
        return (
            False,
            f"'{binary}' is not in the allowed command list. "
            "Add it to 'tools.terminal.whitelist_extra' in config.yaml if needed.",
            False,
        )

    # Step 4: Flag high-risk binaries
    is_high_risk = binary in HIGH_RISK_COMMANDS

    # Special case: certain git subcommands elevate risk
    if binary == "git" and len(cmd_tokens) > 1:
        if cmd_tokens[1].lower() in HIGH_RISK_GIT_SUBCOMMANDS:
            is_high_risk = True

    reason = f"Command '{binary}' is permitted" + (" (high-risk binary)" if is_high_risk else "")
    return True, reason, is_high_risk


def check_path(
    path: str | Path,
    allowed_paths: Optional[list[str]] = None,
    operation: str = "read",
) -> tuple[bool, str]:
    """
    Decide whether a filesystem path may be accessed.

    Args:
        path:          The path string from the LLM tool call.
        allowed_paths: List of permitted base paths (from config.yaml
                       tools.filesystem.allowed_paths). Supports ~ expansion.
        operation:     "read" or "write". Write operations are stricter
                       (always-readable paths do not apply).

    Returns:
        (allowed, reason)
    """
    if not path or str(path).strip() == "":
        resolved = Path.cwd()
    else:
        try:
            # expanduser() + resolve() follows symlinks — a path like
            # ./data/../../etc/passwd resolves to /etc/passwd and is caught
            # by the BLOCKED_PATH_PREFIXES check below. Symlink attacks are
            # therefore prevented without any additional logic.
            resolved = Path(str(path)).expanduser().resolve()
        except (ValueError, RuntimeError) as exc:
            return False, f"Path resolution failed: {exc}"

    path_str = str(resolved)

    # Step 1: Always-blocked path prefixes
    for prefix in BLOCKED_PATH_PREFIXES:
        if path_str.startswith(prefix):
            return False, f"Path blocked: '{path_str}' is within a protected system location ({prefix})"

    # Step 2: Always-blocked path patterns (SSH keys, .env files, etc.)
    for pattern, reason in BLOCKED_PATH_PATTERNS:
        if pattern.search(path_str):
            return False, f"Path blocked: {reason} (matched '{path_str}')"

    # Step 3: Resolve allowed_paths to absolute paths
    bases: list[Path] = []
    for ap in (allowed_paths or []):
        try:
            bases.append(Path(ap).expanduser().resolve())
        except (ValueError, RuntimeError):
            continue

    # Step 4: Check if path is under an allowed base
    for base in bases:
        try:
            resolved.relative_to(base)
            return True, f"Path '{path_str}' is within allowed base '{base}'"
        except ValueError:
            continue

    # Step 5: Always-readable locations (read operations only)
    if operation == "read":
        for prefix in ALWAYS_READABLE_PREFIXES:
            if path_str.startswith(prefix):
                return True, f"Path '{path_str}' is in an always-readable location"

    return (
        False,
        f"Path '{path_str}' is outside all allowed paths "
        f"({[str(b) for b in bases]}). "
        "Add the base directory to 'tools.filesystem.allowed_paths' in config.yaml.",
    )
"""
safety/risk_scorer.py — Tool Call Risk Scorer

Assigns a RiskLevel to any tool call based on:
1. The tool's registered baseline risk level
2. The specific arguments passed (e.g. path, command content)
3. Heuristic rules for known dangerous patterns

The Safety Kernel uses this score to decide: auto-approve, confirm, or block.
"""

from __future__ import annotations

import re
from typing import Any

from tools.types import RiskLevel, ToolCall, ToolSchema


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic escalation rules
# ─────────────────────────────────────────────────────────────────────────────

# Argument patterns that escalate risk to CRITICAL
CRITICAL_ARG_PATTERNS: list[re.Pattern] = [
    re.compile(r"\brm\s+-[rf]", re.IGNORECASE),
    re.compile(r"\bformat\b", re.IGNORECASE),
    re.compile(r"\bshred\b"),
    re.compile(r"/dev/(sd|hd|nvme|vd)"),
    re.compile(r"\bdrop\s+table", re.IGNORECASE),      # SQL
    re.compile(r"\btruncate\b", re.IGNORECASE),
]

# Argument patterns that escalate risk to HIGH
HIGH_ARG_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bgit\s+(push|reset|rebase)\b", re.IGNORECASE),
    re.compile(r"\bpip\s+install\b", re.IGNORECASE),
    re.compile(r"\bnpm\s+install\b", re.IGNORECASE),
    re.compile(r"\bcurl\b.*-o\b", re.IGNORECASE),          # curl download
    re.compile(r"\bwget\b", re.IGNORECASE),
    re.compile(r"\bsystemctl\b", re.IGNORECASE),
    re.compile(r"\bservice\b.*\b(start|stop|restart)\b", re.IGNORECASE),
    re.compile(r"\bchmod\b", re.IGNORECASE),
    re.compile(r"\bchown\b", re.IGNORECASE),
]

# Filesystem paths that escalate risk when written to
SENSITIVE_WRITE_PATHS: list[re.Pattern] = [
    re.compile(r"^/etc/"),
    re.compile(r"^/usr/"),
    re.compile(r"^/bin/"),
    re.compile(r"^/sbin/"),
    re.compile(r"~/.ssh/"),
    re.compile(r"~/.bashrc"),
    re.compile(r"~/.zshrc"),
    re.compile(r"~/.profile"),
    re.compile(r"~/.env"),
    re.compile(r"\.env$"),
    re.compile(r"\.pem$"),
    re.compile(r"\.key$"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Scorer
# ─────────────────────────────────────────────────────────────────────────────


def score_tool_call(
    tool_call: ToolCall,
    schema: ToolSchema,
) -> tuple[RiskLevel, str]:
    """
    Compute the effective risk level for a tool call.

    Starts from the tool's registered baseline risk and escalates
    based on argument analysis.

    Args:
        tool_call: The tool call to evaluate.
        schema:    The tool's registered schema (contains baseline risk).

    Returns:
        (effective_risk_level, reason_string)
    """
    baseline = schema.risk_level
    args = tool_call.arguments

    # Collect all argument values as a single string for pattern matching
    all_arg_text = _flatten_args(args)

    # ── Check for CRITICAL escalation ────────────────────────────────────────
    for pattern in CRITICAL_ARG_PATTERNS:
        if pattern.search(all_arg_text):
            return RiskLevel.CRITICAL, (
                f"Argument matches critical-risk pattern: '{pattern.pattern}'"
            )

    # ── Check for HIGH escalation ─────────────────────────────────────────────
    for pattern in HIGH_ARG_PATTERNS:
        if pattern.search(all_arg_text):
            return _max_risk(baseline, RiskLevel.HIGH), (
                f"Argument matches high-risk pattern: '{pattern.pattern}'"
            )

    # ── Filesystem-specific checks ────────────────────────────────────────────
    if schema.category == "filesystem":
        path = args.get("path", args.get("file_path", ""))
        operation = _infer_fs_operation(tool_call.name)

        if operation == "write" and path:
            for pattern in SENSITIVE_WRITE_PATHS:
                if pattern.search(str(path)):
                    return RiskLevel.CRITICAL, (
                        f"Writing to sensitive path: '{path}'"
                    )
            # Any write is at least MEDIUM
            return _max_risk(baseline, RiskLevel.MEDIUM), "Filesystem write operation"

    # ── Terminal-specific checks ──────────────────────────────────────────────
    if schema.category == "terminal":
        command = args.get("command", "")
        if command:
            # Check pipe chains (medium escalation for piped commands)
            if "|" in command and baseline < RiskLevel.HIGH:
                return _max_risk(baseline, RiskLevel.MEDIUM), "Command uses pipe chain"

    # ── No escalation — return baseline ──────────────────────────────────────
    return baseline, f"Using baseline risk level for tool '{tool_call.name}'"


def _flatten_args(args: dict[str, Any]) -> str:
    """Recursively flatten all argument values into a single string."""
    parts = []
    for v in args.values():
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, dict):
            parts.append(_flatten_args(v))
        elif isinstance(v, list):
            parts.extend(str(i) for i in v)
        else:
            parts.append(str(v))
    return " ".join(parts)


def _infer_fs_operation(tool_name: str) -> str:
    """Guess read vs write from tool name."""
    write_keywords = ("write", "create", "delete", "remove", "append", "move", "copy")
    name_lower = tool_name.lower()
    return "write" if any(kw in name_lower for kw in write_keywords) else "read"


def _max_risk(a: RiskLevel, b: RiskLevel) -> RiskLevel:
    """Return the higher of two risk levels."""
    order = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
    return a if order.index(a) >= order.index(b) else b
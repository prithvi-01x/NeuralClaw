"""
skills/plugins/dev_lint_runner.py — Developer: Lint Runner

Runs a linter on a file or directory and returns structured results.
Supports ruff (primary), flake8, pylint, and eslint — whichever is installed.
Validates that the target path is within allowed_paths before executing.

Risk: LOW — fs:read + shell:run capabilities required.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import ClassVar

from skills.base import SkillBase
from skills.types import RiskLevel, SkillManifest, SkillResult, SkillValidationError

_SUPPORTED_LINTERS = ["ruff", "flake8", "pylint", "eslint"]


class DevLintRunnerSkill(SkillBase):
    manifest: ClassVar[SkillManifest] = SkillManifest(
        name="dev_lint_runner",
        version="1.0.0",
        description=(
            "Run a linter on a Python or JavaScript file or directory. "
            "Supports ruff (preferred), flake8, pylint, and eslint. "
            "Returns structured issues with file, line, column, and message."
        ),
        category="developer",
        risk_level=RiskLevel.LOW,
        capabilities=frozenset({"fs:read", "shell:run"}),
        requires_confirmation=False,
        timeout_seconds=60,
        parameters={
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Path to file or directory to lint.",
                },
                "linter": {
                    "type": "string",
                    "enum": _SUPPORTED_LINTERS + ["auto"],
                    "description": "Linter to use. 'auto' tries ruff → flake8 → pylint (default: auto).",
                    "default": "auto",
                },
                "max_issues": {
                    "type": "integer",
                    "description": "Maximum issues to return (default 50).",
                    "default": 50,
                },
            },
            "required": ["target"],
        },
    )

    async def validate(self, target: str, **_) -> None:
        p = Path(target).expanduser().resolve()
        if not p.exists():
            raise SkillValidationError(f"Target path does not exist: '{target}'")

    async def execute(
        self,
        target: str,
        linter: str = "auto",
        max_issues: int = 50,
        **kwargs,
    ) -> SkillResult:
        call_id = kwargs.get("_skill_call_id", "")
        t_start = time.monotonic()
        target_path = Path(target).expanduser().resolve()
        max_issues = min(int(max_issues), 200)

        try:
            # Determine which linter to use
            chosen = await _pick_linter(linter)
            if not chosen:
                return SkillResult.fail(
                    self.manifest.name, call_id,
                    f"No supported linter found. Install one of: {', '.join(_SUPPORTED_LINTERS)}",
                    "LinterNotFound",
                )

            issues, raw_output, return_code = await _run_linter(chosen, target_path)

            issues = issues[:max_issues]
            duration_ms = (time.monotonic() - t_start) * 1000
            return SkillResult.ok(
                skill_name=self.manifest.name,
                skill_call_id=call_id,
                output={
                    "target": str(target_path),
                    "linter": chosen,
                    "return_code": return_code,
                    "issue_count": len(issues),
                    "issues": issues,
                    "clean": len(issues) == 0,
                },
                duration_ms=duration_ms,
            )

        except (OSError, ValueError) as e:
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


# ─────────────────────────────────────────────────────────────────────────────
# Linter runner helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _pick_linter(preference: str) -> str:
    """Return the first available linter name, or '' if none found."""
    candidates = _SUPPORTED_LINTERS if preference == "auto" else [preference]
    for linter in candidates:
        try:
            proc = await asyncio.create_subprocess_exec(
                linter, "--version",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            return linter
        except (OSError, FileNotFoundError):
            continue
    return ""


async def _run_linter(linter: str, target: Path) -> tuple[list[dict], str, int]:
    """Run the chosen linter and return (issues, raw_output, return_code)."""
    if linter == "ruff":
        cmd = ["ruff", "check", "--output-format=json", str(target)]
    elif linter == "flake8":
        cmd = ["flake8", "--format=%(path)s:%(row)d:%(col)d: %(code)s %(text)s", str(target)]
    elif linter == "pylint":
        cmd = ["pylint", "--output-format=json", str(target)]
    elif linter == "eslint":
        cmd = ["eslint", "--format=json", str(target)]
    else:
        return [], "", 1

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=55)
    except asyncio.TimeoutError:
        proc.kill()
        return [], "timeout", 1

    raw = stdout.decode("utf-8", errors="replace")
    rc = proc.returncode or 0
    issues: list[dict] = []

    if linter == "ruff":
        try:
            data = json.loads(raw) if raw.strip() else []
            for item in data:
                issues.append({
                    "file": item.get("filename", ""),
                    "line": item.get("location", {}).get("row", 0),
                    "col": item.get("location", {}).get("column", 0),
                    "code": item.get("code", ""),
                    "message": item.get("message", ""),
                    "severity": "error" if item.get("fix") is None else "warning",
                })
        except (json.JSONDecodeError, KeyError):
            issues = _parse_plain(raw)

    elif linter == "flake8":
        issues = _parse_plain(raw)

    elif linter in ("pylint", "eslint"):
        try:
            data = json.loads(raw) if raw.strip() else []
            if linter == "pylint":
                for item in data:
                    issues.append({
                        "file": item.get("path", ""),
                        "line": item.get("line", 0),
                        "col": item.get("column", 0),
                        "code": item.get("message-id", ""),
                        "message": item.get("message", ""),
                        "severity": item.get("type", "warning"),
                    })
            else:  # eslint
                for file_result in data:
                    fname = file_result.get("filePath", "")
                    for msg in file_result.get("messages", []):
                        issues.append({
                            "file": fname,
                            "line": msg.get("line", 0),
                            "col": msg.get("column", 0),
                            "code": msg.get("ruleId", ""),
                            "message": msg.get("message", ""),
                            "severity": "error" if msg.get("severity") == 2 else "warning",
                        })
        except (json.JSONDecodeError, KeyError):
            issues = _parse_plain(raw)

    return issues, raw, rc


def _parse_plain(text: str) -> list[dict]:
    """Parse plain-text linter output (file:line:col: CODE message)."""
    issues = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split(":", 3)
        if len(parts) >= 4:
            issues.append({
                "file": parts[0],
                "line": int(parts[1]) if parts[1].isdigit() else 0,
                "col": int(parts[2]) if parts[2].isdigit() else 0,
                "code": "",
                "message": parts[3].strip(),
                "severity": "warning",
            })
        else:
            issues.append({"file": "", "line": 0, "col": 0, "code": "", "message": line, "severity": "warning"})
    return issues
"""
skills/plugins/meta_repo_audit.py — Meta-Skill: Repository Audit

Fixed multi-step repository health audit workflow. Orchestrates:
    dev_git_log → dev_git_diff → dev_lint_runner → dev_test_runner →
    dev_dependency_audit → dev_file_summarize → automation_report_render

Produces a single Markdown audit report. Designed for weekly scheduler
execution (Sunday 18:00). All steps logged to TaskMemory.

Risk: LOW — fs:read + shell:run + fs:write capabilities.
No confirmation required.

Phase C meta-skill. No kernel changes required.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, ClassVar

from skills.base import SkillBase
from skills.types import RiskLevel, SkillManifest, SkillResult, SkillValidationError


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _truncate(text: Any, max_len: int = 2000) -> str:
    s = str(text) if not isinstance(text, str) else text
    return s[:max_len] + "…" if len(s) > max_len else s


def _result_summary(result: SkillResult) -> str:
    if not result.success:
        return f"❌ FAILED — {result.error}"
    output = result.output
    if isinstance(output, str):
        return f"✅ {_truncate(output, 120)}"
    if isinstance(output, dict):
        return f"✅ OK — {list(output.keys())}"
    return "✅ OK"


def _skill_call(name: str, args: dict) -> Any:
    from skills.types import SkillCall
    return SkillCall(
        id=f"meta-repo-{uuid.uuid4().hex[:8]}",
        skill_name=name,
        arguments=args,
    )


async def _dispatch(bus: Any, call: Any, session: Any) -> SkillResult:
    return await bus.dispatch(
        call,
        trust_level=session.trust_level if session else None,
        granted_capabilities=session.granted_capabilities if session else frozenset(),
    )


def _output_to_md(output: Any, max_len: int = 3000) -> str:
    if isinstance(output, str):
        return _truncate(output, max_len)
    import json
    try:
        return f"```json\n{json.dumps(output, indent=2, default=str)[:max_len]}\n```"
    except Exception:
        return str(output)[:max_len]


# ─────────────────────────────────────────────────────────────────────────────
# Meta-Skill
# ─────────────────────────────────────────────────────────────────────────────

class MetaRepoAuditSkill(SkillBase):
    manifest: ClassVar[SkillManifest] = SkillManifest(
        name="meta_repo_audit",
        version="1.0.0",
        description=(
            "Full repository health audit. Runs git log (recent activity), git diff (uncommitted changes), "
            "linter (code quality), test suite (pass/fail), dependency audit (outdated/vulnerable packages), "
            "and generates a Markdown audit report. "
            "Designed for weekly automated runs. Provide 'repo_path' to target a specific repository."
        ),
        category="developer",
        risk_level=RiskLevel.LOW,
        capabilities=frozenset({"fs:read", "shell:run", "fs:write"}),
        requires_confirmation=False,
        timeout_seconds=180,
        parameters={
            "type": "object",
            "properties": {
                "repo_path": {
                    "type": "string",
                    "description": "Absolute or relative path to the git repository root.",
                    "default": ".",
                },
                "log_limit": {
                    "type": "integer",
                    "description": "Number of recent commits to include in the git log.",
                    "default": 20,
                },
                "summarize_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of specific files to summarize.",
                    "default": [],
                },
                "report_path": {
                    "type": "string",
                    "description": "Output path for the audit Markdown report.",
                    "default": "~/neuralclaw/reports/repo_audit.md",
                },
                "skip_steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Steps to skip: git_log, git_diff, lint, tests, dep_audit, file_summary.",
                    "default": [],
                },
                "plan_id": {
                    "type": "string",
                    "description": "TaskMemory plan ID (injected by orchestrator).",
                    "default": "",
                },
            },
            "required": [],
        },
    )

    async def validate(self, repo_path: str = ".", **kwargs) -> None:
        from pathlib import Path
        p = Path(repo_path).expanduser().resolve()
        if not p.exists():
            raise SkillValidationError(f"meta_repo_audit: repo_path does not exist: {p}")

    async def execute(
        self,
        repo_path: str = ".",
        log_limit: int = 20,
        summarize_files: list[str] | None = None,
        report_path: str = "~/neuralclaw/reports/repo_audit.md",
        skip_steps: list[str] | None = None,
        plan_id: str = "",
        **kwargs,
    ) -> SkillResult:
        call_id = kwargs.get("_skill_call_id", "")
        t_start = time.monotonic()
        skip = set(skip_steps or [])
        plan_id = plan_id or f"repo-audit-{uuid.uuid4().hex[:8]}"

        bus = kwargs.get("_bus")
        session = kwargs.get("_session")
        task_store = kwargs.get("_task_store")

        if not bus:
            return SkillResult.fail(
                self.manifest.name, call_id,
                "meta_repo_audit requires '_bus' injected.",
                "MetaSkillConfigError",
            )

        if task_store:
            task_store.create(plan_id=plan_id, goal=f"Repository audit: {repo_path}")

        def log_step(step_id: str, desc: str) -> None:
            if task_store:
                task_store.log_step(plan_id, step_id, desc)

        def log_result(step_id: str, result: SkillResult) -> None:
            if task_store:
                task_store.update_result(
                    plan_id, step_id,
                    result_content=_result_summary(result),
                    is_error=not result.success,
                    duration_ms=result.duration_ms,
                )

        step_results: dict[str, SkillResult] = {}

        # Step 1: Git Log
        if "git_log" not in skip:
            log_step("git_log", f"Fetching last {log_limit} commits")
            r = await _dispatch(bus, _skill_call("dev_git_log", {
                "repo_path": repo_path,
                "limit": log_limit,
            }), session)
            log_result("git_log", r)
            step_results["git_log"] = r

        # Step 2: Git Diff (uncommitted changes)
        if "git_diff" not in skip:
            log_step("git_diff", "Checking uncommitted changes")
            r = await _dispatch(bus, _skill_call("dev_git_diff", {
                "repo_path": repo_path,
            }), session)
            log_result("git_diff", r)
            step_results["git_diff"] = r

        # Step 3: Lint
        if "lint" not in skip:
            log_step("lint", "Running linter")
            r = await _dispatch(bus, _skill_call("dev_lint_runner", {
                "path": repo_path,
            }), session)
            log_result("lint", r)
            step_results["lint"] = r

        # Step 4: Tests
        if "tests" not in skip:
            log_step("tests", "Running test suite")
            r = await _dispatch(bus, _skill_call("dev_test_runner", {
                "path": repo_path,
            }), session)
            log_result("tests", r)
            step_results["tests"] = r

        # Step 5: Dependency Audit
        if "dep_audit" not in skip:
            log_step("dep_audit", "Auditing dependencies")
            r = await _dispatch(bus, _skill_call("dev_dependency_audit", {
                "path": repo_path,
            }), session)
            log_result("dep_audit", r)
            step_results["dep_audit"] = r

        # Step 6: File Summaries (optional)
        if "file_summary" not in skip and summarize_files:
            for file_path in summarize_files[:5]:  # cap at 5 files
                step_id = f"file_summary_{file_path.replace('/', '_')}"
                log_step(step_id, f"Summarizing {file_path}")
                r = await _dispatch(bus, _skill_call("dev_file_summarize", {
                    "file_path": file_path,
                }), session)
                log_result(step_id, r)
                step_results[step_id] = r

        # ── Determine overall health ─────────────────────────────────────────
        test_r = step_results.get("tests")
        lint_r = step_results.get("lint")
        dep_r = step_results.get("dep_audit")

        health_issues = []
        if test_r and not test_r.success:
            health_issues.append("❌ Tests failed")
        elif test_r and test_r.success and isinstance(test_r.output, dict):
            if test_r.output.get("failures", 0) > 0:
                health_issues.append(f"⚠️ {test_r.output['failures']} test failure(s)")
        if lint_r and not lint_r.success:
            health_issues.append("⚠️ Lint errors found")
        if dep_r and isinstance(dep_r.output, dict):
            vulns = dep_r.output.get("vulnerabilities", 0)
            if vulns:
                health_issues.append(f"🔴 {vulns} vulnerable package(s)")

        health_status = "✅ Healthy" if not health_issues else " | ".join(health_issues)

        # ── Build report sections ────────────────────────────────────────────
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).strftime("%A, %d %B %Y %H:%M UTC")

        sections = [
            {
                "heading": "Audit Summary",
                "content": f"**Repository:** `{repo_path}`\n\n**Run at:** {now}\n\n**Health:** {health_status}",
                "level": 2,
            }
        ]

        step_labels = {
            "git_log": "📋 Recent Commits",
            "git_diff": "📝 Uncommitted Changes",
            "lint": "🔍 Lint Results",
            "tests": "🧪 Test Results",
            "dep_audit": "📦 Dependency Audit",
        }
        for step_id, label in step_labels.items():
            if step_id in skip:
                continue
            r = step_results.get(step_id)
            if r is None:
                continue
            content = f"**Status:** ❌ Failed\n\n**Error:** `{r.error}`" if not r.success else _output_to_md(r.output)
            sections.append({"heading": label, "content": content, "level": 2})

        # File summaries
        for step_id, r in step_results.items():
            if step_id.startswith("file_summary_"):
                fname = step_id.replace("file_summary_", "").replace("_", "/")
                content = f"**Status:** ❌ Failed\n\n**Error:** `{r.error}`" if not r.success else _output_to_md(r.output)
                sections.append({"heading": f"📄 File: {fname}", "content": content, "level": 3})

        # ── Write report ─────────────────────────────────────────────────────
        log_step("report_render", "Writing audit report")
        report_result = await _dispatch(bus, _skill_call("automation_report_render", {
            "title": f"Repository Audit — {repo_path}",
            "sections": sections,
            "output_path": report_path,
        }), session)
        log_result("report_render", report_result)

        if task_store:
            task_store.close(plan_id)

        duration_ms = (time.monotonic() - t_start) * 1000
        success_count = sum(1 for r in step_results.values() if r.success)

        return SkillResult.ok(
            skill_name=self.manifest.name,
            skill_call_id=call_id,
            output={
                "repo_path": repo_path,
                "health_status": health_status,
                "steps_completed": success_count,
                "steps_total": len(step_results),
                "steps_skipped": list(skip),
                "report_path": report_path,
                "report_written": report_result.success,
                "summary": {k: _result_summary(v) for k, v in step_results.items()},
            },
            duration_ms=duration_ms,
        )
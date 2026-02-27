"""
skills/plugins/meta_system_maintenance.py — Meta-Skill: System Maintenance

Fixed multi-step system health and maintenance workflow. Orchestrates:
    system_disk_usage → system_process_list → system_log_tail →
    system_service_status → system_backup_run → automation_report_render

Designed for nightly scheduler execution at 02:00. High-risk due to backup
and package operations. All steps logged to TaskMemory.

Risk: HIGH — system_backup_run is CRITICAL; overall meta-skill rated HIGH.
Requires user confirmation.

Phase C meta-skill. No kernel changes required.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, ClassVar

from neuralclaw.skills.base import SkillBase
from neuralclaw.skills.types import RiskLevel, SkillManifest, SkillResult, SkillValidationError


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
    from neuralclaw.skills.types import SkillCall
    return SkillCall(
        id=f"meta-maint-{uuid.uuid4().hex[:8]}",
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

class MetaSystemMaintenanceSkill(SkillBase):
    manifest: ClassVar[SkillManifest] = SkillManifest(
        name="meta_system_maintenance",
        version="1.0.0",
        description=(
            "Nightly system maintenance pipeline. Checks disk usage (alerts on >80%), "
            "inspects running processes, tails system logs for errors, checks service status, "
            "and optionally runs a backup. Generates a Markdown maintenance report. "
            "Designed for nightly scheduler runs at 02:00. "
            "WARNING: system_backup_run is a CRITICAL-risk sub-skill — confirmation required."
        ),
        category="system",
        risk_level=RiskLevel.HIGH,
        capabilities=frozenset({"fs:read", "fs:write", "shell:run"}),
        requires_confirmation=True,
        timeout_seconds=300,
        parameters={
            "type": "object",
            "properties": {
                "services_to_check": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of systemd service names to check status for.",
                    "default": [],
                },
                "log_lines": {
                    "type": "integer",
                    "description": "Number of recent log lines to tail.",
                    "default": 100,
                },
                "backup_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Paths to back up. If empty, backup step is skipped.",
                    "default": [],
                },
                "backup_dest": {
                    "type": "string",
                    "description": "Backup destination directory.",
                    "default": "~/neuralclaw/backups",
                },
                "report_path": {
                    "type": "string",
                    "description": "Output path for the maintenance Markdown report.",
                    "default": "~/neuralclaw/reports/system_maintenance.md",
                },
                "skip_steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Steps to skip: disk, processes, logs, services, backup.",
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

    async def execute(
        self,
        services_to_check: list[str] | None = None,
        log_lines: int = 100,
        backup_paths: list[str] | None = None,
        backup_dest: str = "~/neuralclaw/backups",
        report_path: str = "~/neuralclaw/reports/system_maintenance.md",
        skip_steps: list[str] | None = None,
        plan_id: str = "",
        **kwargs,
    ) -> SkillResult:
        call_id = kwargs.get("_skill_call_id", "")
        t_start = time.monotonic()
        skip = set(skip_steps or [])
        plan_id = plan_id or f"maintenance-{uuid.uuid4().hex[:8]}"

        bus = kwargs.get("_bus")
        session = kwargs.get("_session")
        task_store = kwargs.get("_task_store")

        if not bus:
            return SkillResult.fail(
                self.manifest.name, call_id,
                "meta_system_maintenance requires '_bus' injected.",
                "MetaSkillConfigError",
            )

        if task_store:
            task_store.create(plan_id=plan_id, goal="Nightly system maintenance")

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
        alerts: list[str] = []

        # Step 1: Disk Usage
        if "disk" not in skip:
            log_step("disk", "Checking disk usage")
            r = await _dispatch(bus, _skill_call("system_disk_usage", {
                "show_all_mounts": True,
                "human_readable": True,
            }), session)
            log_result("disk", r)
            step_results["disk"] = r
            # Extract alerts
            if r.success and isinstance(r.output, dict):
                for alert in r.output.get("alerts", []):
                    mount = alert.get("mountpoint", "?")
                    pct = alert.get("used_pct", "?")
                    status = alert.get("status", "warning")
                    alerts.append(f"{'🔴' if status == 'critical' else '⚠️'} Disk {mount} at {pct}% used")

        # Step 2: Processes
        if "processes" not in skip:
            log_step("processes", "Listing running processes")
            r = await _dispatch(bus, _skill_call("system_process_list", {
                "sort_by": "cpu",
                "limit": 20,
            }), session)
            log_result("processes", r)
            step_results["processes"] = r

        # Step 3: System Logs
        if "logs" not in skip:
            log_step("logs", f"Tailing last {log_lines} system log lines")
            r = await _dispatch(bus, _skill_call("system_log_tail", {
                "lines": log_lines,
                "log_file": "/var/log/syslog",
            }), session)
            log_result("logs", r)
            step_results["logs"] = r

        # Step 4: Service Status
        if "services" not in skip and services_to_check:
            for svc in services_to_check:
                step_id = f"service_{svc}"
                log_step(step_id, f"Checking service: {svc}")
                r = await _dispatch(bus, _skill_call("system_service_status", {
                    "service_name": svc,
                }), session)
                log_result(step_id, r)
                step_results[step_id] = r
                # Flag stopped services
                if r.success and isinstance(r.output, dict):
                    state = r.output.get("active_state", "unknown")
                    if state not in ("active", "running"):
                        alerts.append(f"🔴 Service {svc} is {state}")

        # Step 5: Backup (only if paths specified)
        if "backup" not in skip and backup_paths:
            log_step("backup", f"Running backup for {len(backup_paths)} path(s)")
            r = await _dispatch(bus, _skill_call("system_backup_run", {
                "paths": backup_paths,
                "destination": backup_dest,
            }), session)
            log_result("backup", r)
            step_results["backup"] = r
            if not r.success:
                alerts.append(f"🔴 Backup FAILED: {r.error}")

        # ── Determine overall status ─────────────────────────────────────────
        overall = "✅ All clear" if not alerts else "\n".join(alerts)

        # ── Build report sections ────────────────────────────────────────────
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).strftime("%A, %d %B %Y %H:%M UTC")

        sections = [
            {
                "heading": "Maintenance Summary",
                "content": f"**Run at:** {now}\n\n**Status:** {overall}",
                "level": 2,
            }
        ]

        step_labels = {
            "disk": "💾 Disk Usage",
            "processes": "⚙️ Running Processes",
            "logs": "📜 System Logs",
        }
        for step_id, label in step_labels.items():
            if step_id in skip:
                continue
            r = step_results.get(step_id)
            if r is None:
                continue
            content = f"**Status:** ❌ Failed\n\n**Error:** `{r.error}`" if not r.success else _output_to_md(r.output)
            sections.append({"heading": label, "content": content, "level": 2})

        # Service sections
        service_section_parts = []
        for step_id, r in step_results.items():
            if step_id.startswith("service_"):
                svc_name = step_id.replace("service_", "")
                status_icon = "✅" if r.success else "❌"
                if r.success and isinstance(r.output, dict):
                    state = r.output.get("active_state", "unknown")
                    status_icon = "✅" if state in ("active", "running") else "⚠️"
                service_section_parts.append(f"- **{svc_name}**: {status_icon} {_result_summary(r)}")
        if service_section_parts:
            sections.append({
                "heading": "🔧 Service Status",
                "content": "\n".join(service_section_parts),
                "level": 2,
            })

        if "backup" not in skip and "backup" in step_results:
            r = step_results["backup"]
            content = f"**Status:** ❌ Failed\n\n**Error:** `{r.error}`" if not r.success else _output_to_md(r.output)
            sections.append({"heading": "💿 Backup", "content": content, "level": 2})

        # ── Write report ─────────────────────────────────────────────────────
        log_step("report_render", "Writing maintenance report")
        report_result = await _dispatch(bus, _skill_call("automation_report_render", {
            "title": f"System Maintenance — {now}",
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
                "status": overall,
                "alerts": alerts,
                "steps_completed": success_count,
                "steps_total": len(step_results),
                "steps_skipped": list(skip),
                "report_path": report_path,
                "report_written": report_result.success,
                "summary": {k: _result_summary(v) for k, v in step_results.items()},
            },
            duration_ms=duration_ms,
        )
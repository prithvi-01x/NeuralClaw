"""
skills/plugins/meta_recon_pipeline.py — Meta-Skill: Recon Pipeline

A fixed, auditable multi-step reconnaissance workflow. Orchestrates:
    cyber_dns_enum → cyber_subdomain_enum → cyber_port_scan →
    cyber_http_probe → cyber_tech_fingerprint → cyber_vuln_report_gen

The orchestrator sees this as a single plan step. All internal coordination
is invisible — sub-skill results flow through TaskMemory and are aggregated
into a final report written via automation_report_render.

Risk: CRITICAL — requires net:scan + net:fetch + fs:write capabilities.
Requires explicit user confirmation before execution.

Phase C meta-skill. No kernel changes required.

Usage:
    meta_recon_pipeline(target="example.com", report_path="~/neuralclaw/reports/recon.md")
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, ClassVar, Optional

from neuralclaw.skills.base import SkillBase
from neuralclaw.skills.types import RiskLevel, SkillManifest, SkillResult, SkillValidationError


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _truncate(text: Any, max_len: int = 2000) -> str:
    """Safely convert to string and truncate for embedding in reports."""
    s = str(text) if not isinstance(text, str) else text
    return s[:max_len] + "…" if len(s) > max_len else s


def _result_summary(result: SkillResult) -> str:
    """Return a short human-readable summary of a sub-skill result."""
    if not result.success:
        return f"❌ FAILED — {result.error}"
    output = result.output
    if isinstance(output, dict):
        # Try to extract a meaningful count or headline
        for key in ("records", "subdomains", "open_ports", "endpoints", "findings", "technologies"):
            if key in output:
                val = output[key]
                count = len(val) if isinstance(val, list) else val
                return f"✅ {count} {key} found"
        return f"✅ OK — {list(output.keys())}"
    if isinstance(output, str):
        return f"✅ {_truncate(output, 120)}"
    return "✅ OK"


def _skill_call(name: str, args: dict) -> "SkillCall":  # type: ignore[name-defined]
    """Build a SkillCall without importing orchestrator internals."""
    from neuralclaw.skills.types import SkillCall
    return SkillCall(
        id=f"meta-recon-{uuid.uuid4().hex[:8]}",
        skill_name=name,
        arguments=args,
    )


async def _dispatch(bus: Any, call: Any, session: Any) -> SkillResult:
    """Dispatch a sub-skill call through the bus with session context."""
    return await bus.dispatch(
        call,
        trust_level=session.trust_level if session else None,
        granted_capabilities=session.granted_capabilities if session else frozenset(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Meta-Skill
# ─────────────────────────────────────────────────────────────────────────────

class MetaReconPipelineSkill(SkillBase):
    manifest: ClassVar[SkillManifest] = SkillManifest(
        name="meta_recon_pipeline",
        version="1.0.0",
        description=(
            "Full reconnaissance pipeline for an authorized target domain or IP. "
            "Runs DNS enumeration → subdomain discovery → port scan → HTTP probing → "
            "technology fingerprinting → vulnerability report generation in sequence. "
            "Each step is logged to TaskMemory. Final report written to disk as Markdown. "
            "ONLY use on targets you own or have explicit written authorization to test."
        ),
        category="cybersecurity",
        risk_level=RiskLevel.CRITICAL,
        capabilities=frozenset({"net:scan", "net:fetch", "fs:write"}),
        requires_confirmation=True,
        timeout_seconds=300,
        parameters={
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Target domain name or IP address (e.g. 'example.com').",
                },
                "ports": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Port list for port scan. Defaults to common ports.",
                    "default": [],
                },
                "report_path": {
                    "type": "string",
                    "description": "Output path for the Markdown report. Default: ~/neuralclaw/reports/recon_<target>.md",
                    "default": "",
                },
                "skip_steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Step names to skip: dns_enum, subdomain_enum, port_scan, http_probe, tech_fingerprint, vuln_report.",
                    "default": [],
                },
                "plan_id": {
                    "type": "string",
                    "description": "TaskMemory plan ID for step logging (injected by orchestrator).",
                    "default": "",
                },
            },
            "required": ["target"],
        },
    )

    async def validate(self, target: str = "", **kwargs) -> None:
        if not target or not target.strip():
            raise SkillValidationError("meta_recon_pipeline: 'target' must be a non-empty domain or IP.")
        # Basic sanity — no slashes, not a URL
        if target.startswith(("http://", "https://")):
            raise SkillValidationError(
                "meta_recon_pipeline: 'target' must be a bare domain or IP, not a URL. "
                f"Strip the scheme: '{target.split('//')[-1].split('/')[0]}'"
            )

    async def execute(
        self,
        target: str,
        ports: list[int] | None = None,
        report_path: str = "",
        skip_steps: list[str] | None = None,
        plan_id: str = "",
        **kwargs,
    ) -> SkillResult:
        call_id = kwargs.get("_skill_call_id", "")
        t_start = time.monotonic()
        skip = set(skip_steps or [])
        plan_id = plan_id or f"recon-{uuid.uuid4().hex[:8]}"

        # Retrieve bus + session from kwargs (injected by the bus if available)
        bus = kwargs.get("_bus")
        session = kwargs.get("_session")
        task_store = kwargs.get("_task_store")

        if not bus:
            return SkillResult.fail(
                self.manifest.name, call_id,
                "meta_recon_pipeline requires a SkillBus injected as '_bus'. "
                "Ensure the orchestrator passes _bus to meta-skill execute() calls.",
                "MetaSkillConfigError",
            )

        # ── Setup TaskMemory ─────────────────────────────────────────────────
        if task_store:
            task_store.create(plan_id=plan_id, goal=f"Recon pipeline: {target}")

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

        # ── Collect results for report ───────────────────────────────────────
        step_results: dict[str, SkillResult] = {}

        # Step 1: DNS Enumeration
        if "dns_enum" not in skip:
            log_step("dns_enum", f"DNS enumeration for {target}")
            r = await _dispatch(bus, _skill_call("cyber_dns_enum", {"domain": target}), session)
            log_result("dns_enum", r)
            step_results["dns_enum"] = r

        # Step 2: Subdomain Enumeration
        if "subdomain_enum" not in skip:
            log_step("subdomain_enum", f"Subdomain enumeration for {target}")
            r = await _dispatch(bus, _skill_call("cyber_subdomain_enum", {"domain": target}), session)
            log_result("subdomain_enum", r)
            step_results["subdomain_enum"] = r

        # Step 3: Port Scan
        if "port_scan" not in skip:
            port_args: dict[str, Any] = {"target": target}
            if ports:
                port_args["ports"] = ports
            log_step("port_scan", f"Port scan on {target}")
            r = await _dispatch(bus, _skill_call("cyber_port_scan", port_args), session)
            log_result("port_scan", r)
            step_results["port_scan"] = r

        # Step 4: HTTP Probe
        if "http_probe" not in skip:
            log_step("http_probe", f"HTTP probe on {target}")
            r = await _dispatch(bus, _skill_call("cyber_http_probe", {"target": target}), session)
            log_result("http_probe", r)
            step_results["http_probe"] = r

        # Step 5: Technology Fingerprinting
        if "tech_fingerprint" not in skip:
            log_step("tech_fingerprint", f"Technology fingerprinting on {target}")
            r = await _dispatch(bus, _skill_call("cyber_tech_fingerprint", {"target": target}), session)
            log_result("tech_fingerprint", r)
            step_results["tech_fingerprint"] = r

        # Step 6: Vulnerability Report
        if "vuln_report" not in skip:
            log_step("vuln_report", f"Generating vulnerability report for {target}")
            # Aggregate findings for the vuln reporter
            open_ports: list = []
            pr = step_results.get("port_scan")
            if pr and pr.success and isinstance(pr.output, dict):
                open_ports = pr.output.get("open_ports", [])
            r = await _dispatch(bus, _skill_call("cyber_vuln_report_gen", {
                "target": target,
                "open_ports": open_ports,
            }), session)
            log_result("vuln_report", r)
            step_results["vuln_report"] = r

        # ── Build report sections ────────────────────────────────────────────
        sections = []
        step_labels = {
            "dns_enum": "DNS Enumeration",
            "subdomain_enum": "Subdomain Enumeration",
            "port_scan": "Port Scan",
            "http_probe": "HTTP Probe",
            "tech_fingerprint": "Technology Fingerprint",
            "vuln_report": "Vulnerability Report",
        }
        for step_id, label in step_labels.items():
            if step_id in skip:
                sections.append({"heading": label, "content": "_Skipped._", "level": 2})
                continue
            r = step_results.get(step_id)
            if r is None:
                continue
            if not r.success:
                content = f"**Status:** ❌ Failed\n\n**Error:** `{r.error}`"
            else:
                import json
                try:
                    content = f"```json\n{json.dumps(r.output, indent=2, default=str)[:4000]}\n```"
                except Exception:
                    content = str(r.output)[:4000]
            sections.append({"heading": label, "content": content, "level": 2})

        # ── Write report ─────────────────────────────────────────────────────
        report_path = report_path or f"~/neuralclaw/reports/recon_{target.replace('.', '_')}.md"
        log_step("report_render", "Writing recon report to disk")
        report_result = await _dispatch(bus, _skill_call("automation_report_render", {
            "title": f"Recon Report — {target}",
            "sections": sections,
            "output_path": report_path,
        }), session)
        log_result("report_render", report_result)

        # ── Finalize TaskMemory ──────────────────────────────────────────────
        if task_store:
            task_store.close(plan_id)

        duration_ms = (time.monotonic() - t_start) * 1000
        success_count = sum(1 for r in step_results.values() if r.success)
        total = len(step_results)

        output = {
            "target": target,
            "steps_completed": success_count,
            "steps_total": total,
            "steps_skipped": list(skip),
            "report_path": report_path,
            "report_written": report_result.success,
            "summary": {k: _result_summary(v) for k, v in step_results.items()},
        }

        return SkillResult.ok(
            skill_name=self.manifest.name,
            skill_call_id=call_id,
            output=output,
            duration_ms=duration_ms,
        )
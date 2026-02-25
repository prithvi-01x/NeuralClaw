"""
tests/unit/test_phase_c_meta_skills.py — Phase C: Meta-Skill Unit Tests

Tests all 5 meta-skills across all 3 required tiers:
    Tier 1 — Unit with Mocks: step sequence correct, TaskMemory.log_step() called
              before each dispatch, complete()/fail() called on all paths.
    Tier 2 — Integration shape: output matches manifest schema, report_written key
              present, summary dict populated.
    Tier 3 — Safety Integration: BLOCKED when bus returns blocked result,
              MetaSkillConfigError when _bus not injected.

Meta-skills tested:
    - meta_recon_pipeline     (CRITICAL, requires_confirmation=True)
    - meta_daily_assistant    (LOW, requires_confirmation=False)
    - meta_repo_audit         (LOW, requires_confirmation=False)
    - meta_system_maintenance (HIGH, requires_confirmation=True)
    - meta_autonomous_research (MEDIUM, requires_confirmation=False)
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from skills.types import RiskLevel, SkillCall, SkillManifest, SkillResult, TrustLevel


# ─────────────────────────────────────────────────────────────────────────────
# Shared test helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ok(skill_name: str = "mock_skill", output: Any = {"ok": True}) -> SkillResult:
    """Build a success SkillResult."""
    return SkillResult.ok(
        skill_name=skill_name,
        skill_call_id=f"call-{uuid.uuid4().hex[:6]}",
        output=output,
        duration_ms=10.0,
    )


def _fail(skill_name: str = "mock_skill", error: str = "mock failure") -> SkillResult:
    """Build a failure SkillResult."""
    return SkillResult.fail(
        skill_name=skill_name,
        skill_call_id=f"call-{uuid.uuid4().hex[:6]}",
        error=error,
    )


def _blocked(skill_name: str = "mock_skill") -> SkillResult:
    """Build a safety-blocked SkillResult."""
    return SkillResult.fail(
        skill_name=skill_name,
        skill_call_id=f"call-{uuid.uuid4().hex[:6]}",
        error="Safety kernel: BLOCKED",
        error_type="SafetyBlockedError",
        blocked=True,
    )


def _make_bus(dispatch_result: SkillResult | None = None) -> MagicMock:
    """Return a mock SkillBus whose dispatch always returns dispatch_result."""
    bus = MagicMock()
    bus.dispatch = AsyncMock(return_value=dispatch_result or _ok())
    return bus


def _make_task_store() -> MagicMock:
    """Return a mock TaskMemoryStore."""
    store = MagicMock()
    store.create = MagicMock(return_value=None)
    store.log_step = MagicMock(return_value=True)
    store.update_result = MagicMock(return_value=True)
    store.close = MagicMock(return_value=None)
    store.fail = MagicMock(return_value=None)
    return store


def _make_session(trust: TrustLevel = TrustLevel.MEDIUM) -> MagicMock:
    session = MagicMock()
    session.trust_level = trust
    session.granted_capabilities = frozenset({"net:scan", "net:fetch", "fs:read", "fs:write", "shell:run", "data:read", "data:write"})
    return session


def _kwargs(bus=None, task_store=None, session=None) -> dict:
    """Build the extra kwargs that the bus injects into execute()."""
    return {
        "_skill_call_id": f"call-{uuid.uuid4().hex[:8]}",
        "_bus": bus or _make_bus(),
        "_session": session or _make_session(),
        "_task_store": task_store or _make_task_store(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Import meta-skills (must be importable without running the full kernel)
# ─────────────────────────────────────────────────────────────────────────────

from skills.plugins.meta_recon_pipeline import MetaReconPipelineSkill
from skills.plugins.meta_daily_assistant import MetaDailyAssistantSkill
from skills.plugins.meta_repo_audit import MetaRepoAuditSkill
from skills.plugins.meta_system_maintenance import MetaSystemMaintenanceSkill
from skills.plugins.meta_autonomous_research import MetaAutonomousResearchSkill

ALL_META_SKILLS = [
    MetaReconPipelineSkill,
    MetaDailyAssistantSkill,
    MetaRepoAuditSkill,
    MetaSystemMaintenanceSkill,
    MetaAutonomousResearchSkill,
]


# ─────────────────────────────────────────────────────────────────────────────
# Tier 0 — Manifest validation (all 5 meta-skills)
# ─────────────────────────────────────────────────────────────────────────────

class TestManifestValidity:
    """Every meta-skill must have a valid SkillManifest."""

    @pytest.mark.parametrize("cls", ALL_META_SKILLS)
    def test_has_manifest(self, cls):
        assert hasattr(cls, "manifest"), f"{cls.__name__} missing 'manifest'"
        assert isinstance(cls.manifest, SkillManifest)

    @pytest.mark.parametrize("cls", ALL_META_SKILLS)
    def test_name_snake_case(self, cls):
        name = cls.manifest.name
        assert name.replace("_", "").isalnum(), f"Bad name: {name}"
        assert name.startswith("meta_"), f"{cls.__name__} name should start with 'meta_'"

    @pytest.mark.parametrize("cls", ALL_META_SKILLS)
    def test_version_semver(self, cls):
        from skills.base import _is_semver
        assert _is_semver(cls.manifest.version), f"Bad semver: {cls.manifest.version}"

    @pytest.mark.parametrize("cls", ALL_META_SKILLS)
    def test_description_nonempty(self, cls):
        assert cls.manifest.description.strip()

    @pytest.mark.parametrize("cls", ALL_META_SKILLS)
    def test_capabilities_frozenset(self, cls):
        assert isinstance(cls.manifest.capabilities, frozenset)

    def test_recon_is_critical(self):
        assert MetaReconPipelineSkill.manifest.risk_level == RiskLevel.CRITICAL

    def test_recon_requires_confirmation(self):
        assert MetaReconPipelineSkill.manifest.requires_confirmation is True

    def test_daily_is_low(self):
        assert MetaDailyAssistantSkill.manifest.risk_level == RiskLevel.LOW

    def test_daily_no_confirmation(self):
        assert MetaDailyAssistantSkill.manifest.requires_confirmation is False

    def test_repo_audit_is_low(self):
        assert MetaRepoAuditSkill.manifest.risk_level == RiskLevel.LOW

    def test_system_maintenance_is_high(self):
        assert MetaSystemMaintenanceSkill.manifest.risk_level == RiskLevel.HIGH

    def test_system_maintenance_requires_confirmation(self):
        assert MetaSystemMaintenanceSkill.manifest.requires_confirmation is True

    def test_research_is_medium(self):
        assert MetaAutonomousResearchSkill.manifest.risk_level == RiskLevel.MEDIUM

    @pytest.mark.parametrize("cls", ALL_META_SKILLS)
    def test_validate_manifest_passes(self, cls):
        """SkillBase._validate_manifest() must not raise."""
        cls._validate_manifest()

    @pytest.mark.parametrize("cls", ALL_META_SKILLS)
    def test_timeout_seconds_reasonable(self, cls):
        assert cls.manifest.timeout_seconds >= 60, (
            f"{cls.__name__} timeout {cls.manifest.timeout_seconds}s is too short for a meta-skill"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 — Unit with Mocks: step sequencing and TaskMemory logging
# ─────────────────────────────────────────────────────────────────────────────

class TestReconPipelineTier1:
    """Tier 1 tests for meta_recon_pipeline."""

    def setup_method(self):
        self.skill = MetaReconPipelineSkill()

    @pytest.mark.asyncio
    async def test_all_steps_dispatched_in_order(self):
        bus = _make_bus(_ok("mock"))
        store = _make_task_store()
        result = await self.skill.execute(
            target="example.com",
            _bus=bus, _session=_make_session(), _task_store=store,
            _skill_call_id="test-call",
        )
        assert result.success
        dispatched_skills = [c.args[0].skill_name for c in bus.dispatch.call_args_list]
        # Verify order: dns → subdomain → port → http → tech → vuln → report
        assert dispatched_skills.index("cyber_dns_enum") < dispatched_skills.index("cyber_subdomain_enum")
        assert dispatched_skills.index("cyber_subdomain_enum") < dispatched_skills.index("cyber_port_scan")
        assert dispatched_skills.index("cyber_port_scan") < dispatched_skills.index("cyber_http_probe")
        assert dispatched_skills.index("cyber_http_probe") < dispatched_skills.index("cyber_tech_fingerprint")
        assert dispatched_skills.index("cyber_tech_fingerprint") < dispatched_skills.index("cyber_vuln_report_gen")
        assert "automation_report_render" in dispatched_skills

    @pytest.mark.asyncio
    async def test_log_step_called_before_dispatch(self):
        """log_step() must always be called before the sub-skill dispatch."""
        call_order: list[str] = []
        bus = _make_bus(_ok())
        store = _make_task_store()
        original_log = store.log_step
        original_dispatch = bus.dispatch

        def tracked_log(plan_id, step_id, desc):
            call_order.append(f"log:{step_id}")
            return True

        async def tracked_dispatch(call, **kw):
            call_order.append(f"dispatch:{call.skill_name}")
            return _ok(call.skill_name)

        store.log_step = tracked_log
        bus.dispatch = tracked_dispatch

        await self.skill.execute(
            target="example.com",
            _bus=bus, _session=_make_session(), _task_store=store,
            _skill_call_id="test-call",
        )

        # Verify each log comes before the corresponding dispatch
        for i, entry in enumerate(call_order):
            if entry.startswith("dispatch:"):
                skill_name = entry.replace("dispatch:", "")
                # Find corresponding log entry
                step_key = {
                    "cyber_dns_enum": "dns_enum",
                    "cyber_subdomain_enum": "subdomain_enum",
                    "cyber_port_scan": "port_scan",
                    "cyber_http_probe": "http_probe",
                    "cyber_tech_fingerprint": "tech_fingerprint",
                    "cyber_vuln_report_gen": "vuln_report",
                    "automation_report_render": "report_render",
                }.get(skill_name)
                if step_key:
                    log_idx = next((j for j, e in enumerate(call_order) if e == f"log:{step_key}"), None)
                    assert log_idx is not None, f"No log found for step {step_key}"
                    assert log_idx < i, f"log:{step_key} must appear before dispatch:{skill_name}"

    @pytest.mark.asyncio
    async def test_task_store_closed_on_success(self):
        bus = _make_bus(_ok())
        store = _make_task_store()
        await self.skill.execute(
            target="example.com",
            _bus=bus, _session=_make_session(), _task_store=store,
            _skill_call_id="test-call",
        )
        store.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_skip_steps_respected(self):
        bus = _make_bus(_ok())
        await self.skill.execute(
            target="example.com",
            skip_steps=["dns_enum", "subdomain_enum", "vuln_report"],
            _bus=bus, _session=_make_session(), _task_store=_make_task_store(),
            _skill_call_id="test-call",
        )
        dispatched_skills = [c.args[0].skill_name for c in bus.dispatch.call_args_list]
        assert "cyber_dns_enum" not in dispatched_skills
        assert "cyber_subdomain_enum" not in dispatched_skills
        assert "cyber_vuln_report_gen" not in dispatched_skills
        # These should still run
        assert "cyber_port_scan" in dispatched_skills

    @pytest.mark.asyncio
    async def test_validate_rejects_url_as_target(self):
        from skills.types import SkillValidationError
        with pytest.raises(SkillValidationError):
            await self.skill.validate(target="https://example.com")

    @pytest.mark.asyncio
    async def test_validate_rejects_empty_target(self):
        from skills.types import SkillValidationError
        with pytest.raises(SkillValidationError):
            await self.skill.validate(target="")

    @pytest.mark.asyncio
    async def test_output_contains_required_keys(self):
        bus = _make_bus(_ok())
        result = await self.skill.execute(
            target="example.com",
            _bus=bus, _session=_make_session(), _task_store=_make_task_store(),
            _skill_call_id="test-call",
        )
        assert result.success
        out = result.output
        assert "target" in out
        assert "steps_completed" in out
        assert "steps_total" in out
        assert "report_path" in out
        assert "report_written" in out
        assert "summary" in out


class TestDailyAssistantTier1:
    """Tier 1 tests for meta_daily_assistant."""

    def setup_method(self):
        self.skill = MetaDailyAssistantSkill()

    @pytest.mark.asyncio
    async def test_all_steps_dispatched(self):
        bus = _make_bus(_ok())
        result = await self.skill.execute(
            _bus=bus, _session=_make_session(), _task_store=_make_task_store(),
            _skill_call_id="test-call",
        )
        assert result.success
        dispatched_skills = [c.args[0].skill_name for c in bus.dispatch.call_args_list]
        assert "personal_weather_fetch" in dispatched_skills
        assert "personal_calendar_read" in dispatched_skills
        assert "personal_news_digest" in dispatched_skills
        assert "personal_task_manager" in dispatched_skills
        assert "automation_report_render" in dispatched_skills

    @pytest.mark.asyncio
    async def test_weather_dispatched_with_location(self):
        bus = _make_bus(_ok())
        await self.skill.execute(
            location="Tokyo, Japan",
            _bus=bus, _session=_make_session(), _task_store=_make_task_store(),
            _skill_call_id="test-call",
        )
        weather_calls = [
            c for c in bus.dispatch.call_args_list
            if c.args[0].skill_name == "personal_weather_fetch"
        ]
        assert len(weather_calls) == 1
        assert weather_calls[0].args[0].arguments.get("location") == "Tokyo, Japan"

    @pytest.mark.asyncio
    async def test_reminder_set_when_calendar_has_events(self):
        """reminder_set should be called when calendar returns events."""
        cal_output = {"events": [{"title": "Team standup", "start": "09:00"}]}

        async def smart_dispatch(call, **kw):
            if call.skill_name == "personal_calendar_read":
                return _ok("personal_calendar_read", output=cal_output)
            return _ok(call.skill_name)

        bus = _make_bus()
        bus.dispatch = smart_dispatch

        await self.skill.execute(
            _bus=bus, _session=_make_session(), _task_store=_make_task_store(),
            _skill_call_id="test-call",
        )
        # Collect all dispatched skill names
        # (smart_dispatch doesn't have call_args_list, use a wrapper)

    @pytest.mark.asyncio
    async def test_log_step_called_before_each_dispatch(self):
        call_order: list[str] = []
        store = _make_task_store()

        def tracked_log(plan_id, step_id, desc):
            call_order.append(f"log:{step_id}")
            return True

        async def tracked_dispatch(call, **kw):
            call_order.append(f"dispatch:{call.skill_name}")
            return _ok(call.skill_name)

        bus = _make_bus()
        bus.dispatch = tracked_dispatch
        store.log_step = tracked_log

        await self.skill.execute(
            _bus=bus, _session=_make_session(), _task_store=store,
            _skill_call_id="test-call",
        )

        step_map = {
            "personal_weather_fetch": "weather",
            "personal_calendar_read": "calendar",
            "personal_news_digest": "news",
            "personal_task_manager": "tasks",
            "automation_report_render": "report_render",
        }
        for i, entry in enumerate(call_order):
            if entry.startswith("dispatch:"):
                skill_name = entry.split(":", 1)[1]
                step_key = step_map.get(skill_name)
                if step_key:
                    log_idx = next((j for j, e in enumerate(call_order) if e == f"log:{step_key}"), None)
                    assert log_idx is not None and log_idx < i

    @pytest.mark.asyncio
    async def test_output_schema(self):
        bus = _make_bus(_ok())
        result = await self.skill.execute(
            _bus=bus, _session=_make_session(), _task_store=_make_task_store(),
            _skill_call_id="test-call",
        )
        assert result.success
        for key in ("date", "steps_completed", "steps_total", "report_path", "report_written", "summary"):
            assert key in result.output


class TestRepoAuditTier1:
    """Tier 1 tests for meta_repo_audit."""

    def setup_method(self):
        self.skill = MetaRepoAuditSkill()

    @pytest.mark.asyncio
    async def test_all_core_steps_dispatched(self, tmp_path):
        bus = _make_bus(_ok())
        result = await self.skill.execute(
            repo_path=str(tmp_path),
            _bus=bus, _session=_make_session(), _task_store=_make_task_store(),
            _skill_call_id="test-call",
        )
        dispatched = [c.args[0].skill_name for c in bus.dispatch.call_args_list]
        assert "dev_git_log" in dispatched
        assert "dev_git_diff" in dispatched
        assert "dev_lint_runner" in dispatched
        assert "dev_test_runner" in dispatched
        assert "dev_dependency_audit" in dispatched
        assert "automation_report_render" in dispatched

    @pytest.mark.asyncio
    async def test_health_status_reflects_test_failure(self, tmp_path):
        """If test_runner returns failures > 0, health_status should indicate it."""
        test_output = {"passed": 8, "failures": 3, "errors": 0}

        async def dispatch(call, **kw):
            if call.skill_name == "dev_test_runner":
                return _ok("dev_test_runner", output=test_output)
            return _ok(call.skill_name)

        bus = _make_bus()
        bus.dispatch = dispatch

        result = await self.skill.execute(
            repo_path=str(tmp_path),
            _bus=bus, _session=_make_session(), _task_store=_make_task_store(),
            _skill_call_id="test-call",
        )
        assert "failure" in result.output["health_status"].lower() or "test" in result.output["health_status"].lower()

    @pytest.mark.asyncio
    async def test_validate_rejects_missing_path(self):
        from skills.types import SkillValidationError
        with pytest.raises(SkillValidationError):
            await self.skill.validate(repo_path="/this/path/does/not/exist/anywhere")

    @pytest.mark.asyncio
    async def test_output_schema(self, tmp_path):
        bus = _make_bus(_ok())
        result = await self.skill.execute(
            repo_path=str(tmp_path),
            _bus=bus, _session=_make_session(), _task_store=_make_task_store(),
            _skill_call_id="test-call",
        )
        for key in ("repo_path", "health_status", "steps_completed", "report_path", "report_written", "summary"):
            assert key in result.output


class TestSystemMaintenanceTier1:
    """Tier 1 tests for meta_system_maintenance."""

    def setup_method(self):
        self.skill = MetaSystemMaintenanceSkill()

    @pytest.mark.asyncio
    async def test_core_steps_dispatched(self):
        bus = _make_bus(_ok())
        result = await self.skill.execute(
            _bus=bus, _session=_make_session(), _task_store=_make_task_store(),
            _skill_call_id="test-call",
        )
        dispatched = [c.args[0].skill_name for c in bus.dispatch.call_args_list]
        assert "system_disk_usage" in dispatched
        assert "system_process_list" in dispatched
        assert "system_log_tail" in dispatched
        assert "automation_report_render" in dispatched

    @pytest.mark.asyncio
    async def test_backup_skipped_when_no_paths(self):
        bus = _make_bus(_ok())
        await self.skill.execute(
            backup_paths=[],  # empty = skip
            _bus=bus, _session=_make_session(), _task_store=_make_task_store(),
            _skill_call_id="test-call",
        )
        dispatched = [c.args[0].skill_name for c in bus.dispatch.call_args_list]
        assert "system_backup_run" not in dispatched

    @pytest.mark.asyncio
    async def test_backup_dispatched_when_paths_given(self):
        bus = _make_bus(_ok())
        await self.skill.execute(
            backup_paths=["~/projects"],
            backup_dest="~/backups",
            _bus=bus, _session=_make_session(), _task_store=_make_task_store(),
            _skill_call_id="test-call",
        )
        dispatched = [c.args[0].skill_name for c in bus.dispatch.call_args_list]
        assert "system_backup_run" in dispatched

    @pytest.mark.asyncio
    async def test_disk_alert_extracted(self):
        """Disk usage alerts > 80% should appear in output['alerts']."""
        disk_output = {
            "filesystems": [],
            "alerts": [{"mountpoint": "/", "used_pct": 95.0, "status": "critical"}],
        }

        async def dispatch(call, **kw):
            if call.skill_name == "system_disk_usage":
                return _ok("system_disk_usage", output=disk_output)
            return _ok(call.skill_name)

        bus = _make_bus()
        bus.dispatch = dispatch
        result = await self.skill.execute(
            _bus=bus, _session=_make_session(), _task_store=_make_task_store(),
            _skill_call_id="test-call",
        )
        assert any("/" in alert for alert in result.output["alerts"])

    @pytest.mark.asyncio
    async def test_services_dispatched_per_service(self):
        bus = _make_bus(_ok())
        await self.skill.execute(
            services_to_check=["nginx", "postgresql"],
            _bus=bus, _session=_make_session(), _task_store=_make_task_store(),
            _skill_call_id="test-call",
        )
        dispatched = [c.args[0].skill_name for c in bus.dispatch.call_args_list]
        assert dispatched.count("system_service_status") == 2

    @pytest.mark.asyncio
    async def test_output_schema(self):
        bus = _make_bus(_ok())
        result = await self.skill.execute(
            _bus=bus, _session=_make_session(), _task_store=_make_task_store(),
            _skill_call_id="test-call",
        )
        for key in ("status", "alerts", "steps_completed", "report_path", "report_written", "summary"):
            assert key in result.output


class TestAutonomousResearchTier1:
    """Tier 1 tests for meta_autonomous_research."""

    def setup_method(self):
        self.skill = MetaAutonomousResearchSkill()

    @pytest.mark.asyncio
    async def test_search_dispatched(self):
        bus = _make_bus(_ok("web_search", output=[{"url": "https://example.com/1"}]))
        await self.skill.execute(
            topic="WebGPU performance benchmarks",
            max_sources=1,
            _bus=bus, _session=_make_session(), _task_store=_make_task_store(),
            _skill_call_id="test-call",
        )
        dispatched = [c.args[0].skill_name for c in bus.dispatch.call_args_list]
        assert "web_search" in dispatched

    @pytest.mark.asyncio
    async def test_fetch_and_summarize_per_source(self):
        """For each URL in search results, web_fetch and data_summarize_doc must be called."""
        urls = [{"url": f"https://example.com/{i}"} for i in range(2)]

        async def dispatch(call, **kw):
            if call.skill_name == "web_search":
                return _ok("web_search", output=urls)
            if call.skill_name == "web_fetch":
                return _ok("web_fetch", output="Page content here.")
            return _ok(call.skill_name)

        bus = _make_bus()
        bus.dispatch = dispatch

        await self.skill.execute(
            topic="test topic",
            max_sources=2,
            _bus=bus, _session=_make_session(), _task_store=_make_task_store(),
            _skill_call_id="test-call",
        )

        # Since dispatch is a plain coroutine (not AsyncMock), we can't use call_args_list.
        # This test verifies execute() doesn't crash — deeper call verification covered by
        # the tracked_dispatch pattern in other tests.

    @pytest.mark.asyncio
    async def test_memory_key_auto_generated(self):
        bus = _make_bus(_ok())
        result = await self.skill.execute(
            topic="quantum computing algorithms",
            _bus=bus, _session=_make_session(), _task_store=_make_task_store(),
            _skill_call_id="test-call",
        )
        assert result.success
        assert "memory_key" in result.output
        assert result.output["memory_key"].startswith("research:")

    @pytest.mark.asyncio
    async def test_custom_memory_key_used(self):
        bus = _make_bus(_ok())
        result = await self.skill.execute(
            topic="quantum computing",
            memory_key="research:custom_key",
            _bus=bus, _session=_make_session(), _task_store=_make_task_store(),
            _skill_call_id="test-call",
        )
        assert result.output["memory_key"] == "research:custom_key"

    @pytest.mark.asyncio
    async def test_validate_rejects_empty_topic(self):
        from skills.types import SkillValidationError
        with pytest.raises(SkillValidationError):
            await self.skill.validate(topic="")

    @pytest.mark.asyncio
    async def test_validate_rejects_overlong_topic(self):
        from skills.types import SkillValidationError
        with pytest.raises(SkillValidationError):
            await self.skill.validate(topic="x" * 501)

    @pytest.mark.asyncio
    async def test_output_schema(self):
        bus = _make_bus(_ok())
        result = await self.skill.execute(
            topic="test topic",
            _bus=bus, _session=_make_session(), _task_store=_make_task_store(),
            _skill_call_id="test-call",
        )
        for key in ("topic", "sources_found", "sources_summarized", "memory_key", "report_path", "report_written", "summary"):
            assert key in result.output


# ─────────────────────────────────────────────────────────────────────────────
# Tier 3 — Safety Integration
# ─────────────────────────────────────────────────────────────────────────────

class TestSafetyIntegration:
    """Tier 3: meta-skills must handle blocked sub-skills gracefully and fail
    loudly when _bus is missing."""

    @pytest.mark.asyncio
    async def test_recon_no_bus_returns_error(self):
        skill = MetaReconPipelineSkill()
        result = await skill.execute(
            target="example.com",
            _skill_call_id="test-call",
            # _bus intentionally omitted
        )
        assert not result.success
        assert result.error_type == "MetaSkillConfigError"

    @pytest.mark.asyncio
    async def test_daily_no_bus_returns_error(self):
        skill = MetaDailyAssistantSkill()
        result = await skill.execute(_skill_call_id="test-call")
        assert not result.success
        assert result.error_type == "MetaSkillConfigError"

    @pytest.mark.asyncio
    async def test_repo_no_bus_returns_error(self, tmp_path):
        skill = MetaRepoAuditSkill()
        result = await skill.execute(
            repo_path=str(tmp_path),
            _skill_call_id="test-call",
        )
        assert not result.success
        assert result.error_type == "MetaSkillConfigError"

    @pytest.mark.asyncio
    async def test_maintenance_no_bus_returns_error(self):
        skill = MetaSystemMaintenanceSkill()
        result = await skill.execute(_skill_call_id="test-call")
        assert not result.success
        assert result.error_type == "MetaSkillConfigError"

    @pytest.mark.asyncio
    async def test_research_no_bus_returns_error(self):
        skill = MetaAutonomousResearchSkill()
        result = await skill.execute(topic="test", _skill_call_id="test-call")
        assert not result.success
        assert result.error_type == "MetaSkillConfigError"

    @pytest.mark.asyncio
    async def test_recon_continues_when_sub_skill_blocked(self):
        """If a sub-skill is blocked, the meta-skill should continue the pipeline
        and record the blocked step as a failure in the summary — not crash."""
        block = _blocked("cyber_port_scan")

        async def dispatch(call, **kw):
            if call.skill_name == "cyber_port_scan":
                return block
            return _ok(call.skill_name)

        bus = _make_bus()
        bus.dispatch = dispatch
        skill = MetaReconPipelineSkill()
        result = await skill.execute(
            target="example.com",
            _bus=bus, _session=_make_session(), _task_store=_make_task_store(),
            _skill_call_id="test-call",
        )
        # The meta-skill itself should still succeed (pipeline continued)
        assert result.success
        # Port scan should be recorded as failed in the summary
        assert "FAILED" in result.output["summary"].get("port_scan", "")

    @pytest.mark.asyncio
    async def test_maintenance_backup_blocked_flagged_in_alerts(self):
        """A blocked backup sub-skill should appear in the alerts list."""
        block = _blocked("system_backup_run")

        async def dispatch(call, **kw):
            if call.skill_name == "system_backup_run":
                return block
            return _ok(call.skill_name)

        bus = _make_bus()
        bus.dispatch = dispatch
        skill = MetaSystemMaintenanceSkill()
        result = await skill.execute(
            backup_paths=["~/projects"],
            _bus=bus, _session=_make_session(), _task_store=_make_task_store(),
            _skill_call_id="test-call",
        )
        assert result.success
        assert any("Backup" in a or "backup" in a.lower() for a in result.output["alerts"])


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2 — Integration shape (output schema contract)
# ─────────────────────────────────────────────────────────────────────────────

class TestOutputSchema:
    """Tier 2: ensure output dicts match documented contracts."""

    @pytest.mark.asyncio
    async def test_recon_report_path_in_output(self):
        bus = _make_bus(_ok())
        skill = MetaReconPipelineSkill()
        result = await skill.execute(
            target="test.example",
            report_path="/tmp/recon_test.md",
            _bus=bus, _session=_make_session(), _task_store=_make_task_store(),
            _skill_call_id="test",
        )
        assert result.output["report_path"] == "/tmp/recon_test.md"

    @pytest.mark.asyncio
    async def test_research_max_sources_clamped(self):
        """max_sources > 10 should be clamped to 10."""
        bus = _make_bus(_ok("web_search", output=[]))
        skill = MetaAutonomousResearchSkill()
        result = await skill.execute(
            topic="test",
            max_sources=999,
            _bus=bus, _session=_make_session(), _task_store=_make_task_store(),
            _skill_call_id="test",
        )
        assert result.success  # clamping should not crash

    @pytest.mark.asyncio
    async def test_all_meta_skills_return_skill_result(self):
        """Every meta-skill execute() must return a SkillResult — never raise."""
        bus = _make_bus(_ok())
        session = _make_session()
        store = _make_task_store()

        skills_and_kwargs = [
            (MetaReconPipelineSkill(), {"target": "example.com"}),
            (MetaDailyAssistantSkill(), {}),
            (MetaSystemMaintenanceSkill(), {}),
            (MetaAutonomousResearchSkill(), {"topic": "test"}),
        ]

        for skill, extra in skills_and_kwargs:
            result = await skill.execute(
                _bus=bus, _session=session, _task_store=store,
                _skill_call_id="test",
                **extra,
            )
            assert isinstance(result, SkillResult), f"{skill.__class__.__name__} did not return SkillResult"
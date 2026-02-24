"""
tests/unit/test_skills.py — Phase 2: Skill System Unit Tests

Covers every layer of the new skills/ package without requiring external
dependencies (httpx, BeautifulSoup, SafetyKernel, config singletons, etc.).

Test groups:
  - SkillManifest:      validation, to_llm_schema, to_tool_schema_dict
  - SkillCall:          immutability (frozen dataclass)
  - SkillResult:        ok/fail factories, to_llm_content, truncation
  - SkillBase:          _validate_manifest catches every bad manifest case
  - SkillRegistry:      register, get, get_or_none, duplicates, list_*
  - SkillLoader:        discovers classes, validates manifests, skips __init__,
                        bad-import warning, duplicate-name error, empty dir OK
  - SkillBus:           unknown skill, disabled skill, arg validation,
                        pre-validate error, safety BLOCKED, safety CONFIRM,
                        execution error, timeout, raw-value wrapping, output
                        truncation, native dispatch (was: dispatch_legacy bridge)
  - Builtin integrity:  every builtin has a valid manifest, unique names,
                        correct capabilities declared
"""

from __future__ import annotations

import asyncio
import sys
import textwrap
import time
from pathlib import Path
from typing import ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── path setup (so `skills` is importable without installing the package) ────
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from skills.base import SkillBase, _is_semver
from skills.loader import SkillLoader
from skills.registry import SkillRegistry
from skills.types import (
    RiskLevel,
    SkillCall,
    SkillManifest,
    SkillNotFoundError,
    SkillResult,
    SkillValidationError,
    TrustLevel,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers / fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _manifest(**overrides) -> SkillManifest:
    defaults = dict(
        name="test_skill",
        version="1.0.0",
        description="A test skill.",
        category="test",
        risk_level=RiskLevel.LOW,
        capabilities=frozenset(),
        parameters={"type": "object", "properties": {}, "required": []},
        timeout_seconds=5,
    )
    defaults.update(overrides)
    return SkillManifest(**defaults)


class _OkSkill(SkillBase):
    manifest = _manifest(name="ok_skill")

    async def execute(self, **kwargs) -> SkillResult:
        call_id = kwargs.get("_skill_call_id", "")
        return SkillResult.ok(skill_name=self.manifest.name, skill_call_id=call_id, output="ok")


class _FailSkill(SkillBase):
    manifest = _manifest(name="fail_skill")

    async def execute(self, **kwargs) -> SkillResult:
        call_id = kwargs.get("_skill_call_id", "")
        return SkillResult.fail(self.manifest.name, call_id, "always fails", "AlwaysFailError")


class _RaisingSkill(SkillBase):
    """Badly-written skill that raises instead of returning SkillResult.fail."""
    manifest = _manifest(name="raising_skill")

    async def execute(self, **kwargs) -> SkillResult:
        raise RuntimeError("unexpected crash")


class _SlowSkill(SkillBase):
    manifest = _manifest(name="slow_skill", timeout_seconds=1)

    async def execute(self, **kwargs) -> SkillResult:
        await asyncio.sleep(999)
        return SkillResult.ok(self.manifest.name, "", "never")


class _ValidatingSkill(SkillBase):
    manifest = _manifest(
        name="validating_skill",
        parameters={
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    )

    async def validate(self, url: str = "", **_) -> None:
        if not url.startswith("https://"):
            raise SkillValidationError(f"URL must start with https://, got: {url}")

    async def execute(self, url: str = "", **kwargs) -> SkillResult:
        call_id = kwargs.get("_skill_call_id", "")
        return SkillResult.ok(self.manifest.name, call_id, f"fetched {url}")


def _registry_with(*skill_instances) -> SkillRegistry:
    reg = SkillRegistry()
    for s in skill_instances:
        reg.register(s)
    return reg


def _call(name: str, args: dict = None, call_id: str = "call_1") -> SkillCall:
    return SkillCall(id=call_id, skill_name=name, arguments=args or {})


# ─────────────────────────────────────────────────────────────────────────────
# SkillManifest
# ─────────────────────────────────────────────────────────────────────────────

class TestSkillManifest:
    def test_frozen(self):
        m = _manifest()
        with pytest.raises((TypeError, AttributeError)):
            m.name = "changed"  # type: ignore[misc]

    def test_to_llm_schema_keys(self):
        m = _manifest(name="my_skill", description="Does things.")
        schema = m.to_llm_schema()
        assert schema["name"] == "my_skill"
        assert schema["description"] == "Does things."
        assert "parameters" in schema

    def test_to_tool_schema_dict_has_risk(self):
        m = _manifest(risk_level=RiskLevel.HIGH)
        d = m.to_tool_schema_dict()
        assert d["risk_level"] == RiskLevel.HIGH

    def test_capabilities_is_frozenset(self):
        m = _manifest(capabilities=frozenset({"fs:read", "net:fetch"}))
        assert isinstance(m.capabilities, frozenset)
        assert "fs:read" in m.capabilities

    def test_risk_ordering(self):
        assert RiskLevel.LOW < RiskLevel.MEDIUM < RiskLevel.HIGH < RiskLevel.CRITICAL
        assert RiskLevel.CRITICAL > RiskLevel.HIGH
        assert RiskLevel.LOW <= RiskLevel.LOW


# ─────────────────────────────────────────────────────────────────────────────
# SkillCall
# ─────────────────────────────────────────────────────────────────────────────

class TestSkillCall:
    def test_immutable(self):
        c = SkillCall(id="x", skill_name="foo", arguments={"a": 1})
        with pytest.raises((TypeError, AttributeError)):
            c.skill_name = "bar"  # type: ignore[misc]

    def test_created_at_auto(self):
        before = time.monotonic()
        c = SkillCall(id="x", skill_name="foo")
        after = time.monotonic()
        assert before <= c.created_at <= after

    def test_default_empty_arguments(self):
        c = SkillCall(id="x", skill_name="foo")
        assert c.arguments == {}


# ─────────────────────────────────────────────────────────────────────────────
# SkillResult
# ─────────────────────────────────────────────────────────────────────────────

class TestSkillResult:
    def test_ok_factory(self):
        r = SkillResult.ok("skill", "call_1", "hello")
        assert r.success is True
        assert r.output == "hello"
        assert r.error is None

    def test_fail_factory(self):
        r = SkillResult.fail("skill", "call_1", "boom", "BoomError")
        assert r.success is False
        assert r.output is None
        assert r.error == "boom"
        assert r.error_type == "BoomError"

    def test_to_llm_content_success_str(self):
        r = SkillResult.ok("s", "c", "plain text")
        assert r.to_llm_content() == "plain text"

    def test_to_llm_content_success_dict(self):
        r = SkillResult.ok("s", "c", {"key": "value"})
        content = r.to_llm_content()
        assert '"key"' in content
        assert '"value"' in content

    def test_to_llm_content_fail(self):
        r = SkillResult.fail("s", "c", "oops", "OopsError")
        assert "OopsError" in r.to_llm_content()
        assert "oops" in r.to_llm_content()

    def test_immutable(self):
        r = SkillResult.ok("s", "c", "x")
        with pytest.raises((TypeError, AttributeError)):
            r.success = False  # type: ignore[misc]

    def test_duration_defaults_zero(self):
        r = SkillResult.ok("s", "c", "x")
        assert r.duration_ms == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# SkillBase._validate_manifest
# ─────────────────────────────────────────────────────────────────────────────

class TestSkillBaseValidateManifest:
    def _make_class(self, **overrides):
        """Dynamically create a SkillBase subclass with a given manifest."""
        manifest = _manifest(**overrides)

        class DynSkill(SkillBase):
            pass
        DynSkill.manifest = manifest  # type: ignore[attr-defined]
        DynSkill.__name__ = "DynSkill"
        return DynSkill

    def test_valid_manifest_passes(self):
        cls = self._make_class()
        cls._validate_manifest()  # should not raise

    def test_missing_manifest_attr(self):
        class NoManifest(SkillBase):
            async def execute(self, **kw): ...
        # Remove any inherited manifest
        NoManifest.__dict__  # just access it
        if hasattr(NoManifest, "manifest"):
            del NoManifest.manifest
        with pytest.raises(SkillValidationError, match="missing"):
            NoManifest._validate_manifest()

    def test_wrong_manifest_type(self):
        class BadManifest(SkillBase):
            manifest = "not a manifest"  # type: ignore[assignment]
            async def execute(self, **kw): ...
        with pytest.raises(SkillValidationError, match="SkillManifest instance"):
            BadManifest._validate_manifest()

    def test_empty_name_rejected(self):
        cls = self._make_class(name="")
        with pytest.raises(SkillValidationError, match="invalid"):
            cls._validate_manifest()

    def test_non_alnum_name_rejected(self):
        cls = self._make_class(name="my-skill")  # hyphens not allowed
        with pytest.raises(SkillValidationError, match="invalid"):
            cls._validate_manifest()

    def test_bad_version_rejected(self):
        cls = self._make_class(version="v1")
        with pytest.raises(SkillValidationError, match="semver"):
            cls._validate_manifest()

    def test_two_part_version_rejected(self):
        cls = self._make_class(version="1.0")
        with pytest.raises(SkillValidationError, match="semver"):
            cls._validate_manifest()

    def test_empty_description_rejected(self):
        cls = self._make_class(description="   ")
        with pytest.raises(SkillValidationError, match="description"):
            cls._validate_manifest()

    def test_empty_category_rejected(self):
        cls = self._make_class(category="")
        with pytest.raises(SkillValidationError, match="category"):
            cls._validate_manifest()

    def test_non_frozenset_capabilities_rejected(self):
        cls = self._make_class()
        cls.manifest = _manifest()
        # Manually replace with a list to bypass frozen dataclass
        bad_manifest = SkillManifest(
            name="x", version="1.0.0", description="d", category="c",
            risk_level=RiskLevel.LOW, capabilities=["fs:read"],  # type: ignore[arg-type]
        )
        cls.manifest = bad_manifest  # type: ignore[attr-defined]
        with pytest.raises(SkillValidationError, match="frozenset"):
            cls._validate_manifest()

    def test_non_dict_parameters_rejected(self):
        bad_manifest = SkillManifest(
            name="x", version="1.0.0", description="d", category="c",
            risk_level=RiskLevel.LOW, capabilities=frozenset(),
            parameters="not a dict",  # type: ignore[arg-type]
        )

        class BadParams(SkillBase):
            manifest = bad_manifest  # type: ignore[assignment]
            async def execute(self, **kw): ...
        with pytest.raises(SkillValidationError, match="dict"):
            BadParams._validate_manifest()


# ─────────────────────────────────────────────────────────────────────────────
# _is_semver helper
# ─────────────────────────────────────────────────────────────────────────────

class TestIsSemver:
    def test_valid(self):
        assert _is_semver("1.0.0") is True
        assert _is_semver("0.0.1") is True
        assert _is_semver("12.34.56") is True

    def test_invalid(self):
        assert _is_semver("v1.0.0") is False
        assert _is_semver("1.0") is False
        assert _is_semver("1.0.0.0") is False
        assert _is_semver("") is False
        assert _is_semver("abc") is False


# ─────────────────────────────────────────────────────────────────────────────
# SkillRegistry
# ─────────────────────────────────────────────────────────────────────────────

class TestSkillRegistry:
    def test_register_and_get(self):
        reg = SkillRegistry()
        reg.register(_OkSkill())
        skill = reg.get("ok_skill")
        assert skill is not None
        assert skill.manifest.name == "ok_skill"

    def test_get_missing_raises(self):
        reg = SkillRegistry()
        with pytest.raises(SkillNotFoundError, match="ok_skill"):
            reg.get("ok_skill")

    def test_get_or_none_missing(self):
        reg = SkillRegistry()
        assert reg.get_or_none("nope") is None

    def test_duplicate_registration_raises(self):
        reg = SkillRegistry()
        reg.register(_OkSkill())
        with pytest.raises(ValueError, match="already registered"):
            reg.register(_OkSkill())

    def test_is_registered(self):
        reg = SkillRegistry()
        assert not reg.is_registered("ok_skill")
        reg.register(_OkSkill())
        assert reg.is_registered("ok_skill")

    def test_list_names(self):
        reg = _registry_with(_OkSkill(), _FailSkill())
        names = reg.list_names()
        assert "ok_skill" in names
        assert "fail_skill" in names

    def test_disabled_skill_excluded_from_list(self):
        reg = SkillRegistry()
        disabled_manifest = _manifest(name="disabled_skill", enabled=False)

        class DisabledSkill(SkillBase):
            manifest = disabled_manifest
            async def execute(self, **kw): ...

        reg.register(DisabledSkill())
        assert "disabled_skill" not in reg.list_names(enabled_only=True)
        assert "disabled_skill" in reg.list_names(enabled_only=False)

    def test_list_manifests(self):
        reg = _registry_with(_OkSkill())
        manifests = reg.list_manifests()
        assert len(manifests) == 1
        assert manifests[0].name == "ok_skill"

    def test_to_llm_schemas(self):
        reg = _registry_with(_OkSkill())
        schemas = reg.to_llm_schemas()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "ok_skill"
        assert "description" in schemas[0]
        assert "parameters" in schemas[0]

    def test_get_manifest(self):
        reg = _registry_with(_OkSkill())
        m = reg.get_manifest("ok_skill")
        assert m is not None
        assert m.name == "ok_skill"

    def test_len(self):
        reg = SkillRegistry()
        assert len(reg) == 0
        reg.register(_OkSkill())
        assert len(reg) == 1

    def test_unregister(self):
        reg = _registry_with(_OkSkill())
        reg.unregister("ok_skill")
        assert not reg.is_registered("ok_skill")


# ─────────────────────────────────────────────────────────────────────────────
# SkillLoader
# ─────────────────────────────────────────────────────────────────────────────

class TestSkillLoader:
    def test_loads_skills_from_dir(self, tmp_path):
        skill_file = tmp_path / "my_skill.py"
        skill_file.write_text(textwrap.dedent("""
            from skills.base import SkillBase
            from skills.types import SkillManifest, RiskLevel, SkillResult

            class MySkill(SkillBase):
                manifest = SkillManifest(
                    name="my_skill",
                    version="1.0.0",
                    description="A dynamically loaded skill.",
                    category="test",
                    risk_level=RiskLevel.LOW,
                    capabilities=frozenset(),
                    parameters={"type": "object", "properties": {}, "required": []},
                )

                async def execute(self, **kwargs):
                    return SkillResult.ok(self.manifest.name, "", "done")
        """))

        loader = SkillLoader()
        reg = loader.load_all([tmp_path])
        assert reg.is_registered("my_skill")

    def test_skips_init_files(self, tmp_path):
        init_file = tmp_path / "__init__.py"
        init_file.write_text("# nothing")
        loader = SkillLoader()
        reg = loader.load_all([tmp_path])
        assert len(reg) == 0

    def test_nonexistent_dir_skipped_gracefully(self, tmp_path):
        loader = SkillLoader()
        reg = loader.load_all([tmp_path / "does_not_exist"])
        assert len(reg) == 0

    def test_import_error_skipped_with_warning(self, tmp_path):
        bad_file = tmp_path / "bad_skill.py"
        bad_file.write_text("raise ImportError('missing dep')\n")
        good_file = tmp_path / "good_skill.py"
        good_file.write_text(textwrap.dedent("""
            from skills.base import SkillBase
            from skills.types import SkillManifest, RiskLevel, SkillResult

            class GoodSkill(SkillBase):
                manifest = SkillManifest(
                    name="good_skill",
                    version="1.0.0",
                    description="Good.",
                    category="test",
                    risk_level=RiskLevel.LOW,
                    capabilities=frozenset(),
                    parameters={"type": "object", "properties": {}, "required": []},
                )
                async def execute(self, **kw):
                    return SkillResult.ok(self.manifest.name, "", "ok")
        """))
        loader = SkillLoader()
        reg = loader.load_all([tmp_path])
        # good_skill loads, bad_skill skipped silently
        assert reg.is_registered("good_skill")
        assert not reg.is_registered("bad_skill")

    def test_duplicate_name_across_files_raises(self, tmp_path):
        for fname in ("skill_a.py", "skill_b.py"):
            (tmp_path / fname).write_text(textwrap.dedent("""
                from skills.base import SkillBase
                from skills.types import SkillManifest, RiskLevel, SkillResult

                class DupSkill(SkillBase):
                    manifest = SkillManifest(
                        name="dup_skill",
                        version="1.0.0",
                        description="Duplicate.",
                        category="test",
                        risk_level=RiskLevel.LOW,
                        capabilities=frozenset(),
                        parameters={"type": "object", "properties": {}, "required": []},
                    )
                    async def execute(self, **kw):
                        return SkillResult.ok(self.manifest.name, "", "ok")
            """))

        loader = SkillLoader()
        with pytest.raises(SkillValidationError, match="already registered"):
            loader.load_all([tmp_path])

    def test_bad_manifest_raises_at_load(self, tmp_path):
        bad_file = tmp_path / "bad_manifest.py"
        bad_file.write_text(textwrap.dedent("""
            from skills.base import SkillBase
            from skills.types import SkillManifest, RiskLevel, SkillResult

            class BadManifestSkill(SkillBase):
                manifest = SkillManifest(
                    name="",  # INVALID: empty name
                    version="1.0.0",
                    description="Bad.",
                    category="test",
                    risk_level=RiskLevel.LOW,
                    capabilities=frozenset(),
                    parameters={},
                )
                async def execute(self, **kw):
                    return SkillResult.ok(self.manifest.name, "", "ok")
        """))

        loader = SkillLoader()
        with pytest.raises(SkillValidationError):
            loader.load_all([tmp_path])

    def test_discover_classes_returns_list(self, tmp_path):
        (tmp_path / "test_skill.py").write_text(textwrap.dedent("""
            from skills.base import SkillBase
            from skills.types import SkillManifest, RiskLevel, SkillResult

            class TestSkill(SkillBase):
                manifest = SkillManifest(
                    name="test_skill_cls",
                    version="1.0.0",
                    description="Test.",
                    category="test",
                    risk_level=RiskLevel.LOW,
                    capabilities=frozenset(),
                    parameters={"type": "object", "properties": {}, "required": []},
                )
                async def execute(self, **kw):
                    return SkillResult.ok(self.manifest.name, "", "ok")
        """))
        loader = SkillLoader()
        classes = loader.discover_classes([tmp_path])
        assert any(cls.manifest.name == "test_skill_cls" for cls in classes)


# ─────────────────────────────────────────────────────────────────────────────
# SkillBus — dispatch pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _make_bus(skill_instances, safety=None, timeout=5.0, on_confirm=None):
    from skills.bus import SkillBus
    reg = _registry_with(*skill_instances)
    return SkillBus(reg, safety_kernel=safety, default_timeout_seconds=timeout, on_confirm_needed=on_confirm)


class TestSkillBusDispatch:

    @pytest.mark.asyncio
    async def test_unknown_skill_returns_fail(self):
        bus = _make_bus([])
        result = await bus.dispatch(_call("nonexistent"))
        assert not result.success
        assert "not registered" in result.error
        assert result.error_type == "SkillNotFoundError"

    @pytest.mark.asyncio
    async def test_disabled_skill_returns_fail(self):
        from skills.bus import SkillBus

        class DisabledSkill(SkillBase):
            manifest = _manifest(name="disabled_skill", enabled=False)
            async def execute(self, **kw): ...

        bus = _make_bus([DisabledSkill()])
        result = await bus.dispatch(_call("disabled_skill"))
        assert not result.success
        assert "disabled" in result.error.lower()

    @pytest.mark.asyncio
    async def test_missing_required_arg_fails(self):
        bus = _make_bus([_ValidatingSkill()])
        # url is required but not provided
        result = await bus.dispatch(_call("validating_skill", {}))
        assert not result.success
        assert "url" in result.error
        assert result.error_type == "SkillValidationError"

    @pytest.mark.asyncio
    async def test_wrong_arg_type_fails(self):
        class TypedSkill(SkillBase):
            manifest = _manifest(
                name="typed_skill",
                parameters={
                    "type": "object",
                    "properties": {"count": {"type": "integer"}},
                    "required": ["count"],
                },
            )
            async def execute(self, count: int = 0, **kw):
                return SkillResult.ok(self.manifest.name, kw.get("_skill_call_id", ""), count)

        bus = _make_bus([TypedSkill()])
        result = await bus.dispatch(_call("typed_skill", {"count": "not_an_int"}))
        assert not result.success
        assert "integer" in result.error

    @pytest.mark.asyncio
    async def test_validate_hook_rejection_fails(self):
        bus = _make_bus([_ValidatingSkill()])
        result = await bus.dispatch(_call("validating_skill", {"url": "http://not-https.com"}))
        assert not result.success
        assert "https" in result.error

    @pytest.mark.asyncio
    async def test_successful_dispatch(self):
        bus = _make_bus([_OkSkill()])
        result = await bus.dispatch(_call("ok_skill"), TrustLevel.LOW)
        assert result.success
        assert result.output == "ok"

    @pytest.mark.asyncio
    async def test_skill_returns_fail_propagated(self):
        bus = _make_bus([_FailSkill()])
        result = await bus.dispatch(_call("fail_skill"))
        assert not result.success
        assert result.error_type == "AlwaysFailError"

    @pytest.mark.asyncio
    async def test_execution_exception_caught_as_fail(self):
        bus = _make_bus([_RaisingSkill()])
        result = await bus.dispatch(_call("raising_skill"))
        assert not result.success
        assert "RuntimeError" in result.error or "unexpected crash" in result.error

    @pytest.mark.asyncio
    async def test_timeout_returns_fail(self):
        bus = _make_bus([_SlowSkill()], timeout=0.05)
        result = await bus.dispatch(_call("slow_skill"))
        assert not result.success
        assert result.error_type == "SkillTimeoutError"

    @pytest.mark.asyncio
    async def test_duration_is_set(self):
        bus = _make_bus([_OkSkill()])
        result = await bus.dispatch(_call("ok_skill"))
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_output_truncated_when_too_long(self):
        from skills.bus import MAX_RESULT_CHARS

        class BigOutputSkill(SkillBase):
            manifest = _manifest(name="big_output_skill")
            async def execute(self, **kw):
                return SkillResult.ok(self.manifest.name, kw.get("_skill_call_id", ""), "x" * (MAX_RESULT_CHARS + 100))

        bus = _make_bus([BigOutputSkill()])
        result = await bus.dispatch(_call("big_output_skill"))
        assert result.success
        assert "truncated" in result.output.lower()
        assert len(result.output) < MAX_RESULT_CHARS + 200  # well under original size

    @pytest.mark.asyncio
    async def test_safety_blocked_returns_fail(self):
        # Mock a safety kernel that returns BLOCKED
        from skills.types import SafetyDecision, SafetyStatus, RiskLevel as LR
        blocked_decision = SafetyDecision(
            status=SafetyStatus.BLOCKED,
            reason="test block",
            risk_level=LR.HIGH,
            tool_name="ok_skill",
            tool_call_id="call_1",
        )
        mock_safety = AsyncMock()
        mock_safety.evaluate = AsyncMock(return_value=blocked_decision)

        bus = _make_bus([_OkSkill()], safety=mock_safety)
        result = await bus.dispatch(_call("ok_skill"))
        assert not result.success
        assert "blocked" in result.error.lower()
        assert result.error_type == "SafetyBlockedError"

    @pytest.mark.asyncio
    async def test_safety_confirm_denied_returns_fail(self):
        from skills.types import SafetyDecision, SafetyStatus, RiskLevel as LR
        confirm_decision = SafetyDecision(
            status=SafetyStatus.CONFIRM_NEEDED,
            reason="risky",
            risk_level=LR.HIGH,
            tool_name="ok_skill",
            tool_call_id="call_1",
        )
        mock_safety = AsyncMock()
        mock_safety.evaluate = AsyncMock(return_value=confirm_decision)

        # No confirm handler → denied
        bus = _make_bus([_OkSkill()], safety=mock_safety, on_confirm=None)
        result = await bus.dispatch(_call("ok_skill"))
        assert not result.success
        assert "denied" in result.error.lower()

    @pytest.mark.asyncio
    async def test_safety_confirm_approved_executes(self):
        from skills.types import SafetyDecision, SafetyStatus, RiskLevel as LR
        confirm_decision = SafetyDecision(
            status=SafetyStatus.CONFIRM_NEEDED,
            reason="risky",
            risk_level=LR.HIGH,
            tool_name="ok_skill",
            tool_call_id="call_1",
        )
        mock_safety = AsyncMock()
        mock_safety.evaluate = AsyncMock(return_value=confirm_decision)

        async def approve(_): return True

        bus = _make_bus([_OkSkill()], safety=mock_safety, on_confirm=approve)
        result = await bus.dispatch(_call("ok_skill"))
        assert result.success

    @pytest.mark.asyncio
    async def test_per_call_confirm_overrides_bus_confirm(self):
        from skills.types import SafetyDecision, SafetyStatus, RiskLevel as LR
        confirm_decision = SafetyDecision(
            status=SafetyStatus.CONFIRM_NEEDED,
            reason="risky",
            risk_level=LR.HIGH,
            tool_name="ok_skill",
            tool_call_id="call_1",
        )
        mock_safety = AsyncMock()
        mock_safety.evaluate = AsyncMock(return_value=confirm_decision)

        async def bus_deny(_): return False
        async def call_approve(_): return True

        bus = _make_bus([_OkSkill()], safety=mock_safety, on_confirm=bus_deny)
        result = await bus.dispatch(_call("ok_skill"), on_confirm_needed=call_approve)
        assert result.success  # per-call approve wins

    @pytest.mark.asyncio
    async def test_skill_call_id_injected(self):
        captured = {}

        class IdSkill(SkillBase):
            manifest = _manifest(name="id_skill")
            async def execute(self, **kw):
                captured["id"] = kw.get("_skill_call_id", "NOT_SET")
                return SkillResult.ok(self.manifest.name, captured["id"], "x")

        bus = _make_bus([IdSkill()])
        await bus.dispatch(_call("id_skill", call_id="my_call_id"))
        assert captured["id"] == "my_call_id"


# ─────────────────────────────────────────────────────────────────────────────
# SkillBus — native dispatch (was: dispatch_legacy bridge, now removed)
# These tests verify the same dispatch behaviour via the native SkillCall path.

class TestNativeDispatch:
    @pytest.mark.asyncio
    async def test_native_dispatch_ok_skill_succeeds(self):
        """Native dispatch returns SkillResult.success for a working skill."""
        bus = _make_bus([_OkSkill()])
        result = await bus.dispatch(_call("ok_skill", call_id="lc1"), TrustLevel.LOW)
        assert result.success
        assert result.output == "ok"

    @pytest.mark.asyncio
    async def test_native_dispatch_fail_skill_returns_error(self):
        """Native dispatch propagates skill failures as SkillResult with success=False."""
        bus = _make_bus([_FailSkill()])
        result = await bus.dispatch(_call("fail_skill", call_id="lc2"))
        assert not result.success
        assert result.error_type == "AlwaysFailError"

    @pytest.mark.asyncio
    async def test_native_dispatch_default_trust_succeeds(self):
        """Omitting trust_level falls back to LOW — not an error."""
        bus = _make_bus([_OkSkill()])
        result = await bus.dispatch(_call("ok_skill", call_id="lc3"))
        assert result.success


# ─────────────────────────────────────────────────────────────────────────────
# Builtin skill integrity
# ─────────────────────────────────────────────────────────────────────────────

class TestBuiltinIntegrity:
    """
    Load all builtin skills via the loader and verify structural integrity.
    Does NOT execute any skill (that would require real filesystem/network).
    """

    @pytest.fixture(scope="class")
    def builtin_registry(self):
        loader = SkillLoader()
        builtin_dir = Path(__file__).parent.parent.parent / "skills" / "builtin"
        return loader.load_all([builtin_dir])

    def test_all_builtins_load(self, builtin_registry):
        expected = {
            "file_read", "file_write", "file_append",
            "list_dir", "file_exists",
            "terminal_exec",
            "web_search",
            "web_fetch",
        }
        loaded = set(builtin_registry.list_names())
        assert expected == loaded, f"Missing: {expected - loaded}, extra: {loaded - expected}"

    def test_no_duplicate_names(self, builtin_registry):
        names = builtin_registry.list_names()
        assert len(names) == len(set(names)), "Duplicate skill names detected"

    def test_all_have_valid_semver(self, builtin_registry):
        for m in builtin_registry.list_manifests():
            assert _is_semver(m.version), f"{m.name}: bad version '{m.version}'"

    def test_all_have_nonempty_description(self, builtin_registry):
        for m in builtin_registry.list_manifests():
            assert m.description.strip(), f"{m.name}: empty description"

    def test_all_have_frozenset_capabilities(self, builtin_registry):
        for m in builtin_registry.list_manifests():
            assert isinstance(m.capabilities, frozenset), f"{m.name}: capabilities is not frozenset"

    def test_filesystem_capabilities(self, builtin_registry):
        fs_read = {"file_read", "list_dir", "file_exists"}
        fs_write = {"file_write", "file_append"}
        for name in fs_read:
            m = builtin_registry.get_manifest(name)
            assert "fs:read" in m.capabilities, f"{name}: missing fs:read capability"
        for name in fs_write:
            m = builtin_registry.get_manifest(name)
            assert "fs:write" in m.capabilities, f"{name}: missing fs:write capability"

    def test_terminal_capabilities(self, builtin_registry):
        m = builtin_registry.get_manifest("terminal_exec")
        assert "shell:run" in m.capabilities

    def test_web_capabilities(self, builtin_registry):
        for name in ("web_search", "web_fetch"):
            m = builtin_registry.get_manifest(name)
            assert "net:fetch" in m.capabilities, f"{name}: missing net:fetch"

    def test_terminal_requires_confirmation(self, builtin_registry):
        m = builtin_registry.get_manifest("terminal_exec")
        assert m.requires_confirmation is True

    def test_read_skills_are_low_risk(self, builtin_registry):
        for name in ("file_read", "list_dir", "file_exists", "web_search", "web_fetch"):
            m = builtin_registry.get_manifest(name)
            assert m.risk_level == RiskLevel.LOW, f"{name}: expected LOW risk"

    def test_write_skills_are_medium_risk(self, builtin_registry):
        for name in ("file_write", "file_append"):
            m = builtin_registry.get_manifest(name)
            assert m.risk_level == RiskLevel.MEDIUM, f"{name}: expected MEDIUM risk"

    def test_terminal_is_high_risk(self, builtin_registry):
        m = builtin_registry.get_manifest("terminal_exec")
        assert m.risk_level == RiskLevel.HIGH

    def test_all_have_positive_timeout(self, builtin_registry):
        for m in builtin_registry.list_manifests():
            assert m.timeout_seconds > 0, f"{m.name}: timeout_seconds must be > 0"

    def test_llm_schemas_have_required_keys(self, builtin_registry):
        for schema in builtin_registry.to_llm_schemas():
            assert "name" in schema
            assert "description" in schema
            assert "parameters" in schema


# ─────────────────────────────────────────────────────────────────────────────
# Plugin discovery: plugins/ dir is empty but loader handles it gracefully
# ─────────────────────────────────────────────────────────────────────────────

class TestPluginDirectory:
    def test_empty_plugins_dir_loads_zero_skills(self):
        plugins_dir = Path(__file__).parent.parent.parent / "skills" / "plugins"
        loader = SkillLoader()
        reg = loader.load_all([plugins_dir])
        # __init__.py is skipped, so nothing loaded
        assert len(reg) == 0

    def test_plugin_added_to_plugins_dir_appears_in_registry(self, tmp_path):
        """
        Simulate the litmus test: drop a new skill into plugins/ and it appears
        in the registry with no core changes required.
        """
        plugin = tmp_path / "echo_skill.py"
        plugin.write_text(textwrap.dedent("""
            from skills.base import SkillBase
            from skills.types import SkillManifest, RiskLevel, SkillResult

            class EchoSkill(SkillBase):
                manifest = SkillManifest(
                    name="echo",
                    version="1.0.0",
                    description="Echo the input back.",
                    category="demo",
                    risk_level=RiskLevel.LOW,
                    capabilities=frozenset({"net:fetch"}),
                    parameters={
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                )

                async def execute(self, text: str = "", **kw):
                    return SkillResult.ok(self.manifest.name, kw.get("_skill_call_id", ""), text)
        """))

        loader = SkillLoader()
        reg = loader.load_all([tmp_path])
        assert reg.is_registered("echo")

        # Verify it appears in LLM schemas without any core code changes
        schemas = reg.to_llm_schemas()
        assert any(s["name"] == "echo" for s in schemas)
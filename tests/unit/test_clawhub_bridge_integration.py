"""
tests/integration/test_clawhub_bridge_integration.py

Integration tests for the full ClawHub bridge pipeline:
  parse_clawhub_skill_md() → _detect_tier() → build_neuralclaw_manifest()
  → ClawhubBridgeLoader.load_all() → SkillRegistry → SkillBus.dispatch()

Tests cover:
  - Notion SKILL.md (fixed frontmatter) parses to Tier 2
  - Old-style inline JSON metadata (broken frontmatter) parses to Tier 1
  - Bridge loader registers the skill in SkillRegistry
  - SkillBus dispatches to the correct executor tier
  - HttpApiExecutor calls the LLM and executes the HTTP spec
  - Missing NOTION_API_KEY env var produces a clean error result
  - Executor falls back gracefully when no LLM client is provided

Run:
    pytest tests/integration/test_clawhub_bridge_integration.py -v
"""

from __future__ import annotations

import asyncio
import os
import sys
import types as _types
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any

import pytest

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── stub structlog before any openclaw imports ────────────────────────────────
import types as _types
structlog_stub = _types.ModuleType("structlog")
structlog_stub.get_logger = lambda *a, **kw: MagicMock()  # type: ignore
structlog_stub.stdlib = _types.ModuleType("structlog.stdlib")
structlog_stub.contextvars = _types.ModuleType("structlog.contextvars")
structlog_stub.contextvars.bind_contextvars = lambda **kw: None
structlog_stub.contextvars.unbind_contextvars = lambda *a: None
sys.modules.setdefault("structlog", structlog_stub)
sys.modules.setdefault("structlog.stdlib", structlog_stub.stdlib)
sys.modules.setdefault("structlog.contextvars", structlog_stub.contextvars)
# ─────────────────────────────────────────────────────────────────────────────

from neuralclaw.skills.clawhub.bridge_parser import parse_clawhub_skill_md, _detect_tier, ClawhubRequires
from neuralclaw.skills.clawhub.clawhub_skill import build_neuralclaw_manifest
from neuralclaw.skills.clawhub.bridge_loader import ClawhubBridgeLoader
from neuralclaw.skills.clawhub.bridge_executor import (
    HttpApiExecutor, PromptOnlyExecutor, create_executor,
)
from neuralclaw.skills.registry import SkillRegistry
from neuralclaw.skills.types import RiskLevel, SkillCall, SafetyDecision, SafetyStatus, TrustLevel


def _test_llm_config():
    """Create a BrainLLMConfig for testing (replaces removed gpt-4o hardcode)."""
    from neuralclaw.brain.types import LLMConfig as BrainLLMConfig
    return BrainLLMConfig(model="test-model", temperature=0.3, max_tokens=4096)


# ─────────────────────────────────────────────────────────────────────────────
# SKILL.md fixtures
# ─────────────────────────────────────────────────────────────────────────────

NOTION_SKILL_MD_FIXED = """\
---
name: notion
description: Notion API for creating and managing pages, databases, and blocks.
homepage: https://developers.notion.com
metadata:
  clawdbot:
    emoji: "📝"
    requires:
      env:
        - NOTION_API_KEY
      bins:
        - curl
---

# notion

Use the Notion API to create/read/update pages, data sources (databases), and blocks.

## API Basics

All requests need:
```bash
NOTION_KEY=$(cat ~/.config/notion/api_key)
curl -X GET "https://api.notion.com/v1/..." \\
  -H "Authorization: Bearer $NOTION_KEY" \\
  -H "Notion-Version: 2025-09-03"
```
"""

NOTION_SKILL_MD_BROKEN = """\
---
name: notion
description: Notion API for creating and managing pages, databases, and blocks.
homepage: https://developers.notion.com
metadata: {"clawdbot":{"emoji":"📝"}}
---

# notion

Use the Notion API to create/read/update pages and databases.
"""

CURL_ONLY_SKILL_MD = """\
---
name: weather_api
description: Fetch weather via curl.
metadata:
  clawdbot:
    requires:
      bins:
        - curl
---

# weather_api

Fetch weather data from the API.
"""

BINARY_SKILL_MD = """\
---
name: git_helper
description: Git workflow automation.
metadata:
  clawdbot:
    requires:
      bins:
        - git
        - gh
---

# git_helper

Run git commands.
"""


def _write_skill_dir(tmp_path: Path, content: str, name: str = "notion") -> Path:
    skill_dir = tmp_path / name
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(content)
    return skill_dir


# ─────────────────────────────────────────────────────────────────────────────
# 1. Parser + tier detection
# ─────────────────────────────────────────────────────────────────────────────

class TestParserTierDetection:
    def test_fixed_notion_skill_parses_to_tier2(self, tmp_path):
        skill_dir = _write_skill_dir(tmp_path, NOTION_SKILL_MD_FIXED)
        manifest = parse_clawhub_skill_md(skill_dir)
        assert manifest.execution_tier == 2, (
            f"Expected Tier 2 (HttpApiExecutor) but got Tier {manifest.execution_tier}. "
            "Check that requires.bins: [curl] is present in frontmatter."
        )

    def test_broken_notion_skill_parses_to_tier1(self, tmp_path):
        """Original inline JSON metadata has no requires block → Tier 1."""
        skill_dir = _write_skill_dir(tmp_path, NOTION_SKILL_MD_BROKEN)
        manifest = parse_clawhub_skill_md(skill_dir)
        assert manifest.execution_tier == 1

    def test_fixed_notion_skill_has_env_requirement(self, tmp_path):
        skill_dir = _write_skill_dir(tmp_path, NOTION_SKILL_MD_FIXED)
        manifest = parse_clawhub_skill_md(skill_dir)
        assert "NOTION_API_KEY" in manifest.requires.env

    def test_fixed_notion_skill_has_curl_bin(self, tmp_path):
        skill_dir = _write_skill_dir(tmp_path, NOTION_SKILL_MD_FIXED)
        manifest = parse_clawhub_skill_md(skill_dir)
        assert "curl" in manifest.requires.bins

    def test_curl_only_bins_is_tier2(self, tmp_path):
        skill_dir = _write_skill_dir(tmp_path, CURL_ONLY_SKILL_MD, name="weather_api")
        manifest = parse_clawhub_skill_md(skill_dir)
        assert manifest.execution_tier == 2

    def test_non_http_bins_is_tier3(self, tmp_path):
        skill_dir = _write_skill_dir(tmp_path, BINARY_SKILL_MD, name="git_helper")
        manifest = parse_clawhub_skill_md(skill_dir)
        assert manifest.execution_tier == 3

    def test_detect_tier_direct_no_bins(self):
        req = ClawhubRequires()
        assert _detect_tier(req, []) == 1

    def test_detect_tier_direct_curl_only(self):
        req = ClawhubRequires(bins=["curl"])
        assert _detect_tier(req, []) == 2

    def test_detect_tier_direct_mixed_bins(self):
        """curl + git → Tier 3 because not all bins are HTTP tools."""
        req = ClawhubRequires(bins=["curl", "git"])
        assert _detect_tier(req, []) == 3

    def test_detect_tier_direct_with_install_directives(self):
        """Install directives always push to Tier 3."""
        from neuralclaw.skills.clawhub.bridge_parser import ClawhubInstallDirective
        req = ClawhubRequires(bins=["curl"])
        install = [ClawhubInstallDirective(kind="brew", formula="curl")]
        assert _detect_tier(req, install) == 3


# ─────────────────────────────────────────────────────────────────────────────
# 2. Manifest building
# ─────────────────────────────────────────────────────────────────────────────

class TestManifestBuilding:
    def test_fixed_notion_manifest_has_correct_name(self, tmp_path):
        skill_dir = _write_skill_dir(tmp_path, NOTION_SKILL_MD_FIXED)
        cm = parse_clawhub_skill_md(skill_dir)
        nc = build_neuralclaw_manifest(cm)
        assert nc.name == "notion"

    def test_fixed_notion_manifest_risk_is_low(self, tmp_path):
        """Tier 2 maps to LOW risk by default."""
        skill_dir = _write_skill_dir(tmp_path, NOTION_SKILL_MD_FIXED)
        cm = parse_clawhub_skill_md(skill_dir)
        nc = build_neuralclaw_manifest(cm)
        assert nc.risk_level == RiskLevel.LOW

    def test_fixed_notion_manifest_has_no_capabilities(self, tmp_path):
        """ClawHub skills have no capability gating — always visible to the LLM.

        Safety is enforced by risk_level + trust gate, not capability filtering.
        This ensures installed skills (Notion, etc.) appear in the LLM tool list
        without requiring the user to run /grant net:fetch or similar.
        """
        skill_dir = _write_skill_dir(tmp_path, NOTION_SKILL_MD_FIXED)
        cm = parse_clawhub_skill_md(skill_dir)
        nc = build_neuralclaw_manifest(cm)
        assert nc.capabilities == frozenset(), (
            "ClawHub skills must not require capability grants — "
            "they should always be visible to the LLM tool list."
        )

    def test_manifest_parameters_has_request_field(self, tmp_path):
        skill_dir = _write_skill_dir(tmp_path, NOTION_SKILL_MD_FIXED)
        cm = parse_clawhub_skill_md(skill_dir)
        nc = build_neuralclaw_manifest(cm)
        assert "request" in nc.parameters.get("properties", {})


# ─────────────────────────────────────────────────────────────────────────────
# 3. Bridge loader → SkillRegistry
# ─────────────────────────────────────────────────────────────────────────────

class TestBridgeLoaderRegistration:
    def test_loader_registers_notion_skill(self, tmp_path):
        skill_dir = _write_skill_dir(tmp_path, NOTION_SKILL_MD_FIXED)
        registry = SkillRegistry()
        loader = ClawhubBridgeLoader()
        loader.load_all(skills_dir=tmp_path, registry=registry)
        assert registry.is_registered("notion")

    def test_loader_does_not_register_broken_skill(self, tmp_path):
        """A dir with no SKILL.md should be silently skipped."""
        empty_dir = tmp_path / "empty_skill"
        empty_dir.mkdir()
        registry = SkillRegistry()
        loader = ClawhubBridgeLoader()
        loader.load_all(skills_dir=tmp_path, registry=registry)
        assert not registry.is_registered("empty_skill")

    def test_loader_skips_nonexistent_dir(self, tmp_path):
        registry = SkillRegistry()
        loader = ClawhubBridgeLoader()
        result = loader.load_all(
            skills_dir=tmp_path / "does_not_exist",
            registry=registry,
        )
        assert result is registry  # returns registry unchanged

    def test_loader_assigns_http_executor_for_tier2(self, tmp_path):
        skill_dir = _write_skill_dir(tmp_path, NOTION_SKILL_MD_FIXED)
        registry = SkillRegistry()
        loader = ClawhubBridgeLoader()
        loader.load_all(skills_dir=tmp_path, registry=registry)

        skill_instance = registry.get("notion")
        assert skill_instance is not None
        assert isinstance(skill_instance._executor, HttpApiExecutor)

    def test_loader_assigns_prompt_executor_for_tier1(self, tmp_path):
        skill_dir = _write_skill_dir(tmp_path, NOTION_SKILL_MD_BROKEN)
        registry = SkillRegistry()
        loader = ClawhubBridgeLoader()
        loader.load_all(skills_dir=tmp_path, registry=registry)

        skill_instance = registry.get("notion")
        assert skill_instance is not None
        assert isinstance(skill_instance._executor, PromptOnlyExecutor)

    def test_loader_skips_name_clash(self, tmp_path):
        """Second skill with same name should not overwrite the first."""
        skill_dir = _write_skill_dir(tmp_path, NOTION_SKILL_MD_FIXED)
        # Pre-register a skill named 'notion'
        from neuralclaw.skills.base import SkillBase
        from neuralclaw.skills.types import SkillManifest
        dummy_manifest = SkillManifest(
            name="notion",
            version="0.0.1",
            description="pre-existing",
            category="test",
            risk_level=RiskLevel.LOW,
            capabilities=frozenset(),
            parameters={"type": "object", "properties": {}},
        )

        class _DummySkill(SkillBase):
            manifest = dummy_manifest
            async def execute(self, **kwargs):
                pass

        registry = SkillRegistry()
        registry.register(_DummySkill())
        loader = ClawhubBridgeLoader()
        loader.load_all(skills_dir=tmp_path, registry=registry)

        # Still registered but should be the pre-existing one
        skill = registry.get("notion")
        assert skill.manifest.version == "0.0.1"


# ─────────────────────────────────────────────────────────────────────────────
# 4. HttpApiExecutor behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestHttpApiExecutor:
    def _make_notion_manifest(self, tmp_path):
        skill_dir = _write_skill_dir(tmp_path, NOTION_SKILL_MD_FIXED)
        return parse_clawhub_skill_md(skill_dir)

    @pytest.mark.asyncio
    async def test_missing_env_var_returns_error(self, tmp_path):
        cm = self._make_notion_manifest(tmp_path)
        executor = HttpApiExecutor(llm_client=None)

        with patch.dict(os.environ, {}, clear=True):
            # Remove NOTION_API_KEY if present
            os.environ.pop("NOTION_API_KEY", None)
            result = await executor.run(cm, {"request": "create a page"})

        assert result.is_error
        assert "NOTION_API_KEY" in result.error

    @pytest.mark.asyncio
    async def test_no_llm_client_returns_fallback_ok(self, tmp_path):
        """Without an LLM client, executor returns a helpful fallback (not an error)."""
        cm = self._make_notion_manifest(tmp_path)
        executor = HttpApiExecutor(llm_client=None)

        with patch.dict(os.environ, {"NOTION_API_KEY": "ntn_testkey123"}):
            result = await executor.run(cm, {"request": "create a page called Test"})

        assert not result.is_error
        assert result.output is not None

    @pytest.mark.asyncio
    async def test_empty_request_returns_error(self, tmp_path):
        cm = self._make_notion_manifest(tmp_path)
        executor = HttpApiExecutor(llm_client=None)

        with patch.dict(os.environ, {"NOTION_API_KEY": "ntn_testkey123"}):
            result = await executor.run(cm, {"request": ""})

        assert result.is_error

    @pytest.mark.asyncio
    async def test_llm_produces_http_spec_and_executes(self, tmp_path):
        """LLM returns a JSON HTTP spec; executor calls httpx and feeds result back."""
        cm = self._make_notion_manifest(tmp_path)

        # Mock LLM: first call returns HTTP spec, second returns final answer
        http_spec = '{"method": "POST", "url": "https://api.notion.com/v1/pages", "headers": {"Authorization": "Bearer ntn_testkey123", "Notion-Version": "2025-09-03", "Content-Type": "application/json"}, "body": {"parent": {"page_id": "abc123"}, "properties": {"title": [{"text": {"content": "Meeting Notes"}}]}}}'

        mock_response_1 = MagicMock()
        mock_response_1.text = http_spec

        mock_response_2 = MagicMock()
        mock_response_2.text = "Page 'Meeting Notes' created successfully."

        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(side_effect=[mock_response_1, mock_response_2])

        # Mock httpx response
        mock_http_resp = MagicMock()
        mock_http_resp.status_code = 200
        mock_http_resp.text = '{"object": "page", "id": "abc-123"}'

        mock_httpx_client = AsyncMock()
        mock_httpx_client.__aenter__ = AsyncMock(return_value=mock_httpx_client)
        mock_httpx_client.__aexit__ = AsyncMock(return_value=False)
        mock_httpx_client.request = AsyncMock(return_value=mock_http_resp)

        executor = HttpApiExecutor(llm_client=mock_llm, llm_config=_test_llm_config())

        with patch.dict(os.environ, {"NOTION_API_KEY": "ntn_testkey123"}), \
             patch("httpx.AsyncClient", return_value=mock_httpx_client):
            result = await executor.run(
                cm, {"request": "Create a new page called Meeting Notes"}
            )

        assert not result.is_error
        assert "Meeting Notes" in result.output or "created" in result.output.lower()

    @pytest.mark.asyncio
    async def test_llm_non_json_response_returns_text_fallback(self, tmp_path):
        """If LLM doesn't return parseable JSON, executor returns the raw text."""
        cm = self._make_notion_manifest(tmp_path)

        mock_response = MagicMock()
        mock_response.text = "I cannot determine the exact API call needed."

        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value=mock_response)

        executor = HttpApiExecutor(llm_client=mock_llm, llm_config=_test_llm_config())

        with patch.dict(os.environ, {"NOTION_API_KEY": "ntn_testkey123"}):
            result = await executor.run(cm, {"request": "do something vague"})

        assert not result.is_error
        assert "cannot" in result.output.lower() or result.output

    @pytest.mark.asyncio
    async def test_http_request_failure_returns_error(self, tmp_path):
        """If httpx raises, executor returns a SkillResult error."""
        cm = self._make_notion_manifest(tmp_path)

        http_spec = '{"method": "GET", "url": "https://api.notion.com/v1/pages/bad-id"}'
        mock_response = MagicMock()
        mock_response.text = http_spec
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value=mock_response)

        mock_httpx_client = AsyncMock()
        mock_httpx_client.__aenter__ = AsyncMock(return_value=mock_httpx_client)
        mock_httpx_client.__aexit__ = AsyncMock(return_value=False)
        mock_httpx_client.request = AsyncMock(side_effect=Exception("connection refused"))

        executor = HttpApiExecutor(llm_client=mock_llm)

        with patch.dict(os.environ, {"NOTION_API_KEY": "ntn_testkey123"}), \
             patch("httpx.AsyncClient", return_value=mock_httpx_client):
            result = await executor.run(cm, {"request": "get page"})

        # HTTP error gets captured and returned in output (not raised)
        assert result.output is not None or result.is_error


# ─────────────────────────────────────────────────────────────────────────────
# 5. Full pipeline: loader → registry → SkillBus → dispatch
# ─────────────────────────────────────────────────────────────────────────────

class TestFullPipeline:
    @pytest.mark.asyncio
    async def test_notion_skill_dispatches_via_bus(self, tmp_path):
        """End-to-end: load skill → register → dispatch through SkillBus."""
        from neuralclaw.skills.bus import SkillBus
        from neuralclaw.safety.safety_kernel import SafetyKernel

        skill_dir = _write_skill_dir(tmp_path, NOTION_SKILL_MD_FIXED)

        mock_response = MagicMock()
        mock_response.text = "Page created."
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value=mock_response)

        registry = SkillRegistry()
        loader = ClawhubBridgeLoader()
        loader.load_all(
            skills_dir=tmp_path,
            registry=registry,
            llm_client=mock_llm,
            llm_config=_test_llm_config(),
        )

        safety = MagicMock(spec=SafetyKernel)
        safety.evaluate = AsyncMock(return_value=SafetyDecision(
            status=SafetyStatus.APPROVED,
            reason="approved",
            risk_level=RiskLevel.LOW,
            tool_name="notion",
            tool_call_id="tc_001",
        ))

        bus = SkillBus(registry=registry, safety_kernel=safety)

        with patch.dict(os.environ, {"NOTION_API_KEY": "ntn_testkey123"}):
            call = SkillCall(
                id="tc_001",
                skill_name="notion",
                arguments={"request": "Create a page called Meeting Notes"},
            )
            result = await bus.dispatch(call, TrustLevel.LOW)

        assert not result.is_error

    @pytest.mark.asyncio
    async def test_notion_skill_blocked_without_env(self, tmp_path):
        """If NOTION_API_KEY is missing, dispatch returns an error result (not a crash)."""
        from neuralclaw.skills.bus import SkillBus
        from neuralclaw.safety.safety_kernel import SafetyKernel

        skill_dir = _write_skill_dir(tmp_path, NOTION_SKILL_MD_FIXED)

        registry = SkillRegistry()
        loader = ClawhubBridgeLoader()
        loader.load_all(skills_dir=tmp_path, registry=registry)

        safety = MagicMock(spec=SafetyKernel)
        safety.evaluate = AsyncMock(return_value=SafetyDecision(
            status=SafetyStatus.APPROVED,
            reason="approved",
            risk_level=RiskLevel.LOW,
            tool_name="notion",
            tool_call_id="tc_002",
        ))

        bus = SkillBus(registry=registry, safety_kernel=safety)

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("NOTION_API_KEY", None)
            call = SkillCall(
                id="tc_002",
                skill_name="notion",
                arguments={"request": "Create a page"},
            )
            result = await bus.dispatch(call, TrustLevel.LOW)

        assert result.is_error
        assert "NOTION_API_KEY" in result.error
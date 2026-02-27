"""
tests/unit/test_clawhub_bridge.py — Unit Tests for ClawHub Bridge Adapter

Covers:
  1. Bridge parser — frontmatter parsing, tier detection, aliases, references
  2. Dependency checker — binary and env checks
  3. Env injector — validation and value retrieval
  4. Manifest builder — name sanitization, tier→risk mapping
  5. ClawhubSkill — execute delegation
  6. Bridge loader — discovery, registration, name clashes
  7. Installer — lock file management
"""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# 1. Bridge Parser
# ─────────────────────────────────────────────────────────────────────────────

class TestBridgeParser:
    """Tests for skills/clawhub/bridge_parser.py"""

    def _write_skill(self, tmp_path: Path, frontmatter: str, body: str) -> Path:
        """Helper to create a SKILL.md in a temp dir."""
        skill_dir = tmp_path / "test_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            f"---\n{frontmatter}---\n\n{body}", encoding="utf-8",
        )
        return skill_dir

    def test_parse_tier1_prompt_only(self, tmp_path):
        """A skill with no bins/install → Tier 1."""
        from neuralclaw.skills.clawhub.bridge_parser import parse_clawhub_skill_md

        skill_dir = self._write_skill(tmp_path, textwrap.dedent("""\
            name: commit-msg-helper
            description: Write commit messages
            version: "1.0.0"
            metadata:
              openclaw:
                emoji: "📝"
        """), "You are a helpful commit message writer.")

        m = parse_clawhub_skill_md(skill_dir)
        assert m.name == "commit-msg-helper"
        assert m.description == "Write commit messages"
        assert m.execution_tier == 1
        assert m.emoji == "📝"
        assert "commit message writer" in m.body

    def test_parse_tier2_http(self, tmp_path):
        """A skill requiring only curl → Tier 2."""
        from neuralclaw.skills.clawhub.bridge_parser import parse_clawhub_skill_md

        skill_dir = self._write_skill(tmp_path, textwrap.dedent("""\
            name: weather-api
            description: Fetch weather data
            version: "2.0.0"
            metadata:
              openclaw:
                primaryEnv: WEATHER_API_KEY
                requires:
                  env:
                    - WEATHER_API_KEY
                  bins:
                    - curl
        """), "Use curl to fetch weather.")

        m = parse_clawhub_skill_md(skill_dir)
        assert m.execution_tier == 2
        assert "WEATHER_API_KEY" in m.requires.env
        assert "curl" in m.requires.bins
        assert m.primary_env == "WEATHER_API_KEY"

    def test_parse_tier3_binary(self, tmp_path):
        """A skill requiring a non-HTTP binary → Tier 3."""
        from neuralclaw.skills.clawhub.bridge_parser import parse_clawhub_skill_md

        skill_dir = self._write_skill(tmp_path, textwrap.dedent("""\
            name: todoist-cli
            description: Manage Todoist tasks
            version: "1.2.0"
            metadata:
              openclaw:
                requires:
                  bins:
                    - todoist
                  env:
                    - TODOIST_API_TOKEN
                install:
                  - kind: brew
                    formula: todoist
                    bins:
                      - todoist
        """), "Use todoist CLI to manage tasks.")

        m = parse_clawhub_skill_md(skill_dir)
        assert m.execution_tier == 3
        assert "todoist" in m.requires.bins
        assert len(m.install_directives) == 1
        assert m.install_directives[0].kind == "brew"
        assert m.install_directives[0].formula == "todoist"

    def test_parse_clawdbot_alias(self, tmp_path):
        """The parser should accept 'clawdbot' as metadata namespace alias."""
        from neuralclaw.skills.clawhub.bridge_parser import parse_clawhub_skill_md

        skill_dir = self._write_skill(tmp_path, textwrap.dedent("""\
            name: alias-test
            description: Test aliases
            version: "1.0.0"
            metadata:
              clawdbot:
                emoji: "🤖"
                requires:
                  env:
                    - SOME_KEY
        """), "Body text.")

        m = parse_clawhub_skill_md(skill_dir)
        assert m.emoji == "🤖"
        assert "SOME_KEY" in m.requires.env

    def test_parse_no_frontmatter(self, tmp_path):
        """A SKILL.md with no frontmatter should still parse (Tier 1 default)."""
        from neuralclaw.skills.clawhub.bridge_parser import parse_clawhub_skill_md

        skill_dir = tmp_path / "plain_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "Just instructions, no frontmatter.", encoding="utf-8",
        )

        m = parse_clawhub_skill_md(skill_dir)
        assert m.execution_tier == 1
        assert "instructions" in m.body

    def test_parse_references(self, tmp_path):
        """References/ folder content should be appended to body."""
        from neuralclaw.skills.clawhub.bridge_parser import parse_clawhub_skill_md

        skill_dir = self._write_skill(tmp_path, textwrap.dedent("""\
            name: ref-test
            description: Test references
            version: "1.0.0"
        """), "Main body.")

        refs = skill_dir / "references"
        refs.mkdir()
        (refs / "api.md").write_text("API reference content.", encoding="utf-8")

        m = parse_clawhub_skill_md(skill_dir)
        assert "API reference content" in m.body
        assert "api.md" in m.extra_files

    def test_parse_missing_skill_md(self, tmp_path):
        """FileNotFoundError when no SKILL.md exists."""
        from neuralclaw.skills.clawhub.bridge_parser import parse_clawhub_skill_md

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(FileNotFoundError):
            parse_clawhub_skill_md(empty_dir)

    def test_sanitize_name(self):
        """Name sanitizer converts hyphens and special chars to underscores."""
        from neuralclaw.skills.clawhub.bridge_parser import _sanitize_name
        assert _sanitize_name("todoist-cli") == "todoist_cli"
        assert _sanitize_name("My Skill!") == "my_skill"
        assert _sanitize_name("___") == "unknown_skill"
        assert _sanitize_name("") == "unknown_skill"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dependency Checker
# ─────────────────────────────────────────────────────────────────────────────

class TestDependencyChecker:
    """Tests for skills/clawhub/dependency_checker.py"""

    def test_check_bins_all_present(self):
        from neuralclaw.skills.clawhub.dependency_checker import check_bins
        from neuralclaw.skills.clawhub.bridge_parser import ClawhubRequires

        req = ClawhubRequires(bins=["python", "sh"])
        ok, missing = check_bins(req)
        # python and sh should be available in test env
        assert ok is True
        assert missing == []

    def test_check_bins_missing(self):
        from neuralclaw.skills.clawhub.dependency_checker import check_bins
        from neuralclaw.skills.clawhub.bridge_parser import ClawhubRequires

        req = ClawhubRequires(bins=["nonexistent_binary_xyz_99"])
        ok, missing = check_bins(req)
        assert ok is False
        assert "nonexistent_binary_xyz_99" in missing

    def test_check_env_present(self):
        from neuralclaw.skills.clawhub.dependency_checker import check_env
        from neuralclaw.skills.clawhub.bridge_parser import ClawhubRequires

        with patch.dict(os.environ, {"TEST_VAR_XYZ": "value"}):
            req = ClawhubRequires(env=["TEST_VAR_XYZ"])
            ok, missing = check_env(req)
            assert ok is True
            assert missing == []

    def test_check_env_missing(self):
        from neuralclaw.skills.clawhub.dependency_checker import check_env
        from neuralclaw.skills.clawhub.bridge_parser import ClawhubRequires

        # Ensure it's not set
        os.environ.pop("MISSING_TEST_VAR_999", None)
        req = ClawhubRequires(env=["MISSING_TEST_VAR_999"])
        ok, missing = check_env(req)
        assert ok is False
        assert "MISSING_TEST_VAR_999" in missing

    def test_check_any_bins(self):
        from neuralclaw.skills.clawhub.dependency_checker import check_bins
        from neuralclaw.skills.clawhub.bridge_parser import ClawhubRequires

        # python should exist, fake_bin should not
        req = ClawhubRequires(any_bins=["python", "fake_bin_xyz"])
        ok, missing = check_bins(req)
        assert ok is True

    def test_build_install_command(self):
        from neuralclaw.skills.clawhub.dependency_checker import build_install_command
        from neuralclaw.skills.clawhub.bridge_parser import ClawhubInstallDirective

        d = ClawhubInstallDirective(kind="brew", formula="todoist")
        assert build_install_command(d) == "brew install todoist"

        d2 = ClawhubInstallDirective(kind="node", package="@todoist/cli")
        assert build_install_command(d2) == "npm install -g @todoist/cli"

        d3 = ClawhubInstallDirective(kind="unknown_kind")
        assert build_install_command(d3) is None


# ─────────────────────────────────────────────────────────────────────────────
# 3. Env Injector
# ─────────────────────────────────────────────────────────────────────────────

class TestEnvInjector:
    """Tests for skills/clawhub/env_injector.py"""

    def test_validate_env_all_set(self):
        from neuralclaw.skills.clawhub.env_injector import validate_env
        from neuralclaw.skills.clawhub.bridge_parser import ClawhubRequires

        with patch.dict(os.environ, {"A_VAR": "1", "B_VAR": "2"}):
            req = ClawhubRequires(env=["A_VAR", "B_VAR"])
            ok, missing = validate_env(req)
            assert ok is True
            assert missing == []

    def test_validate_env_partial(self):
        from neuralclaw.skills.clawhub.env_injector import validate_env
        from neuralclaw.skills.clawhub.bridge_parser import ClawhubRequires

        os.environ.pop("C_VAR_MISSING", None)
        with patch.dict(os.environ, {"A_VAR": "1"}):
            req = ClawhubRequires(env=["A_VAR", "C_VAR_MISSING"])
            ok, missing = validate_env(req)
            assert ok is False
            assert "C_VAR_MISSING" in missing

    def test_get_env_values(self):
        from neuralclaw.skills.clawhub.env_injector import get_env_values
        from neuralclaw.skills.clawhub.bridge_parser import ClawhubRequires

        with patch.dict(os.environ, {"X_VAR": "hello"}):
            req = ClawhubRequires(env=["X_VAR", "MISSING_ONE"])
            vals = get_env_values(req)
            assert vals == {"X_VAR": "hello"}

    def test_format_missing_env_message(self):
        from neuralclaw.skills.clawhub.env_injector import format_missing_env_message

        msg = format_missing_env_message("test_skill", ["API_KEY"], primary_env="API_KEY")
        assert "API_KEY" in msg
        assert "(primary)" in msg
        assert "test_skill" in msg


# ─────────────────────────────────────────────────────────────────────────────
# 4. Manifest Builder & ClawhubSkill
# ─────────────────────────────────────────────────────────────────────────────

class TestClawhubSkill:
    """Tests for skills/clawhub/clawhub_skill.py"""

    def test_sanitize_name(self):
        from neuralclaw.skills.clawhub.clawhub_skill import _sanitize_name
        assert _sanitize_name("todoist-cli") == "todoist_cli"
        assert _sanitize_name("My Cool Skill") == "my_cool_skill"

    def test_build_manifest_tier1(self):
        from neuralclaw.skills.clawhub.clawhub_skill import build_neuralclaw_manifest
        from neuralclaw.skills.clawhub.bridge_parser import ClawhubSkillManifest, ClawhubRequires
        from neuralclaw.skills.types import RiskLevel

        cm = ClawhubSkillManifest(
            name="test-prompt-skill",
            description="A test skill",
            version="1.0.0",
            requires=ClawhubRequires(),
            execution_tier=1,
        )
        manifest = build_neuralclaw_manifest(cm)
        assert manifest.name == "test_prompt_skill"
        assert manifest.risk_level == RiskLevel.LOW
        assert manifest.category == "clawhub"
        assert manifest.requires_confirmation is False

    def test_build_manifest_tier3_high_risk(self):
        from neuralclaw.skills.clawhub.clawhub_skill import build_neuralclaw_manifest
        from neuralclaw.skills.clawhub.bridge_parser import ClawhubSkillManifest, ClawhubRequires
        from neuralclaw.skills.types import RiskLevel

        cm = ClawhubSkillManifest(
            name="binary-skill",
            description="Needs binaries",
            version="2.1.0",
            requires=ClawhubRequires(bins=["todoist"]),
            execution_tier=3,
        )
        manifest = build_neuralclaw_manifest(cm)
        assert manifest.risk_level == RiskLevel.HIGH
        assert manifest.requires_confirmation is True
        # Tier-3 ClawHub skills require 'shell:run' capability gate.
        # Tier-1 and tier-2 skills have no capability gating.
        assert manifest.capabilities == frozenset({"shell:run"})

    def test_build_manifest_with_settings(self):
        from neuralclaw.skills.clawhub.clawhub_skill import build_neuralclaw_manifest
        from neuralclaw.skills.clawhub.bridge_parser import ClawhubSkillManifest, ClawhubRequires
        from neuralclaw.skills.types import RiskLevel

        # Mock settings with custom risk defaults
        settings = MagicMock()
        settings.clawhub.risk_defaults.prompt_only = "MEDIUM"
        settings.clawhub.risk_defaults.api_http = "MEDIUM"
        settings.clawhub.risk_defaults.binary_execution = "CRITICAL"

        cm = ClawhubSkillManifest(
            name="custom-risk",
            description="Custom risk",
            version="1.0.0",
            execution_tier=1,
        )
        manifest = build_neuralclaw_manifest(cm, settings=settings)
        assert manifest.risk_level == RiskLevel.MEDIUM

    @pytest.mark.asyncio
    async def test_execute_no_executor(self):
        """ClawhubSkill without executor returns failure."""
        from neuralclaw.skills.clawhub.clawhub_skill import ClawhubSkill, build_neuralclaw_manifest
        from neuralclaw.skills.clawhub.bridge_parser import ClawhubSkillManifest
        from neuralclaw.skills.types import SkillManifest, RiskLevel

        cm = ClawhubSkillManifest(
            name="no-exec", description="test", version="1.0.0",
            execution_tier=1,
        )
        manifest = build_neuralclaw_manifest(cm)

        cls = type("TestSkill", (ClawhubSkill,), {"manifest": manifest})
        skill = cls()
        result = await skill.execute(request="hello")
        assert result.success is False
        assert "missing executor" in result.error.lower()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Bridge Executor — Tier 1 (PromptOnly)
# ─────────────────────────────────────────────────────────────────────────────

class TestPromptOnlyExecutor:
    """Tests for PromptOnlyExecutor in bridge_executor.py"""

    @pytest.mark.asyncio
    async def test_no_llm_returns_instructions(self):
        """Without LLM client, should return instructions dict."""
        from neuralclaw.skills.clawhub.bridge_executor import PromptOnlyExecutor
        from neuralclaw.skills.clawhub.bridge_parser import ClawhubSkillManifest

        executor = PromptOnlyExecutor(llm_client=None)
        cm = ClawhubSkillManifest(
            name="test", description="test", version="1.0.0",
            body="You are a helper.", execution_tier=1,
        )
        result = await executor.run(cm, {"request": "help me", "_skill_call_id": "t1"})
        assert result.success is True
        assert result.output["type"] == "clawhub_prompt_skill"
        assert "You are a helper" in result.output["instructions"]

    @pytest.mark.asyncio
    async def test_no_request_returns_error(self):
        from neuralclaw.skills.clawhub.bridge_executor import PromptOnlyExecutor
        from neuralclaw.skills.clawhub.bridge_parser import ClawhubSkillManifest

        executor = PromptOnlyExecutor(llm_client=None)
        cm = ClawhubSkillManifest(
            name="test", description="test", version="1.0.0",
            execution_tier=1,
        )
        result = await executor.run(cm, {"request": "", "_skill_call_id": "t1"})
        assert result.success is False


# ─────────────────────────────────────────────────────────────────────────────
# 6. Bridge Loader
# ─────────────────────────────────────────────────────────────────────────────

class TestBridgeLoader:
    """Tests for skills/clawhub/bridge_loader.py"""

    def _create_skill(self, parent: Path, name: str, tier: int = 1) -> Path:
        """Create a minimal ClawHub skill directory."""
        skill_dir = parent / name
        skill_dir.mkdir()
        bins_block = ""
        if tier == 3:
            bins_block = textwrap.dedent("""\
                requires:
                  bins:
                    - fakebinary
            """)
        elif tier == 2:
            bins_block = textwrap.dedent("""\
                requires:
                  bins:
                    - curl
            """)

        (skill_dir / "SKILL.md").write_text(textwrap.dedent(f"""\
            ---
            name: {name}
            description: Test skill {name}
            version: "1.0.0"
            metadata:
              openclaw:
                {bins_block}
            ---

            Instructions for {name}.
        """), encoding="utf-8")
        return skill_dir

    def test_load_discovers_skills(self, tmp_path):
        """Loader should discover and register skills."""
        from neuralclaw.skills.clawhub.bridge_loader import ClawhubBridgeLoader
        from neuralclaw.skills.registry import SkillRegistry

        self._create_skill(tmp_path, "skill_a")
        self._create_skill(tmp_path, "skill_b")

        registry = SkillRegistry()
        loader = ClawhubBridgeLoader()
        loader.load_all(tmp_path, registry)

        assert registry.is_registered("skill_a")
        assert registry.is_registered("skill_b")
        assert len(registry) == 2

    def test_load_skips_hidden_dirs(self, tmp_path):
        """Directories starting with . or _ should be skipped."""
        from neuralclaw.skills.clawhub.bridge_loader import ClawhubBridgeLoader
        from neuralclaw.skills.registry import SkillRegistry

        self._create_skill(tmp_path, ".hidden_skill")
        self._create_skill(tmp_path, "_private_skill")
        self._create_skill(tmp_path, "visible_skill")

        registry = SkillRegistry()
        loader = ClawhubBridgeLoader()
        loader.load_all(tmp_path, registry)

        assert not registry.is_registered("hidden_skill")
        assert not registry.is_registered("private_skill")
        assert registry.is_registered("visible_skill")

    def test_load_skips_name_clash(self, tmp_path):
        """If a skill name is already registered, it should be skipped."""
        from neuralclaw.skills.clawhub.bridge_loader import ClawhubBridgeLoader
        from neuralclaw.skills.clawhub.clawhub_skill import ClawhubSkill, build_neuralclaw_manifest
        from neuralclaw.skills.clawhub.bridge_parser import ClawhubSkillManifest
        from neuralclaw.skills.registry import SkillRegistry

        # Pre-register a skill with the same name
        cm = ClawhubSkillManifest(
            name="clash_skill", description="test", version="1.0.0",
            execution_tier=1,
        )
        manifest = build_neuralclaw_manifest(cm)
        cls = type("ClashSkill", (ClawhubSkill,), {"manifest": manifest})
        instance = cls()

        registry = SkillRegistry()
        registry.register(instance)

        # Create a ClawHub skill with the same name
        self._create_skill(tmp_path, "clash_skill")

        loader = ClawhubBridgeLoader()
        loader.load_all(tmp_path, registry)

        # Should still have just 1 skill (the pre-registered one)
        assert len(registry) == 1

    def test_load_nonexistent_dir(self, tmp_path):
        """Should handle missing directory gracefully."""
        from neuralclaw.skills.clawhub.bridge_loader import ClawhubBridgeLoader
        from neuralclaw.skills.registry import SkillRegistry

        registry = SkillRegistry()
        loader = ClawhubBridgeLoader()
        loader.load_all(tmp_path / "nonexistent", registry)

        assert len(registry) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 7. Installer — Lock File
# ─────────────────────────────────────────────────────────────────────────────

class TestInstaller:
    """Tests for onboard/clawhub_installer.py lock management."""

    def test_lock_read_write(self, tmp_path):
        from neuralclaw.onboard.clawhub_installer import _read_lock, _write_lock

        settings = MagicMock()
        settings.clawhub.skills_dir = str(tmp_path / "skills")
        Path(settings.clawhub.skills_dir).mkdir(parents=True)

        # Initially empty
        lock = _read_lock(settings)
        assert lock["version"] == 1
        assert lock["skills"] == {}

        # Write some data
        lock["skills"]["test_skill"] = {"version": "1.0.0"}
        _write_lock(settings, lock)

        # Read it back
        lock2 = _read_lock(settings)
        assert "test_skill" in lock2["skills"]
        assert lock2["skills"]["test_skill"]["version"] == "1.0.0"

    def test_list_installed_empty(self, tmp_path):
        from neuralclaw.onboard.clawhub_installer import list_installed

        settings = MagicMock()
        settings.clawhub.skills_dir = str(tmp_path / "skills")
        Path(settings.clawhub.skills_dir).mkdir(parents=True)

        result = list_installed(settings, console=None)
        assert result == 0

    def test_remove_not_installed(self, tmp_path):
        from neuralclaw.onboard.clawhub_installer import remove_skill

        settings = MagicMock()
        settings.clawhub.skills_dir = str(tmp_path / "skills")
        Path(settings.clawhub.skills_dir).mkdir(parents=True)

        result = remove_skill("nonexistent", settings, console=None)
        assert result == 1


# ─────────────────────────────────────────────────────────────────────────────
# 8. Config — ClawhubSettings
# ─────────────────────────────────────────────────────────────────────────────

class TestClawhubConfig:
    """Tests for ClawhubSettings in config/settings.py"""

    def test_default_settings(self):
        from neuralclaw.config.settings import ClawhubSettings
        s = ClawhubSettings()
        assert s.enabled is True
        assert s.skills_dir == "./data/clawhub/skills"
        assert s.execution.allow_binary_skills is True
        assert s.execution.auto_install_deps is False
        assert s.risk_defaults.prompt_only == "LOW"
        assert s.risk_defaults.binary_execution == "HIGH"

    def test_custom_settings(self):
        from neuralclaw.config.settings import ClawhubSettings
        s = ClawhubSettings(
            enabled=False,
            skills_dir="/custom/path",
            execution={"auto_install_deps": True},
            risk_defaults={"prompt_only": "MEDIUM"},
        )
        assert s.enabled is False
        assert s.skills_dir == "/custom/path"
        assert s.execution.auto_install_deps is True
        assert s.risk_defaults.prompt_only == "MEDIUM"

    def test_tier_detection_logic(self):
        """Tier detection edge cases."""
        from neuralclaw.skills.clawhub.bridge_parser import _detect_tier, ClawhubRequires, ClawhubInstallDirective

        # No bins, no install → Tier 1
        assert _detect_tier(ClawhubRequires(), []) == 1

        # Only curl → Tier 2
        assert _detect_tier(ClawhubRequires(bins=["curl"]), []) == 2

        # curl + wget → still Tier 2
        assert _detect_tier(ClawhubRequires(bins=["curl", "wget"]), []) == 2

        # curl + non-HTTP bin → Tier 3
        assert _detect_tier(ClawhubRequires(bins=["curl", "ffmpeg"]), []) == 3

        # No bins but has install directive → Tier 3
        d = ClawhubInstallDirective(kind="brew", formula="todoist")
        assert _detect_tier(ClawhubRequires(), [d]) == 3

        # HTTP bins but also install directive → Tier 3
        assert _detect_tier(ClawhubRequires(bins=["curl"]), [d]) == 3

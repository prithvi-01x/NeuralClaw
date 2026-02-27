"""
tests/unit/test_config.py — Phase 1: Config Hardening Tests

Covers:
  - Valid config loads cleanly
  - whitelist_extra: ["*"] is rejected at parse time
  - Blocked system paths are rejected (working_dir + allowed_paths)
  - Invalid trust level is rejected
  - Invalid log level is rejected
  - Invalid LLM provider is rejected
  - validate_all() raises ConfigError with numbered list
  - validate_all() catches missing API key for chosen provider
  - validate_all() catches missing API key for fallback provider
  - validate_all() catches wildcard whitelist that slipped through
  - NEURALCLAW_CONFIG env var is respected by load_settings()
  - Explicit config_path argument takes priority over env var
"""

from __future__ import annotations

import os
import textwrap
import pytest
from pathlib import Path
from pydantic import ValidationError
from unittest.mock import patch


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_settings(**overrides):
    """Build a Settings object from keyword overrides (no YAML file needed)."""
    from neuralclaw.config.settings import Settings
    return Settings(**overrides)


def _make_terminal_cfg(**kwargs):
    from neuralclaw.config.settings import TerminalToolConfig
    return TerminalToolConfig(**kwargs)


def _make_filesystem_cfg(**kwargs):
    from neuralclaw.config.settings import FilesystemToolConfig
    return FilesystemToolConfig(**kwargs)


def _make_agent_cfg(**kwargs):
    from neuralclaw.config.settings import AgentConfig
    return AgentConfig(**kwargs)


def _make_llm_cfg(**kwargs):
    from neuralclaw.config.settings import LLMConfig
    return LLMConfig(**kwargs)


def _make_logging_cfg(**kwargs):
    from neuralclaw.config.settings import LoggingConfig
    return LoggingConfig(**kwargs)


def _make_safety_cfg(**kwargs):
    from neuralclaw.config.settings import SafetyConfig
    return SafetyConfig(**kwargs)


# ── TerminalToolConfig ────────────────────────────────────────────────────────

class TestTerminalToolConfig:
    def test_valid_empty_whitelist(self):
        cfg = _make_terminal_cfg(whitelist_extra=[])
        assert cfg.whitelist_extra == []

    def test_valid_specific_commands(self):
        cfg = _make_terminal_cfg(whitelist_extra=["ffprobe", "convert"])
        assert "ffprobe" in cfg.whitelist_extra

    def test_wildcard_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            _make_terminal_cfg(whitelist_extra=["*"])
        assert "wildcard" in str(exc_info.value).lower() or "*" in str(exc_info.value)

    def test_wildcard_in_list_rejected(self):
        with pytest.raises(ValidationError):
            _make_terminal_cfg(whitelist_extra=["ls", "*", "grep"])

    def test_default_working_dir_ok(self):
        cfg = _make_terminal_cfg()
        assert cfg.working_dir == "./data/agent_files"

    def test_home_subdir_ok(self):
        # Use /home/user if it exists, else /tmp (avoids /root which is blocked)
        safe_path = "/home/user/projects" if Path("/home").exists() else "/tmp/work"
        cfg = _make_terminal_cfg(working_dir=safe_path)
        assert cfg.working_dir == safe_path

    def test_etc_working_dir_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            _make_terminal_cfg(working_dir="/etc/passwd")
        assert "protected" in str(exc_info.value).lower()

    def test_usr_working_dir_rejected(self):
        with pytest.raises(ValidationError):
            _make_terminal_cfg(working_dir="/usr/bin")

    def test_root_working_dir_rejected(self):
        with pytest.raises(ValidationError):
            _make_terminal_cfg(working_dir="/root")


# ── FilesystemToolConfig ──────────────────────────────────────────────────────

class TestFilesystemToolConfig:
    def test_valid_relative_path(self):
        cfg = _make_filesystem_cfg(allowed_paths=["./data/agent_files"])
        assert "./data/agent_files" in cfg.allowed_paths

    def test_valid_home_path(self):
        cfg = _make_filesystem_cfg(allowed_paths=["~/Documents/neuralclaw"])
        assert len(cfg.allowed_paths) == 1

    def test_etc_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            _make_filesystem_cfg(allowed_paths=["/etc"])
        assert "protected" in str(exc_info.value).lower()

    def test_sys_rejected(self):
        with pytest.raises(ValidationError):
            _make_filesystem_cfg(allowed_paths=["/sys/kernel"])

    def test_proc_rejected(self):
        with pytest.raises(ValidationError):
            _make_filesystem_cfg(allowed_paths=["/proc"])

    def test_mixed_list_one_blocked(self):
        # If any entry is blocked, the whole list is rejected
        with pytest.raises(ValidationError):
            _make_filesystem_cfg(allowed_paths=["./data/agent_files", "/etc"])


# ── AgentConfig ───────────────────────────────────────────────────────────────

class TestAgentConfig:
    def test_valid_trust_low(self):
        assert _make_agent_cfg(default_trust_level="low").default_trust_level == "low"

    def test_valid_trust_medium(self):
        assert _make_agent_cfg(default_trust_level="medium").default_trust_level == "medium"

    def test_valid_trust_high(self):
        assert _make_agent_cfg(default_trust_level="high").default_trust_level == "high"

    def test_invalid_trust_level(self):
        with pytest.raises(ValidationError) as exc_info:
            _make_agent_cfg(default_trust_level="superuser")
        assert "trust" in str(exc_info.value).lower() or "superuser" in str(exc_info.value)

    def test_zero_iterations_rejected(self):
        with pytest.raises(ValidationError):
            _make_agent_cfg(max_iterations_per_turn=0)

    def test_zero_timeout_rejected(self):
        with pytest.raises(ValidationError):
            _make_agent_cfg(max_turn_timeout_seconds=0)

    def test_negative_iterations_rejected(self):
        with pytest.raises(ValidationError):
            _make_agent_cfg(max_iterations_per_turn=-1)


# ── LLMConfig ─────────────────────────────────────────────────────────────────

class TestLLMConfig:
    def test_valid_provider(self):
        for p in ["openai", "anthropic", "ollama", "gemini", "bytez", "openrouter"]:
            cfg = _make_llm_cfg(default_provider=p)
            assert cfg.default_provider == p

    def test_unknown_provider_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            _make_llm_cfg(default_provider="grok")
        assert "grok" in str(exc_info.value) or "supported" in str(exc_info.value).lower()

    def test_temperature_bounds(self):
        _make_llm_cfg(temperature=0.0)
        _make_llm_cfg(temperature=2.0)
        with pytest.raises(ValidationError):
            _make_llm_cfg(temperature=2.1)
        with pytest.raises(ValidationError):
            _make_llm_cfg(temperature=-0.1)

    def test_zero_max_tokens_rejected(self):
        with pytest.raises(ValidationError):
            _make_llm_cfg(max_tokens=0)


# ── LoggingConfig ─────────────────────────────────────────────────────────────

class TestLoggingConfig:
    def test_valid_levels(self):
        for lvl in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            cfg = _make_logging_cfg(level=lvl)
            assert cfg.level == lvl

    def test_case_insensitive(self):
        cfg = _make_logging_cfg(level="debug")
        assert cfg.level == "DEBUG"

    def test_invalid_level_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            _make_logging_cfg(level="VERBOSE")
        assert "VERBOSE" in str(exc_info.value) or "valid" in str(exc_info.value).lower()


# ── SafetyConfig ──────────────────────────────────────────────────────────────

class TestSafetyConfig:
    def test_valid_risk_levels(self):
        cfg = _make_safety_cfg(require_confirmation_for=["HIGH", "CRITICAL"])
        assert "HIGH" in cfg.require_confirmation_for

    def test_normalised_to_upper(self):
        cfg = _make_safety_cfg(require_confirmation_for=["high", "critical"])
        assert cfg.require_confirmation_for == ["HIGH", "CRITICAL"]

    def test_invalid_risk_level_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            _make_safety_cfg(require_confirmation_for=["HIGH", "EXTREME"])
        assert "EXTREME" in str(exc_info.value) or "unknown" in str(exc_info.value).lower()


# ── validate_all ─────────────────────────────────────────────────────────────

class TestValidateAll:
    def _settings_with_key(self, provider: str, key_value: str | None = "sk-test"):
        """Build Settings with a given provider + optional API key."""
        from neuralclaw.config.settings import Settings, LLMConfig
        env_map = {
            "openai":     {"OPENAI_API_KEY":     key_value},
            "anthropic":  {"ANTHROPIC_API_KEY":  key_value},
            "bytez":      {"BYTEZ_API_KEY":       key_value},
            "gemini":     {"GEMINI_API_KEY":       key_value},
            "openrouter": {"OPENROUTER_API_KEY":  key_value},
        }
        kwargs = {}
        if provider in env_map and key_value is not None:
            kwargs.update(env_map[provider])
        elif provider in env_map:
            pass  # key_value is None → don't set it → missing key

        s = Settings(
            llm=LLMConfig(default_provider=provider),
            **kwargs,
        )
        return s

    def test_passes_with_ollama_no_key(self):
        """Ollama needs no API key — validate_all should pass."""
        from neuralclaw.config.settings import Settings, LLMConfig
        s = Settings(llm=LLMConfig(default_provider="ollama"))
        s.validate_all()  # should not raise

    def test_fails_missing_openai_key(self):
        from neuralclaw.config.settings import ConfigError, Settings, LLMConfig
        s = Settings(llm=LLMConfig(default_provider="openai"))
        with pytest.raises(ConfigError) as exc_info:
            s.validate_all()
        assert "OPENAI_API_KEY" in str(exc_info.value)

    def test_fails_missing_anthropic_key(self):
        from neuralclaw.config.settings import ConfigError, Settings, LLMConfig
        s = Settings(llm=LLMConfig(default_provider="anthropic"))
        with pytest.raises(ConfigError) as exc_info:
            s.validate_all()
        assert "ANTHROPIC_API_KEY" in str(exc_info.value)

    def test_passes_with_key_present(self):
        from neuralclaw.config.settings import Settings, LLMConfig
        s = Settings(
            llm=LLMConfig(default_provider="openai"),
            OPENAI_API_KEY="sk-test-key",
        )
        s.validate_all()  # should not raise

    def test_error_message_is_numbered(self):
        """validate_all lists problems with numbers."""
        from neuralclaw.config.settings import ConfigError, Settings, LLMConfig
        s = Settings(llm=LLMConfig(default_provider="openai"))
        with pytest.raises(ConfigError) as exc_info:
            s.validate_all()
        msg = str(exc_info.value)
        assert "1." in msg

    def test_multiple_errors_all_reported(self):
        """All problems are collected and reported, not just the first."""
        from neuralclaw.config.settings import ConfigError, Settings, LLMConfig
        # Missing key + empty timezone = 2 errors
        s = Settings(llm=LLMConfig(default_provider="openai"))
        # Manually corrupt timezone to trigger second error
        object.__setattr__(s.scheduler, "timezone", "   ")
        with pytest.raises(ConfigError) as exc_info:
            s.validate_all()
        msg = str(exc_info.value)
        assert "2" in msg or "problem" in msg.lower()

    def test_fallback_provider_missing_key_caught(self):
        from neuralclaw.config.settings import ConfigError, Settings, LLMConfig
        s = Settings(
            llm=LLMConfig(
                default_provider="ollama",
                fallback_providers=["openai"],
            )
        )
        # No OPENAI_API_KEY → should fail
        with pytest.raises(ConfigError) as exc_info:
            s.validate_all()
        assert "openai" in str(exc_info.value).lower()


# ── Config path resolution ────────────────────────────────────────────────────

class TestConfigPathResolution:
    def test_explicit_path_takes_priority(self, tmp_path):
        """Explicit config_path arg overrides env var."""
        from neuralclaw.config.settings import _resolve_config_path
        cfg_file = tmp_path / "custom.yaml"
        cfg_file.write_text("")
        env_file = tmp_path / "env.yaml"

        with patch.dict(os.environ, {"NEURALCLAW_CONFIG": str(env_file)}):
            resolved = _resolve_config_path(str(cfg_file))
        assert resolved == Path(str(cfg_file))

    def test_env_var_used_when_no_explicit_path(self, tmp_path):
        """NEURALCLAW_CONFIG env var is used when no explicit path given."""
        from neuralclaw.config.settings import _resolve_config_path
        env_file = tmp_path / "env_config.yaml"

        with patch.dict(os.environ, {"NEURALCLAW_CONFIG": str(env_file)}):
            resolved = _resolve_config_path(None)
        assert resolved == Path(str(env_file))

    def test_default_path_when_no_arg_no_env(self):
        """Falls back to config/config.yaml when nothing is set."""
        from neuralclaw.config.settings import _resolve_config_path
        env = {k: v for k, v in os.environ.items() if k != "NEURALCLAW_CONFIG"}
        with patch.dict(os.environ, env, clear=True):
            resolved = _resolve_config_path(None)
        assert resolved == Path("config/config.yaml")

    def test_load_settings_from_file(self, tmp_path):
        """load_settings reads a custom YAML file correctly."""
        from neuralclaw.config.settings import load_settings, _singleton
        import neuralclaw.config.settings as cs

        cfg_file = tmp_path / "test_config.yaml"
        cfg_file.write_text(textwrap.dedent("""
            agent:
              name: "TestAgent"
              default_trust_level: "medium"
            llm:
              default_provider: "ollama"
        """))

        # Reset singleton so we get a fresh load
        cs._singleton = None
        settings = load_settings(str(cfg_file))

        assert settings.agent.name == "TestAgent"
        assert settings.agent.default_trust_level == "medium"
        assert settings.llm.default_provider == "ollama"

        # Reset singleton after test
        cs._singleton = None
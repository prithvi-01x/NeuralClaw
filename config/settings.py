"""
config/settings.py — NeuralClaw Configuration

Single source of truth for all runtime configuration.
Merges config.yaml (defaults/structure) with .env (secrets).

Usage:
    from config.settings import get_settings
    settings = get_settings()
    print(settings.llm.default_model)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ─────────────────────────────────────────────────────────────────────────────
# Sub-models (nested config sections)
# ─────────────────────────────────────────────────────────────────────────────


class AgentConfig:
    name: str = "NeuralClaw"
    version: str = "1.0.0"
    max_iterations_per_turn: int = 10
    max_turn_timeout_seconds: int = 300
    default_trust_level: str = "low"


class LLMProviderConfig:
    base_url: Optional[str] = None


class OllamaProviderConfig:
    base_url: str = "http://localhost:11434"


class LLMConfig:
    default_provider: str = "openai"
    default_model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 4096


class MemoryConfig:
    chroma_persist_dir: str = "./data/chroma"
    sqlite_path: str = "./data/sqlite/episodes.db"
    embedding_model: str = "all-MiniLM-L6-v2"
    max_short_term_turns: int = 20
    relevance_threshold: float = 0.85


class BrowserToolConfig:
    headless: bool = True
    user_agent: str = "Mozilla/5.0 (compatible; NeuralClaw/1.0)"
    timeout_ms: int = 15000


class TerminalToolConfig:
    working_dir: str = "~/agent_files"
    default_timeout_seconds: int = 30
    docker_sandbox: bool = False
    docker_image: str = "python:3.11-slim"
    whitelist_extra: list[str] = []


class FilesystemToolConfig:
    allowed_paths: list[str] = ["~/agent_files"]


class ToolsConfig:
    browser: BrowserToolConfig = BrowserToolConfig()
    terminal: TerminalToolConfig = TerminalToolConfig()
    filesystem: FilesystemToolConfig = FilesystemToolConfig()


class SafetyConfig:
    default_permission_level: str = "read"
    require_confirmation_for: list[str] = ["HIGH", "CRITICAL"]


class MCPServerConfig:
    transport: str = "stdio"
    command: str = ""
    args: list[str] = []
    enabled: bool = False


class MCPConfig:
    servers: dict[str, dict] = {}


class TelegramConfig:
    authorized_user_ids: list[int] = []


class SchedulerConfig:
    timezone: str = "UTC"
    max_concurrent_tasks: int = 3


class LoggingConfig:
    level: str = "INFO"
    log_dir: str = "./data/logs"
    max_file_size_mb: int = 100
    backup_count: int = 5
    console_output: bool = True
    json_format: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Root Settings (reads from .env + config.yaml)
# ─────────────────────────────────────────────────────────────────────────────


class Settings(BaseSettings):
    """
    NeuralClaw runtime settings.

    Priority (highest → lowest):
      1. Environment variables
      2. .env file
      3. config.yaml
      4. Field defaults
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Secrets from .env ────────────────────────────────────────────────────
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    telegram_bot_token: Optional[str] = Field(default=None, alias="TELEGRAM_BOT_TOKEN")
    telegram_user_id: Optional[int] = Field(default=None, alias="TELEGRAM_USER_ID")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    serpapi_key: Optional[str] = Field(default=None, alias="SERPAPI_KEY")

    # ── Structured config (populated from config.yaml via load_from_yaml) ────
    agent: dict[str, Any] = Field(default_factory=dict)
    llm: dict[str, Any] = Field(default_factory=dict)
    memory: dict[str, Any] = Field(default_factory=dict)
    tools: dict[str, Any] = Field(default_factory=dict)
    safety: dict[str, Any] = Field(default_factory=dict)
    mcp: dict[str, Any] = Field(default_factory=dict)
    telegram: dict[str, Any] = Field(default_factory=dict)
    scheduler: dict[str, Any] = Field(default_factory=dict)
    logging: dict[str, Any] = Field(default_factory=dict)

    @field_validator("telegram_user_id", mode="before")
    @classmethod
    def coerce_user_id(cls, v: Any) -> Optional[int]:
        if v in (None, "", "null"):
            return None
        return int(v)

    # ── Convenience accessors ─────────────────────────────────────────────────

    @property
    def agent_name(self) -> str:
        return self.agent.get("name", "NeuralClaw")

    @property
    def log_level(self) -> str:
        return self.logging.get("level", "INFO").upper()

    @property
    def log_dir(self) -> Path:
        return Path(self.logging.get("log_dir", "./data/logs"))

    @property
    def log_json_format(self) -> bool:
        return self.logging.get("json_format", True)

    @property
    def log_console_output(self) -> bool:
        return self.logging.get("console_output", True)

    @property
    def default_llm_provider(self) -> str:
        return self.llm.get("default_provider", "openai")

    @property
    def default_llm_model(self) -> str:
        return self.llm.get("default_model", "gpt-4o")

    @property
    def authorized_telegram_ids(self) -> list[int]:
        """Merge TELEGRAM_USER_ID env var with any IDs in config.yaml."""
        ids: list[int] = list(self.telegram.get("authorized_user_ids", []))
        if self.telegram_user_id and self.telegram_user_id not in ids:
            ids.append(self.telegram_user_id)
        return ids

    def validate_required_for_interface(self, interface: str) -> list[str]:
        """Return list of missing required secrets for a given interface."""
        missing = []
        if interface == "telegram":
            if not self.telegram_bot_token:
                missing.append("TELEGRAM_BOT_TOKEN")
            if not self.authorized_telegram_ids:
                missing.append("TELEGRAM_USER_ID")
        if self.default_llm_provider == "openai" and not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        if self.default_llm_provider == "anthropic" and not self.anthropic_api_key:
            missing.append("ANTHROPIC_API_KEY")
        return missing


# ─────────────────────────────────────────────────────────────────────────────
# Loader: merges config.yaml into Settings
# ─────────────────────────────────────────────────────────────────────────────


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file, returning empty dict if not found."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_settings(config_path: str | Path = "config/config.yaml") -> Settings:
    """
    Load settings by merging config.yaml with environment variables.

    config.yaml provides structure and defaults.
    .env / environment provides secrets and overrides.
    """
    yaml_data = _load_yaml(Path(config_path))

    # Build init kwargs from YAML so Pydantic can validate them
    # Env vars (loaded by BaseSettings automatically) take priority
    init_kwargs: dict[str, Any] = {
        k: v
        for k, v in yaml_data.items()
        if k
        in {
            "agent",
            "llm",
            "memory",
            "tools",
            "safety",
            "mcp",
            "telegram",
            "scheduler",
            "logging",
        }
    }

    return Settings(**init_kwargs)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the cached global Settings instance.

    Call this from anywhere in the codebase:
        from config.settings import get_settings
        s = get_settings()
    """
    return load_settings()
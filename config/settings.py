"""
config/settings.py — NeuralClaw Runtime Settings

Merges config.yaml (defaults/structure) with .env (secrets).
Pydantic-powered — all fields are validated and typed.

New in this version:
  - LLMRetryConfig: max_attempts, base_delay, max_delay
  - LLMConfig.retry + LLMConfig.fallback_providers
  - MemoryConfig.compact_after_turns + compact_keep_recent
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel


# ─────────────────────────────────────────────────────────────────────────────
# Sub-models
# ─────────────────────────────────────────────────────────────────────────────


class AgentConfig(BaseModel):
    name: str = "NeuralClaw"
    version: str = "1.0.0"
    max_iterations_per_turn: int = 10
    max_turn_timeout_seconds: int = 300
    default_trust_level: str = "low"


class LLMRetryConfig(BaseModel):
    """Exponential backoff config for transient LLM errors."""
    max_attempts: int = 3
    base_delay: float = 1.0   # seconds before first retry
    max_delay: float = 30.0   # cap on backoff delay


class LLMConfig(BaseModel):
    default_provider: str = "openai"
    default_model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 4096

    # Retry/failover (new)
    retry: LLMRetryConfig = Field(default_factory=LLMRetryConfig)
    # Ordered list of provider names to try if primary exhausts all retries.
    # Only providers with a valid API key / reachable endpoint are used.
    # Example: fallback_providers: [ollama]  — fall back to local Ollama
    fallback_providers: List[str] = Field(default_factory=list)


class MemoryConfig(BaseModel):
    chroma_persist_dir: str = "./data/chroma"
    sqlite_path: str = "./data/sqlite/episodes.db"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    max_short_term_turns: int = 20
    relevance_threshold: float = 0.55

    # Compaction (new)
    # When turn_count reaches this threshold, /compact is suggested automatically.
    # Set to 0 to disable auto-suggestion.
    compact_after_turns: int = 15
    # After compaction, how many recent turns to keep alongside the summary.
    compact_keep_recent: int = 4


class BrowserToolConfig(BaseModel):
    headless: bool = True
    user_agent: str = "Mozilla/5.0 (compatible; NeuralClaw/1.0)"
    timeout_ms: int = 15000


class TerminalToolConfig(BaseModel):
    working_dir: str = "./data/agent_files"
    default_timeout_seconds: int = 30
    docker_sandbox: bool = False
    docker_image: str = "python:3.11-slim"
    whitelist_extra: list[str] = Field(default_factory=list)


class FilesystemToolConfig(BaseModel):
    allowed_paths: list[str] = Field(default_factory=lambda: ["~/agent_files"])


class ToolsConfig(BaseModel):
    browser: BrowserToolConfig = Field(default_factory=BrowserToolConfig)
    terminal: TerminalToolConfig = Field(default_factory=TerminalToolConfig)
    filesystem: FilesystemToolConfig = Field(default_factory=FilesystemToolConfig)


class SafetyConfig(BaseModel):
    default_permission_level: str = "read"
    require_confirmation_for: list[str] = Field(default_factory=lambda: ["HIGH", "CRITICAL"])


class MCPServerConfig(BaseModel):
    transport: str = "stdio"
    command: str = ""
    args: list[str] = Field(default_factory=list)
    enabled: bool = False


class MCPConfig(BaseModel):
    servers: dict[str, dict] = Field(default_factory=dict)


class TelegramConfig(BaseModel):
    authorized_user_ids: list[int] = Field(default_factory=list)


class SchedulerConfig(BaseModel):
    timezone: str = "UTC"
    max_concurrent_tasks: int = 3


class LoggingConfig(BaseModel):
    level: str = "INFO"
    log_dir: str = "./data/logs"
    max_file_size_mb: int = 100
    backup_count: int = 5
    console_output: bool = False   # False = logs go to file only (clean CLI)
    json_format: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Root Settings
# ─────────────────────────────────────────────────────────────────────────────

_key_map = {
    "openai":     ("OPENAI_API_KEY",     "openai_api_key"),
    "anthropic":  ("ANTHROPIC_API_KEY",  "anthropic_api_key"),
    "bytez":      ("BYTEZ_API_KEY",      "bytez_api_key"),
    "gemini":     ("GEMINI_API_KEY",     "gemini_api_key"),
    "openrouter": ("OPENROUTER_API_KEY", "openrouter_api_key"),
}


class Settings(BaseSettings):
    """
    NeuralClaw runtime settings.

    Priority (highest to lowest):
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

    # -- Secrets from .env ---------------------------------------------------
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    bytez_api_key: Optional[str] = Field(default=None, alias="BYTEZ_API_KEY")
    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    openrouter_api_key: Optional[str] = Field(default=None, alias="OPENROUTER_API_KEY")
    telegram_bot_token: Optional[str] = Field(default=None, alias="TELEGRAM_BOT_TOKEN")
    telegram_user_id: Optional[int] = Field(default=None, alias="TELEGRAM_USER_ID")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    serpapi_key: Optional[str] = Field(default=None, alias="SERPAPI_KEY")

    # -- Structured config (from config.yaml) --------------------------------
    agent: AgentConfig = Field(default_factory=AgentConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @field_validator("telegram_user_id", mode="before")
    @classmethod
    def coerce_user_id(cls, v: Any) -> Optional[int]:
        if v in (None, "", "null"):
            return None
        return int(v)

    @field_validator("agent", mode="before")
    @classmethod
    def _coerce_agent(cls, v: Any) -> Any:
        return AgentConfig(**v) if isinstance(v, dict) else v

    @field_validator("llm", mode="before")
    @classmethod
    def _coerce_llm(cls, v: Any) -> Any:
        return LLMConfig(**v) if isinstance(v, dict) else v

    @field_validator("memory", mode="before")
    @classmethod
    def _coerce_memory(cls, v: Any) -> Any:
        return MemoryConfig(**v) if isinstance(v, dict) else v

    @field_validator("tools", mode="before")
    @classmethod
    def _coerce_tools(cls, v: Any) -> Any:
        return ToolsConfig(**v) if isinstance(v, dict) else v

    @field_validator("safety", mode="before")
    @classmethod
    def _coerce_safety(cls, v: Any) -> Any:
        return SafetyConfig(**v) if isinstance(v, dict) else v

    @field_validator("mcp", mode="before")
    @classmethod
    def _coerce_mcp(cls, v: Any) -> Any:
        return MCPConfig(**v) if isinstance(v, dict) else v

    @field_validator("telegram", mode="before")
    @classmethod
    def _coerce_telegram(cls, v: Any) -> Any:
        return TelegramConfig(**v) if isinstance(v, dict) else v

    @field_validator("scheduler", mode="before")
    @classmethod
    def _coerce_scheduler(cls, v: Any) -> Any:
        return SchedulerConfig(**v) if isinstance(v, dict) else v

    @field_validator("logging", mode="before")
    @classmethod
    def _coerce_logging(cls, v: Any) -> Any:
        return LoggingConfig(**v) if isinstance(v, dict) else v

    # -- Convenience properties ----------------------------------------------

    @property
    def default_llm_provider(self) -> str:
        return self.llm.default_provider

    @property
    def default_llm_model(self) -> str:
        return self.llm.default_model

    @property
    def agent_name(self) -> str:
        return self.agent.name

    @property
    def log_level(self) -> str:
        return self.logging.level.upper()

    @property
    def log_dir(self) -> "Path":
        return Path(self.logging.log_dir)

    @property
    def log_json_format(self) -> bool:
        return self.logging.json_format

    @property
    def log_console_output(self) -> bool:
        return self.logging.console_output

    @property
    def ollama_base_url_v1(self) -> str:
        return self.ollama_base_url + "/v1"

    @property
    def authorized_telegram_ids(self) -> list[int]:
        """Merge TELEGRAM_USER_ID env var with any IDs in config.yaml."""
        ids: list[int] = list(self.telegram.authorized_user_ids)
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
        _key_map = {
            "openai":     ("OPENAI_API_KEY",     self.openai_api_key),
            "anthropic":  ("ANTHROPIC_API_KEY",  self.anthropic_api_key),
            "bytez":      ("BYTEZ_API_KEY",       self.bytez_api_key),
            "gemini":     ("GEMINI_API_KEY",      self.gemini_api_key),
            "openrouter": ("OPENROUTER_API_KEY",  self.openrouter_api_key),
        }
        provider = self.default_llm_provider
        if provider in _key_map:
            env_name, value = _key_map[provider]
            if not value:
                missing.append(env_name)
        return missing


# ─────────────────────────────────────────────────────────────────────────────
# Loader + singleton
# ─────────────────────────────────────────────────────────────────────────────

_singleton: Optional[Settings] = None


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_settings(config_path: str | Path = "config/config.yaml") -> Settings:
    """
    Load settings by merging config.yaml with environment variables.

    config.yaml provides structure and defaults.
    .env / environment provides secrets and overrides.
    Registers the result as the global singleton returned by get_settings().
    """
    global _singleton
    yaml_data = _load_yaml(Path(config_path))

    _KNOWN_SECTIONS = {
        "agent", "llm", "memory", "tools", "safety",
        "mcp", "telegram", "scheduler", "logging",
    }
    init_kwargs = {k: v for k, v in yaml_data.items() if k in _KNOWN_SECTIONS}

    instance = Settings(**init_kwargs)
    if _singleton is None:
        _singleton = instance
    return instance


def get_settings() -> Settings:
    """
    Return the global Settings singleton.
    If load_settings() has been called already, returns that instance.
    Otherwise loads from the default config path.
    """
    global _singleton
    if _singleton is None:
        load_settings()
    return _singleton  # type: ignore[return-value]
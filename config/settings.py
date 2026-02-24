"""
config/settings.py — NeuralClaw Runtime Settings

Merges config.yaml (defaults/structure) with .env (secrets).
Pydantic-powered — all fields are validated and typed.

Phase 1 hardening (core-hardening):
  - TerminalToolConfig rejects whitelist_extra: ["*"] at parse time
  - FilesystemToolConfig rejects absolute paths pointing to system dirs
  - AgentConfig validates default_trust_level is a known value
  - validate_all() performs full startup validation and raises ConfigError
    with a clear, human-readable message listing every problem found
  - load_settings() respects NEURALCLAW_CONFIG env var as a fallback
    when no explicit config_path argument is given
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel


# ─────────────────────────────────────────────────────────────────────────────
# Errors
# ─────────────────────────────────────────────────────────────────────────────

class ConfigError(Exception):
    """Raised by validate_all() when one or more config problems are found."""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_VALID_TRUST_LEVELS = {"low", "medium", "high"}
_VALID_LOG_LEVELS   = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

_BLOCKED_PATH_PREFIXES: tuple[str, ...] = (
    "/etc", "/root", "/proc", "/sys", "/dev", "/boot",
    "/private/etc",
    "/usr", "/bin", "/sbin", "/lib", "/lib64",
)


def _is_blocked_system_path(p: str) -> bool:
    try:
        resolved = str(Path(p).expanduser().resolve())
    except (ValueError, OSError):
        return False
    return any(
        resolved == prefix or resolved.startswith(prefix + "/")
        for prefix in _BLOCKED_PATH_PREFIXES
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sub-models
# ─────────────────────────────────────────────────────────────────────────────

class AgentConfig(BaseModel):
    name: str = "NeuralClaw"
    version: str = "1.0.0"
    max_iterations_per_turn: int = 10
    max_turn_timeout_seconds: int = 300
    default_trust_level: str = "low"

    @field_validator("default_trust_level")
    @classmethod
    def _valid_trust(cls, v: str) -> str:
        if v not in _VALID_TRUST_LEVELS:
            raise ValueError(
                f"agent.default_trust_level must be one of "
                f"{sorted(_VALID_TRUST_LEVELS)}, got '{v}'"
            )
        return v

    @field_validator("max_iterations_per_turn")
    @classmethod
    def _positive_iterations(cls, v: int) -> int:
        if v < 1:
            raise ValueError("agent.max_iterations_per_turn must be >= 1")
        return v

    @field_validator("max_turn_timeout_seconds")
    @classmethod
    def _positive_timeout(cls, v: int) -> int:
        if v < 1:
            raise ValueError("agent.max_turn_timeout_seconds must be >= 1")
        return v


class LLMRetryConfig(BaseModel):
    """Exponential backoff config for transient LLM errors."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0


class LLMConfig(BaseModel):
    default_provider: str = "openai"
    default_model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 4096
    retry: LLMRetryConfig = Field(default_factory=LLMRetryConfig)
    fallback_providers: List[str] = Field(default_factory=list)

    @field_validator("default_provider")
    @classmethod
    def _known_provider(cls, v: str) -> str:
        known = {"openai", "anthropic", "ollama", "gemini", "bytez", "openrouter"}
        if v not in known:
            raise ValueError(
                f"llm.default_provider '{v}' is not supported. "
                f"Supported: {sorted(known)}"
            )
        return v

    @field_validator("temperature")
    @classmethod
    def _valid_temperature(cls, v: float) -> float:
        if not (0.0 <= v <= 2.0):
            raise ValueError("llm.temperature must be between 0.0 and 2.0")
        return v

    @field_validator("max_tokens")
    @classmethod
    def _positive_tokens(cls, v: int) -> int:
        if v < 1:
            raise ValueError("llm.max_tokens must be >= 1")
        return v


class MemoryConfig(BaseModel):
    chroma_persist_dir: str = "./data/chroma"
    sqlite_path: str = "./data/sqlite/episodes.db"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    max_short_term_turns: int = 20
    relevance_threshold: float = 0.55
    compact_after_turns: int = 15
    compact_keep_recent: int = 4

    @field_validator("relevance_threshold")
    @classmethod
    def _valid_threshold(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("memory.relevance_threshold must be between 0.0 and 1.0")
        return v


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

    @field_validator("whitelist_extra")
    @classmethod
    def _reject_wildcard_whitelist(cls, v: list[str]) -> list[str]:
        if "*" in v:
            raise ValueError(
                "tools.terminal.whitelist_extra may not contain '*'. "
                "A wildcard disables the entire command safety whitelist. "
                "List specific command names instead, or leave it empty."
            )
        return v

    @field_validator("working_dir")
    @classmethod
    def _safe_working_dir(cls, v: str) -> str:
        if _is_blocked_system_path(v):
            raise ValueError(
                f"tools.terminal.working_dir '{v}' points to a protected "
                f"system directory. Use './data/agent_files' or a path "
                f"under your home directory."
            )
        return v


class FilesystemToolConfig(BaseModel):
    allowed_paths: list[str] = Field(default_factory=lambda: ["./data/agent_files"])

    @field_validator("allowed_paths")
    @classmethod
    def _safe_allowed_paths(cls, v: list[str]) -> list[str]:
        for p in v:
            if _is_blocked_system_path(p):
                raise ValueError(
                    f"tools.filesystem.allowed_paths contains '{p}', which "
                    f"is a protected system directory. Remove it."
                )
        return v


class ToolsConfig(BaseModel):
    browser: BrowserToolConfig = Field(default_factory=BrowserToolConfig)
    terminal: TerminalToolConfig = Field(default_factory=TerminalToolConfig)
    filesystem: FilesystemToolConfig = Field(default_factory=FilesystemToolConfig)


class SafetyConfig(BaseModel):
    default_permission_level: str = "read"
    require_confirmation_for: list[str] = Field(default_factory=lambda: ["HIGH", "CRITICAL"])

    @field_validator("require_confirmation_for")
    @classmethod
    def _valid_risk_levels(cls, v: list[str]) -> list[str]:
        valid = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        bad = [x for x in v if x.upper() not in valid]
        if bad:
            raise ValueError(
                f"safety.require_confirmation_for has unknown risk levels: {bad}. "
                f"Valid values: {sorted(valid)}"
            )
        return [x.upper() for x in v]


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

    @field_validator("max_concurrent_tasks")
    @classmethod
    def _positive_tasks(cls, v: int) -> int:
        if v < 1:
            raise ValueError("scheduler.max_concurrent_tasks must be >= 1")
        return v


class LoggingConfig(BaseModel):
    level: str = "INFO"
    log_dir: str = "./data/logs"
    max_file_size_mb: int = 100
    backup_count: int = 5
    console_output: bool = False
    json_format: bool = True

    @field_validator("level")
    @classmethod
    def _valid_log_level(cls, v: str) -> str:
        upper = v.upper()
        if upper not in _VALID_LOG_LEVELS:
            raise ValueError(
                f"logging.level '{v}' is not valid. "
                f"Must be one of: {sorted(_VALID_LOG_LEVELS)}"
            )
        return upper


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
        key_map = {
            "openai":     ("OPENAI_API_KEY",     self.openai_api_key),
            "anthropic":  ("ANTHROPIC_API_KEY",  self.anthropic_api_key),
            "bytez":      ("BYTEZ_API_KEY",       self.bytez_api_key),
            "gemini":     ("GEMINI_API_KEY",      self.gemini_api_key),
            "openrouter": ("OPENROUTER_API_KEY",  self.openrouter_api_key),
        }
        provider = self.default_llm_provider
        if provider in key_map:
            env_name, value = key_map[provider]
            if not value:
                missing.append(env_name)
        return missing

    def validate_all(self) -> None:
        """
        Full startup validation. Raises ConfigError listing every problem found.

        Called once at startup in main.py bootstrap() before any subsystem
        initialises. Pydantic field validators catch type/value errors at parse
        time; this method catches cross-field and runtime problems that Pydantic
        can't see (e.g. API key presence for the chosen provider, wildcard
        whitelist bypass that survived parse, blocked paths that resolve at
        runtime).
        """
        errors: list[str] = []

        # ── LLM provider API key ─────────────────────────────────────────────
        provider = self.llm.default_provider
        key_map = {
            "openai":     ("OPENAI_API_KEY",     self.openai_api_key),
            "anthropic":  ("ANTHROPIC_API_KEY",  self.anthropic_api_key),
            "bytez":      ("BYTEZ_API_KEY",       self.bytez_api_key),
            "gemini":     ("GEMINI_API_KEY",      self.gemini_api_key),
            "openrouter": ("OPENROUTER_API_KEY",  self.openrouter_api_key),
        }
        if provider in key_map:
            env_name, value = key_map[provider]
            if not value:
                errors.append(
                    f"LLM provider '{provider}' requires {env_name} to be set "
                    f"in your .env file."
                )

        # ── Fallback providers also need their keys ──────────────────────────
        for fp in self.llm.fallback_providers:
            if fp in key_map:
                env_name, value = key_map[fp]
                if not value:
                    errors.append(
                        f"Fallback provider '{fp}' requires {env_name} but it "
                        f"is not set. Remove '{fp}' from llm.fallback_providers "
                        f"or add the key to .env."
                    )

        # ── whitelist_extra wildcard (double-check at runtime) ───────────────
        if "*" in self.tools.terminal.whitelist_extra:
            errors.append(
                "tools.terminal.whitelist_extra contains '*', which disables "
                "the command safety whitelist entirely. Remove it."
            )

        # ── Terminal working_dir is not a system path ────────────────────────
        wd = self.tools.terminal.working_dir
        if _is_blocked_system_path(wd):
            errors.append(
                f"tools.terminal.working_dir '{wd}' points to a protected "
                f"system directory. Use './data/agent_files'."
            )

        # ── Filesystem allowed_paths are not system dirs ─────────────────────
        for p in self.tools.filesystem.allowed_paths:
            if _is_blocked_system_path(p):
                errors.append(
                    f"tools.filesystem.allowed_paths contains '{p}', which "
                    f"is a protected system directory. Remove it."
                )

        # ── Scheduler timezone is non-empty ─────────────────────────────────
        if not self.scheduler.timezone.strip():
            errors.append(
                "scheduler.timezone must not be empty. Use 'UTC' or a tz name."
            )

        # ── Report all errors together ───────────────────────────────────────
        if errors:
            numbered = "\n".join(f"  {i+1}. {e}" for i, e in enumerate(errors))
            raise ConfigError(
                f"\n\nNeuralClaw startup failed — {len(errors)} configuration "
                f"problem(s) found:\n\n{numbered}\n\n"
                f"Fix the issues above in config/config.yaml or your .env file "
                f"and restart.\n"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Loader + singleton
# ─────────────────────────────────────────────────────────────────────────────

import threading as _threading

_singleton: Optional[Settings] = None
_singleton_lock = _threading.Lock()


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_config_path(config_path: str | Path | None) -> Path:
    """
    Resolve the config file path with this priority:
      1. Explicit config_path argument (from --config CLI flag)
      2. NEURALCLAW_CONFIG environment variable
      3. Default: config/config.yaml
    """
    if config_path is not None:
        return Path(config_path)
    env_path = os.environ.get("NEURALCLAW_CONFIG")
    if env_path:
        return Path(env_path)
    return Path("config/config.yaml")


def load_settings(config_path: str | Path | None = None) -> Settings:
    """
    Load settings by merging config.yaml with environment variables.

    Config path resolution order:
      1. config_path argument  (--config CLI flag)
      2. NEURALCLAW_CONFIG env var
      3. config/config.yaml   (default)
    """
    global _singleton
    resolved_path = _resolve_config_path(config_path)
    yaml_data = _load_yaml(resolved_path)

    _KNOWN_SECTIONS = {
        "agent", "llm", "memory", "tools", "safety",
        "mcp", "telegram", "scheduler", "logging",
    }
    init_kwargs = {k: v for k, v in yaml_data.items() if k in _KNOWN_SECTIONS}

    instance = Settings(**init_kwargs)
    with _singleton_lock:
        _singleton = instance
    return instance


def get_settings() -> Settings:
    """
    Return the global Settings singleton.
    If load_settings() has been called already, returns that instance.
    Otherwise loads from the default config path.

    Thread-safe: guarded by _singleton_lock to prevent double-initialisation
    if called concurrently before the first load completes.
    """
    global _singleton
    if _singleton is not None:
        return _singleton  # fast path — no lock needed once set
    with _singleton_lock:
        # Re-check inside the lock in case another thread just initialised it
        if _singleton is None:
            load_settings()
        return _singleton  # type: ignore[return-value]
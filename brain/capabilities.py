"""
brain/capabilities.py — Model Capability Registry

Provides a clean, extensible system for determining what any given model
can do: tool/function calling, vision, streaming, etc.

Design principles:
  - No single source of truth is hardcoded — rules are layered:
      1. Explicit per-model overrides (highest priority)
      2. Provider-level defaults
      3. Heuristic pattern matching on model name
      4. Safe conservative defaults (lowest priority)
  - All lookups are O(1) after initial resolution
  - Capabilities are cached per (provider, model_id) pair
  - The OllamaClient can inject live capability data after probing the API
  - Works for all providers: OpenAI, Anthropic, Gemini, Ollama, OpenRouter, Bytez

Usage:
    from brain.capabilities import get_capabilities, ModelCapabilities

    caps = get_capabilities("ollama", "codellama:latest")
    if not caps.supports_tools:
        from observability.logger import get_logger as _caps_log; _caps_log("brain.capabilities").info("model.chat_only_mode", provider=provider, model=model_id)

    # After a live probe (OllamaClient does this at connect time):
    register_capabilities("ollama", "qwen2.5:3b", supports_tools=True)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Capability descriptor
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelCapabilities:
    """
    Immutable capability snapshot for a (provider, model_id) pair.

    Attributes:
        supports_tools:   Model accepts tool/function calling schemas.
        supports_vision:  Model accepts image inputs.
        supports_stream:  Model supports token streaming.
        chat_only:        Derived convenience flag — True when supports_tools=False.
        tool_hint:        Short human-readable string for UI ("tools ✓" / "chat only").
    """
    supports_tools:  bool = True
    supports_vision: bool = False
    supports_stream: bool = True

    @property
    def chat_only(self) -> bool:
        return not self.supports_tools

    @property
    def tool_hint(self) -> str:
        return "tools ✓" if self.supports_tools else "chat only"

    def with_updates(self, **kwargs) -> "ModelCapabilities":
        """Return a new instance with selected fields overridden."""
        return replace(self, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Provider-level defaults
# ─────────────────────────────────────────────────────────────────────────────

# What we know to be true for every model of a given provider,
# absent any model-specific override.
_PROVIDER_DEFAULTS: dict[str, ModelCapabilities] = {
    "openai":     ModelCapabilities(supports_tools=True,  supports_vision=False, supports_stream=True),
    "anthropic":  ModelCapabilities(supports_tools=True,  supports_vision=True,  supports_stream=True),
    "gemini":     ModelCapabilities(supports_tools=True,  supports_vision=True,  supports_stream=True),
    "openrouter": ModelCapabilities(supports_tools=True,  supports_vision=False, supports_stream=True),
    "ollama":     ModelCapabilities(supports_tools=False, supports_vision=False, supports_stream=True),
    "bytez":      ModelCapabilities(supports_tools=False, supports_vision=False, supports_stream=False),
    "auto":       ModelCapabilities(supports_tools=True,  supports_vision=False, supports_stream=True),
}

# ─────────────────────────────────────────────────────────────────────────────
# Explicit per-model overrides  (provider → model_id_prefix → caps)
# ─────────────────────────────────────────────────────────────────────────────

# Keys are matched as case-insensitive prefixes/substrings of the model_id.
# More specific entries override less specific ones.
# Format: (provider, pattern) → ModelCapabilities

_MODEL_OVERRIDES: list[tuple[str, str, ModelCapabilities]] = [
    # ── OpenAI ────────────────────────────────────────────────────────────────
    ("openai", "gpt-4o",            ModelCapabilities(supports_tools=True,  supports_vision=True,  supports_stream=True)),
    ("openai", "gpt-4-turbo",       ModelCapabilities(supports_tools=True,  supports_vision=True,  supports_stream=True)),
    ("openai", "gpt-4-vision",      ModelCapabilities(supports_tools=True,  supports_vision=True,  supports_stream=True)),
    ("openai", "gpt-4",             ModelCapabilities(supports_tools=True,  supports_vision=False, supports_stream=True)),
    ("openai", "gpt-3.5",           ModelCapabilities(supports_tools=True,  supports_vision=False, supports_stream=True)),
    ("openai", "o1",                ModelCapabilities(supports_tools=False, supports_vision=False, supports_stream=False)),
    ("openai", "o3",                ModelCapabilities(supports_tools=True,  supports_vision=False, supports_stream=False)),

    # ── Anthropic ─────────────────────────────────────────────────────────────
    ("anthropic", "claude-3-5-sonnet", ModelCapabilities(supports_tools=True, supports_vision=True,  supports_stream=True)),
    ("anthropic", "claude-3-5-haiku",  ModelCapabilities(supports_tools=True, supports_vision=True,  supports_stream=True)),
    ("anthropic", "claude-3-opus",     ModelCapabilities(supports_tools=True, supports_vision=True,  supports_stream=True)),
    ("anthropic", "claude-3-sonnet",   ModelCapabilities(supports_tools=True, supports_vision=True,  supports_stream=True)),
    ("anthropic", "claude-3-haiku",    ModelCapabilities(supports_tools=True, supports_vision=True,  supports_stream=True)),
    ("anthropic", "claude-2",          ModelCapabilities(supports_tools=True, supports_vision=False, supports_stream=True)),

    # ── Gemini ────────────────────────────────────────────────────────────────
    ("gemini", "gemini-2.0",        ModelCapabilities(supports_tools=True, supports_vision=True,  supports_stream=True)),
    ("gemini", "gemini-1.5-pro",    ModelCapabilities(supports_tools=True, supports_vision=True,  supports_stream=True)),
    ("gemini", "gemini-1.5-flash",  ModelCapabilities(supports_tools=True, supports_vision=True,  supports_stream=True)),
    ("gemini", "gemini-1.0",        ModelCapabilities(supports_tools=True, supports_vision=False, supports_stream=True)),

    # ── Ollama — models confirmed to support tool calling ─────────────────────
    # llama3.1+ supports tools; llama3.0 and earlier do not
    ("ollama", "llama3.1",          ModelCapabilities(supports_tools=True,  supports_vision=False, supports_stream=True)),
    ("ollama", "llama3.2",          ModelCapabilities(supports_tools=True,  supports_vision=False, supports_stream=True)),
    ("ollama", "llama3.3",          ModelCapabilities(supports_tools=True,  supports_vision=False, supports_stream=True)),
    ("ollama", "llama-3.1",         ModelCapabilities(supports_tools=True,  supports_vision=False, supports_stream=True)),
    ("ollama", "llama-3.2",         ModelCapabilities(supports_tools=True,  supports_vision=False, supports_stream=True)),
    ("ollama", "llama-3.3",         ModelCapabilities(supports_tools=True,  supports_vision=False, supports_stream=True)),
    ("ollama", "mistral-nemo",      ModelCapabilities(supports_tools=True,  supports_vision=False, supports_stream=True)),
    ("ollama", "mistral-small",     ModelCapabilities(supports_tools=True,  supports_vision=False, supports_stream=True)),
    ("ollama", "mixtral",           ModelCapabilities(supports_tools=True,  supports_vision=False, supports_stream=True)),
    ("ollama", "qwen2.5",           ModelCapabilities(supports_tools=True,  supports_vision=False, supports_stream=True)),
    ("ollama", "qwen2",             ModelCapabilities(supports_tools=True,  supports_vision=False, supports_stream=True)),
    ("ollama", "qwq",               ModelCapabilities(supports_tools=True,  supports_vision=False, supports_stream=True)),
    ("ollama", "firefunction",      ModelCapabilities(supports_tools=True,  supports_vision=False, supports_stream=True)),
    ("ollama", "hermes",            ModelCapabilities(supports_tools=True,  supports_vision=False, supports_stream=True)),
    ("ollama", "command-r",         ModelCapabilities(supports_tools=True,  supports_vision=False, supports_stream=True)),
    ("ollama", "aya",               ModelCapabilities(supports_tools=True,  supports_vision=False, supports_stream=True)),
    ("ollama", "granite3",          ModelCapabilities(supports_tools=True,  supports_vision=False, supports_stream=True)),
    ("ollama", "smollm2",           ModelCapabilities(supports_tools=True,  supports_vision=False, supports_stream=True)),

    # Ollama models confirmed NOT to support tools (explicit no-tools)
    ("ollama", "codellama",         ModelCapabilities(supports_tools=False, supports_vision=False, supports_stream=True)),
    ("ollama", "mistral:",          ModelCapabilities(supports_tools=False, supports_vision=False, supports_stream=True)),
    ("ollama", "mistral-7b",        ModelCapabilities(supports_tools=False, supports_vision=False, supports_stream=True)),
    ("ollama", "gemma",             ModelCapabilities(supports_tools=False, supports_vision=False, supports_stream=True)),
    ("ollama", "gemma2",            ModelCapabilities(supports_tools=False, supports_vision=False, supports_stream=True)),
    ("ollama", "gemma3",            ModelCapabilities(supports_tools=False, supports_vision=False, supports_stream=True)),
    ("ollama", "llama2",            ModelCapabilities(supports_tools=False, supports_vision=False, supports_stream=True)),
    ("ollama", "llama-2",           ModelCapabilities(supports_tools=False, supports_vision=False, supports_stream=True)),
    ("ollama", "llama3:",           ModelCapabilities(supports_tools=False, supports_vision=False, supports_stream=True)),
    ("ollama", "llama3.0",          ModelCapabilities(supports_tools=False, supports_vision=False, supports_stream=True)),
    ("ollama", "phi3",              ModelCapabilities(supports_tools=False, supports_vision=False, supports_stream=True)),
    ("ollama", "phi4",              ModelCapabilities(supports_tools=False, supports_vision=False, supports_stream=True)),
    ("ollama", "phi",               ModelCapabilities(supports_tools=False, supports_vision=False, supports_stream=True)),
    ("ollama", "deepseek",          ModelCapabilities(supports_tools=False, supports_vision=False, supports_stream=True)),
    ("ollama", "vicuna",            ModelCapabilities(supports_tools=False, supports_vision=False, supports_stream=True)),
    ("ollama", "wizard",            ModelCapabilities(supports_tools=False, supports_vision=False, supports_stream=True)),
    ("ollama", "orca",              ModelCapabilities(supports_tools=False, supports_vision=False, supports_stream=True)),
    ("ollama", "solar",             ModelCapabilities(supports_tools=False, supports_vision=False, supports_stream=True)),
    ("ollama", "stablelm",          ModelCapabilities(supports_tools=False, supports_vision=False, supports_stream=True)),
    ("ollama", "starcoder",         ModelCapabilities(supports_tools=False, supports_vision=False, supports_stream=True)),
    ("ollama", "openchat",          ModelCapabilities(supports_tools=False, supports_vision=False, supports_stream=True)),
    ("ollama", "neural-chat",       ModelCapabilities(supports_tools=False, supports_vision=False, supports_stream=True)),
    ("ollama", "nemotron",          ModelCapabilities(supports_tools=False, supports_vision=False, supports_stream=True)),
    ("ollama", "kimi",              ModelCapabilities(supports_tools=False, supports_vision=False, supports_stream=True)),
    ("ollama", "ministral",         ModelCapabilities(supports_tools=False, supports_vision=False, supports_stream=True)),
    ("ollama", "embedding",         ModelCapabilities(supports_tools=False, supports_vision=False, supports_stream=False)),

    # Ollama vision models
    ("ollama", "llava",             ModelCapabilities(supports_tools=False, supports_vision=True,  supports_stream=True)),
    ("ollama", "bakllava",          ModelCapabilities(supports_tools=False, supports_vision=True,  supports_stream=True)),
    ("ollama", "moondream",         ModelCapabilities(supports_tools=False, supports_vision=True,  supports_stream=True)),
    ("ollama", "llama3.2-vision",   ModelCapabilities(supports_tools=True,  supports_vision=True,  supports_stream=True)),
]


# ─────────────────────────────────────────────────────────────────────────────
# Runtime override registry (for live probing / user overrides)
# ─────────────────────────────────────────────────────────────────────────────

# (provider, model_id) → ModelCapabilities
# Written by OllamaClient after probing, or by tests/manual config.
_RUNTIME_REGISTRY: dict[tuple[str, str], ModelCapabilities] = {}


# Resolution cache — avoids re-running pattern matching every call
_RESOLUTION_CACHE: dict[tuple[str, str], ModelCapabilities] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def get_capabilities(provider: str, model_id: str) -> ModelCapabilities:
    """
    Return the ModelCapabilities for the given (provider, model_id) pair.

    Resolution order (highest → lowest priority):
      1. Runtime registry  — set by live probing or explicit override
      2. Per-model override table — pattern-matched on model name
      3. Provider default
      4. Global safe default (supports_tools=True — fail open for unknown providers)
    """
    provider  = (provider or "").lower().strip()
    model_id  = (model_id  or "").lower().strip()
    cache_key = (provider, model_id)

    if cache_key in _RESOLUTION_CACHE:
        return _RESOLUTION_CACHE[cache_key]

    caps = _resolve(provider, model_id)
    _RESOLUTION_CACHE[cache_key] = caps
    return caps


def register_capabilities(
    provider: str,
    model_id: str,
    *,
    supports_tools: Optional[bool] = None,
    supports_vision: Optional[bool] = None,
    supports_stream: Optional[bool] = None,
) -> ModelCapabilities:
    """
    Explicitly register or override capabilities for a model at runtime.

    Used by:
      - OllamaClient after live-probing the model's template for tool support
      - Tests
      - User-facing /caps override command (future)

    Returns the newly registered ModelCapabilities.
    """
    provider  = (provider or "").lower().strip()
    model_id  = (model_id  or "").lower().strip()
    cache_key = (provider, model_id)

    # Start from existing resolved caps so unset kwargs preserve current values
    existing = _resolve(provider, model_id)
    updates: dict = {}
    if supports_tools  is not None: updates["supports_tools"]  = supports_tools
    if supports_vision is not None: updates["supports_vision"] = supports_vision
    if supports_stream is not None: updates["supports_stream"] = supports_stream

    new_caps = existing.with_updates(**updates) if updates else existing
    _RUNTIME_REGISTRY[cache_key] = new_caps
    _RESOLUTION_CACHE[cache_key] = new_caps   # invalidate & reset cache entry

    from observability.logger import get_logger
    get_logger("brain.capabilities").debug(
        "capabilities.registered",
        provider=provider,
        model_id=model_id,
        supports_tools=new_caps.supports_tools,
        supports_vision=new_caps.supports_vision,
    )
    return new_caps


def invalidate_cache(provider: Optional[str] = None, model_id: Optional[str] = None) -> None:
    """
    Clear the resolution cache.
    Pass provider+model_id to clear a specific entry; pass nothing to clear all.
    """
    if provider and model_id:
        key = (provider.lower(), model_id.lower())
        _RESOLUTION_CACHE.pop(key, None)
        _RUNTIME_REGISTRY.pop(key, None)
    else:
        _RESOLUTION_CACHE.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Internal resolution logic
# ─────────────────────────────────────────────────────────────────────────────

_SAFE_DEFAULT = ModelCapabilities(supports_tools=True, supports_vision=False, supports_stream=True)


def _resolve(provider: str, model_id: str) -> ModelCapabilities:
    """Full resolution without cache."""

    # 1. Runtime registry (live probing)
    runtime = _RUNTIME_REGISTRY.get((provider, model_id))
    if runtime is not None:
        return runtime

    # 2. Per-model overrides — find the most specific matching pattern
    #    "More specific" = longer pattern length
    best_match: Optional[ModelCapabilities] = None
    best_len = -1

    for (pat_provider, pattern, caps) in _MODEL_OVERRIDES:
        if pat_provider != provider:
            continue
        # Normalize pattern (strip trailing colon used as word-boundary marker)
        pat = pattern.rstrip(":").lower()
        if model_id.startswith(pat) or pat in model_id:
            if len(pat) > best_len:
                best_len = len(pat)
                best_match = caps

    if best_match is not None:
        return best_match

    # 3. Provider default
    provider_default = _PROVIDER_DEFAULTS.get(provider)
    if provider_default is not None:
        return provider_default

    # 4. Global safe default — unknown provider, assume tools supported
    return _SAFE_DEFAULT


# ─────────────────────────────────────────────────────────────────────────────
# Ollama live probing helper
# ─────────────────────────────────────────────────────────────────────────────


async def probe_ollama_tool_support(model_id: str, base_url: str = "http://localhost:11434") -> bool:
    """
    Query Ollama's /api/show endpoint to check if the model template
    contains tool-calling markup.  This is the most reliable way to detect
    runtime tool support without actually calling the model.

    Returns True if tools appear supported, False otherwise.
    Falls back to pattern-matching on failure.
    """
    import urllib.request
    import json

    # Fast path: check pattern registry first
    known = get_capabilities("ollama", model_id)
    # If we have an explicit non-default entry, trust it
    if ("ollama", model_id.lower()) in _RUNTIME_REGISTRY:
        return known.supports_tools

    try:
        body = json.dumps({"model": model_id}).encode()
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/api/show",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())

        template: str = data.get("template", "") or ""
        system:   str = data.get("system",   "") or ""
        combined = (template + system).lower()

        # Tool-capable templates reference tool tokens or function schemas
        tool_indicators = [
            "{{- range .tools}}",
            "{% for tool in tools %}",
            "<tool_call>",
            "<tools>",
            "[tool_calls]",
            "tool_calls",
            "function_calls",
            '"type": "function"',
            "available_tools",
        ]
        supports = any(ind.lower() in combined for ind in tool_indicators)
        return supports

    except (OSError, RuntimeError, ValueError, AttributeError):
        # If we can't probe, fall back to static pattern matching
        return known.supports_tools
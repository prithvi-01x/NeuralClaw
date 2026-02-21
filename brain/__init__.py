"""
brain/__init__.py — NeuralClaw LLM Brain
"""

from __future__ import annotations

from typing import Optional
import os

from brain.llm_client import (
    BaseLLMClient,
    LLMConnectionError,
    LLMContextError,
    LLMError,
    LLMInvalidRequestError,
    LLMRateLimitError,
)
from brain.types import (
    FinishReason,
    LLMConfig,
    LLMResponse,
    Message,
    Provider,
    Role,
    TokenUsage,
    ToolCall,
    ToolResult,
    ToolSchema,
)

__all__ = [
    "LLMClientFactory",
    "BaseLLMClient",
    "LLMError",
    "LLMConnectionError",
    "LLMRateLimitError",
    "LLMContextError",
    "LLMInvalidRequestError",
    "Message",
    "LLMConfig",
    "LLMResponse",
    "ToolCall",
    "ToolResult",
    "ToolSchema",
    "TokenUsage",
    "Role",
    "Provider",
    "FinishReason",
]

# Default models per provider
_DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o",
    "anthropic": "claude-3-5-sonnet-20241022",
    "ollama": "llama3.1",
    "openrouter": "openai/gpt-4o",
    "gemini": "gemini-1.5-pro",
    "bytez": "openai/gpt-5",  # 🟢 NEW
}


class LLMClientFactory:

    @staticmethod
    def create(
        provider: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ) -> BaseLLMClient:

        provider = provider.lower().strip()

        # ───────── OPENAI
        if provider == "openai":
            if not api_key:
                raise LLMConnectionError("OPENAI_API_KEY is required", provider="openai")
            from brain.openai_client import OpenAIClient
            return OpenAIClient(api_key=api_key, base_url=base_url, **kwargs)

        # ───────── ANTHROPIC
        elif provider == "anthropic":
            if not api_key:
                raise LLMConnectionError("ANTHROPIC_API_KEY is required", provider="anthropic")
            from brain.anthropic_client import AnthropicClient
            return AnthropicClient(api_key=api_key, base_url=base_url)

        # ───────── OLLAMA
        elif provider == "ollama":
            from brain.ollama_client import OllamaClient
            return OllamaClient(base_url=base_url or "http://localhost:11434/v1")

        # ───────── OPENROUTER
        elif provider == "openrouter":
            if not api_key:
                raise LLMConnectionError("OPENROUTER_API_KEY is required", provider="openrouter")
            from brain.openrouter_client import OpenRouterClient
            return OpenRouterClient(
                api_key=api_key,
                app_name=kwargs.get("app_name", "NeuralClaw"),
                site_url=kwargs.get("site_url", "https://github.com/neuralclaw"),
            )

        # ───────── GEMINI
        elif provider == "gemini":
            if not api_key:
                raise LLMConnectionError("GEMINI_API_KEY is required", provider="gemini")
            from brain.gemini_client import GeminiClient
            return GeminiClient(api_key=api_key)

        # ───────── 🟢 BYTEZ (NEW)
        elif provider == "bytez":
            api_key = api_key or os.getenv("BYTEZ_API_KEY")
            if not api_key:
                raise LLMConnectionError("BYTEZ_API_KEY is required", provider="bytez")

            from brain.bytez_client import BytezClient
            return BytezClient(api_key=api_key)

        else:
            raise ValueError(
                f"Unknown LLM provider: '{provider}'. "
                f"Valid options: openai, anthropic, ollama, openrouter, gemini, bytez"
            )

    @staticmethod
    def from_settings(settings) -> BaseLLMClient:

        provider = settings.default_llm_provider

        api_key_map = {
            "openai": settings.openai_api_key,
            "anthropic": settings.anthropic_api_key,
            "ollama": None,
            "openrouter": getattr(settings, "openrouter_api_key", None),
            "gemini": getattr(settings, "gemini_api_key", None),
            "bytez": os.getenv("BYTEZ_API_KEY"),  # 🟢 NEW
        }

        base_url_map = {
            "ollama": settings.ollama_base_url + "/v1",
        }

        return LLMClientFactory.create(
            provider=provider,
            api_key=api_key_map.get(provider),
            base_url=base_url_map.get(provider),
        )

    @staticmethod
    def default_model(provider: str) -> str:
        return _DEFAULT_MODELS.get(provider.lower(), "gpt-4o")

"""
brain/__init__.py — NeuralClaw LLM Brain

Public interface for the brain module.

Usage:
    from brain import LLMClientFactory, LLMConfig, Message, Role

    client = LLMClientFactory.from_settings(settings)
    # or
    client = LLMClientFactory.create("openai", api_key="sk-...")

    response = await client.generate(
        messages=[Message.user("Hello!")],
        config=LLMConfig(model="gpt-4o"),
    )
    print(response.content)
"""

from __future__ import annotations

from typing import Optional

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
    # Factory
    "LLMClientFactory",
    # Base client + exceptions
    "BaseLLMClient",
    "LLMError",
    "LLMConnectionError",
    "LLMRateLimitError",
    "LLMContextError",
    "LLMInvalidRequestError",
    # Types
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
}


class LLMClientFactory:
    """
    Creates and configures LLM clients.

    Preferred usage — create from settings object:
        client = LLMClientFactory.from_settings(settings)

    Manual usage:
        client = LLMClientFactory.create(
            provider="anthropic",
            api_key="sk-ant-...",
        )
    """

    @staticmethod
    def create(
        provider: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ) -> BaseLLMClient:
        """
        Instantiate an LLM client for the given provider.

        Args:
            provider:  One of: openai | anthropic | ollama | openrouter | gemini
            api_key:   API key (not required for ollama)
            base_url:  Override the default API endpoint
            **kwargs:  Provider-specific extras (e.g. organization for OpenAI)

        Returns:
            A configured BaseLLMClient subclass.

        Raises:
            ValueError: Unknown provider name.
            LLMConnectionError: Missing API key for a cloud provider.
        """
        provider = provider.lower().strip()

        if provider == "openai":
            if not api_key:
                raise LLMConnectionError("OPENAI_API_KEY is required", provider="openai")
            from brain.openai_client import OpenAIClient
            return OpenAIClient(api_key=api_key, base_url=base_url, **kwargs)

        elif provider == "anthropic":
            if not api_key:
                raise LLMConnectionError("ANTHROPIC_API_KEY is required", provider="anthropic")
            from brain.anthropic_client import AnthropicClient
            return AnthropicClient(api_key=api_key, base_url=base_url)

        elif provider == "ollama":
            from brain.ollama_client import OllamaClient
            return OllamaClient(base_url=base_url or "http://localhost:11434/v1")

        elif provider == "openrouter":
            if not api_key:
                raise LLMConnectionError("OPENROUTER_API_KEY is required", provider="openrouter")
            from brain.openrouter_client import OpenRouterClient
            return OpenRouterClient(
                api_key=api_key,
                app_name=kwargs.get("app_name", "NeuralClaw"),
                site_url=kwargs.get("site_url", "https://github.com/neuralclaw"),
            )

        elif provider == "gemini":
            if not api_key:
                raise LLMConnectionError("GEMINI_API_KEY is required", provider="gemini")
            from brain.gemini_client import GeminiClient
            return GeminiClient(api_key=api_key)

        else:
            raise ValueError(
                f"Unknown LLM provider: '{provider}'. "
                f"Valid options: openai, anthropic, ollama, openrouter, gemini"
            )

    @staticmethod
    def from_settings(settings) -> BaseLLMClient:
        """
        Create an LLM client from the NeuralClaw Settings object.

        Reads provider, model, and API keys from settings + environment.
        This is the preferred way to create a client in production code.

        Args:
            settings: NeuralClaw Settings instance from config.settings

        Returns:
            Configured BaseLLMClient for the default provider.
        """
        provider = settings.default_llm_provider

        # Map provider → API key from settings
        api_key_map = {
            "openai": settings.openai_api_key,
            "anthropic": settings.anthropic_api_key,
            "ollama": None,  # no key needed
            "openrouter": getattr(settings, "openrouter_api_key", None),
            "gemini": getattr(settings, "gemini_api_key", None),
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
        """Return the recommended default model for a provider."""
        return _DEFAULT_MODELS.get(provider.lower(), "gpt-4o")
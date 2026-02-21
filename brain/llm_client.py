"""
brain/llm_client.py — Abstract LLM Client

All provider implementations (OpenAI, Anthropic, Ollama, OpenRouter, Gemini, Bytez)
must subclass BaseLLMClient and implement generate().

The agent orchestrator only ever calls generate() — it never knows which
provider is underneath.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from brain.types import LLMConfig, LLMResponse, Message, ToolSchema, Provider


class BaseLLMClient(ABC):
    """
    Abstract base for all LLM provider clients.

    Subclasses must implement:
      - generate()     → call the LLM, return normalised LLMResponse
      - health_check() → verify connectivity to the provider
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        config: LLMConfig,
        tools: Optional[list[ToolSchema]] = None,
    ) -> LLMResponse:
        """
        Call the LLM and return a normalised response.
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the provider is reachable and the API key is valid."""
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"


# ─────────────────────────────────────────────────────────────────────────────
# Client Factory (NEW — supports Bytez)
# ─────────────────────────────────────────────────────────────────────────────

def create_llm_client(
    provider: Provider,
    api_key: Optional[str],
    base_url: Optional[str] = None,
):
    """
    Central factory that returns correct client instance.
    """

    if provider == Provider.OPENAI:
        from brain.openai_client import OpenAIClient
        return OpenAIClient(api_key=api_key, base_url=base_url)

    elif provider == Provider.ANTHROPIC:
        from brain.anthropic_client import AnthropicClient
        return AnthropicClient(api_key=api_key)

    elif provider == Provider.OLLAMA:
        from brain.ollama_client import OllamaClient
        return OllamaClient()

    elif provider == Provider.OPENROUTER:
        from brain.openrouter_client import OpenRouterClient
        return OpenRouterClient(api_key=api_key)

    elif provider == Provider.GEMINI:
        from brain.gemini_client import GeminiClient
        return GeminiClient(api_key=api_key)

    # 🟢 NEW: BYTEZ SUPPORT
    elif provider == Provider.BYTEZ:
        from brain.bytez_client import BytezClient
        return BytezClient(api_key=api_key)

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


# ─────────────────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────────────────


class LLMError(Exception):
    """Base exception for all LLM client errors."""
    def __init__(self, message: str, provider: str = "", status_code: Optional[int] = None):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


class LLMConnectionError(LLMError):
    """Provider unreachable or authentication failed."""


class LLMRateLimitError(LLMError):
    """Rate limit hit — caller should implement exponential backoff."""
    def __init__(self, message: str, provider: str = "", retry_after: Optional[float] = None):
        super().__init__(message, provider)
        self.retry_after = retry_after


class LLMContextError(LLMError):
    """Input exceeds model context window."""


class LLMInvalidRequestError(LLMError):
    """Bad request — invalid parameters or schema."""

"""
brain/llm_client.py — Abstract LLM Client

All provider implementations (OpenAI, Anthropic, Ollama, OpenRouter, Gemini)
must subclass BaseLLMClient and implement generate().

The agent orchestrator only ever calls generate() — it never knows which
provider is underneath.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from brain.types import LLMConfig, LLMResponse, Message, ToolSchema


class BaseLLMClient(ABC):
    """
    Abstract base for all LLM provider clients.

    Subclasses must implement:
      - generate()     → call the LLM, return normalised LLMResponse
      - health_check() → verify connectivity to the provider

    Subclasses should also implement:
      - _to_provider_messages() → translate Message list to provider format
      - _to_provider_tools()    → translate ToolSchema list to provider format
      - _from_provider_response() → translate raw API response to LLMResponse
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

        Args:
            messages:  Full conversation history (system + user + assistant turns).
            config:    Model, temperature, max_tokens, etc.
            tools:     Optional list of tools the LLM may call.

        Returns:
            LLMResponse with content and/or tool_calls populated.

        Raises:
            LLMConnectionError: Provider unreachable or auth failed.
            LLMRateLimitError:  Hit rate limit — caller should back off.
            LLMContextError:    Input too long for the model's context window.
            LLMError:           Any other provider error.
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the provider is reachable and the API key is valid."""
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"


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
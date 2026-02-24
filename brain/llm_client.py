"""
brain/llm_client.py — Abstract LLM Client + Retry/Failover

All provider implementations (OpenAI, Anthropic, Ollama, OpenRouter, Gemini, Bytez)
must subclass BaseLLMClient and implement generate().

New in this version:
  - BaseLLMClient.supports_tools class flag (False on Bytez)
  - _call_with_retry() — exponential backoff on transient errors
  - ResilientLLMClient — wraps any client with retry + optional provider failover.
    If the primary provider exhausts all retries, each fallback is tried in order.
"""

from __future__ import annotations

import asyncio
import random
from abc import ABC, abstractmethod
from typing import Optional

from brain.types import LLMConfig, LLMResponse, Message, ToolSchema, Provider


class BaseLLMClient(ABC):
    """
    Abstract base for all LLM provider clients.

    Subclasses must implement:
      - generate()     -> call the LLM, return normalised LLMResponse
      - health_check() -> verify connectivity to the provider

    Class attributes:
      - supports_tools: set False on providers that don't support function calling
        (e.g. Bytez). The orchestrator checks this before building tool schemas.
    """

    supports_tools: bool = True

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
        """Call the LLM and return a normalised response."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the provider is reachable and the API key is valid."""
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"


# ─────────────────────────────────────────────────────────────────────────────
# Retry logic
# ─────────────────────────────────────────────────────────────────────────────


async def _call_with_retry(
    client: BaseLLMClient,
    messages: list[Message],
    config: LLMConfig,
    tools: Optional[list[ToolSchema]],
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
) -> LLMResponse:
    """
    Call client.generate() with exponential backoff on transient errors.

    Retries on:
      - LLMConnectionError  (network blip, timeout, 5xx)
      - LLMRateLimitError   (429 / quota exceeded)

    Does NOT retry (permanent — won't fix themselves):
      - LLMContextError          (input too long)
      - LLMInvalidRequestError   (bad params / unsupported feature)
      - Other LLMError subclasses

    Backoff formula: min(base_delay * 2^attempt + jitter, max_delay)
    If LLMRateLimitError carries retry_after, that value is used instead.
    """
    last_error: Exception | None = None

    for attempt in range(max_attempts):
        try:
            return await client.generate(messages=messages, config=config, tools=tools)

        except (LLMConnectionError, LLMRateLimitError) as e:
            last_error = e

            if attempt == max_attempts - 1:
                break  # exhausted retries

            # Respect Retry-After header if provided
            if isinstance(e, LLMRateLimitError) and e.retry_after:
                delay = min(e.retry_after, max_delay)
            else:
                jitter = random.uniform(0, 0.5)
                delay = min(base_delay * (2 ** attempt) + jitter, max_delay)

            from observability.logger import get_logger
            _log = get_logger("brain.retry")
            _log.warning(
                "llm.retrying",
                attempt=attempt + 1,
                max_attempts=max_attempts,
                delay_s=round(delay, 2),
                error=str(e),
                error_type=type(e).__name__,
            )
            await asyncio.sleep(delay)

        except (LLMContextError, LLMInvalidRequestError, LLMError):
            raise  # permanent — propagate immediately

    raise last_error  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# ResilientLLMClient — retry + optional provider failover
# ─────────────────────────────────────────────────────────────────────────────


class ResilientLLMClient(BaseLLMClient):
    """
    Wraps a primary LLM client with automatic retry and optional provider failover.

    Behaviour:
      1. Calls primary client with up to max_attempts retries (exponential backoff).
      2. If primary exhausts all retries, tries each fallback client in order,
         each also with max_attempts retries.
      3. Permanent errors (context overflow, invalid request) skip failover
         immediately — they won't be fixed by a different provider.

    The supports_tools flag mirrors the primary client. If failing over to a
    provider that supports tools when the primary doesn't (or vice versa), the
    orchestrator's pre-check handles it — ResilientLLMClient doesn't re-check.

    Usage:
        primary  = LLMClientFactory.create("bytez", api_key=...)
        fallback = LLMClientFactory.create("ollama")

        client = ResilientLLMClient(primary=primary, fallbacks=[fallback])
        response = await client.generate(messages, config)
    """

    def __init__(
        self,
        primary: BaseLLMClient,
        fallbacks: Optional[list[BaseLLMClient]] = None,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
    ):
        super().__init__()
        self._primary = primary
        self._fallbacks = fallbacks or []
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._max_delay = max_delay
        # Mirror the primary's tool support so the orchestrator's pre-check works
        self.supports_tools = getattr(primary, "supports_tools", True)
        # Tracks the client that last produced a successful response (for health_check)
        self._active_client: BaseLLMClient = primary

    @property
    def primary(self) -> BaseLLMClient:
        return self._primary

    async def generate(
        self,
        messages: list[Message],
        config: LLMConfig,
        tools: Optional[list[ToolSchema]] = None,
    ) -> LLMResponse:
        from observability.logger import get_logger
        log = get_logger("brain.resilient")

        all_clients = [self._primary] + self._fallbacks
        last_error: Exception | None = None

        for i, client in enumerate(all_clients):
            if i > 0:
                log.warning(
                    "llm.failing_over",
                    from_client=repr(all_clients[i - 1]),
                    to_client=repr(client),
                    reason=str(last_error),
                )
                # Mirror the active client's tool support flag so the
                # orchestrator's pre-check reflects the actual provider in use.
                self.supports_tools = getattr(client, "supports_tools", True)

            try:
                result = await _call_with_retry(
                    client=client,
                    messages=messages,
                    config=config,
                    tools=tools,
                    max_attempts=self._max_attempts,
                    base_delay=self._base_delay,
                    max_delay=self._max_delay,
                )
                self._active_client = client  # remember who succeeded
                return result
            except (LLMContextError, LLMInvalidRequestError):
                raise  # permanent — no point trying fallbacks
            except (LLMConnectionError, LLMRateLimitError, LLMError) as e:
                last_error = e
                log.error(
                    "llm.client_exhausted",
                    client=repr(client),
                    error=str(e),
                    will_try_fallback=i < len(all_clients) - 1,
                )
                # Continue to next fallback

        raise LLMError(
            f"All LLM clients failed. Last error: {last_error}",
            provider="all",
        )

    async def health_check(self) -> bool:
        """Ping the currently-active client (may be a fallback after failover)."""
        return await self._active_client.health_check()

    def __repr__(self) -> str:
        n = len(self._fallbacks)
        suffix = f" + {n} fallback(s)" if n else ""
        return f"<ResilientLLMClient primary={self._primary!r}{suffix}>"


# ─────────────────────────────────────────────────────────────────────────────
# Client Factory
# ─────────────────────────────────────────────────────────────────────────────

def create_llm_client(
    provider: Provider,
    api_key: Optional[str],
    base_url: Optional[str] = None,
):
    """Central factory that returns the correct client instance."""
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
    """Rate limit hit — retry with exponential backoff."""
    def __init__(self, message: str, provider: str = "", retry_after: Optional[float] = None):
        super().__init__(message, provider)
        self.retry_after = retry_after


class LLMContextError(LLMError):
    """Input exceeds model context window."""


class LLMInvalidRequestError(LLMError):
    """Bad request — invalid parameters or unsupported feature."""
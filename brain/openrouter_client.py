"""
brain/openrouter_client.py — OpenRouter LLM Client

OpenRouter is a unified API gateway for 100+ models (GPT-4, Claude, Llama, Gemini, etc.).
It uses an OpenAI-compatible API so we reuse OpenAIClient, just pointing at
api.openrouter.ai with the required extra headers.

Popular OpenRouter models:
  - openai/gpt-4o
  - anthropic/claude-3.5-sonnet
  - meta-llama/llama-3.1-70b-instruct
  - google/gemini-pro-1.5
  - mistralai/mixtral-8x22b-instruct
"""

from __future__ import annotations

from typing import Optional

from openai import AsyncOpenAI

from brain.llm_client import BaseLLMClient, LLMConnectionError, LLMError
from brain.openai_client import OpenAIClient
from brain.types import LLMConfig, LLMResponse, Message, Provider, ToolSchema
from observability.logger import get_logger

log = get_logger(__name__)

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient(BaseLLMClient):
    """
    OpenRouter client — routes requests to any of 100+ models via one API.

    Requires an OpenRouter API key (https://openrouter.ai/keys).
    Set app_name/site_url for OpenRouter's usage analytics dashboard.
    """

    def __init__(
        self,
        api_key: str,
        app_name: str = "NeuralClaw",
        site_url: str = "https://github.com/neuralclaw",
    ):
        super().__init__(api_key=api_key, base_url=_OPENROUTER_BASE_URL)
        self._app_name = app_name
        self._site_url = site_url

        # OpenRouter needs extra headers for attribution
        self._inner = OpenAIClient(
            api_key=api_key,
            base_url=_OPENROUTER_BASE_URL,
        )
        # Patch the underlying AsyncOpenAI client with OR-required headers
        self._inner._client = AsyncOpenAI(
            api_key=api_key,
            base_url=_OPENROUTER_BASE_URL,
            default_headers={
                "HTTP-Referer": site_url,
                "X-Title": app_name,
            },
        )

    async def generate(
        self,
        messages: list[Message],
        config: LLMConfig,
        tools: Optional[list[ToolSchema]] = None,
    ) -> LLMResponse:
        log.debug("openrouter.generate.start", model=config.model)
        try:
            result = await self._inner.generate(messages, config, tools)
            result.provider = Provider.OPENROUTER
            return result
        except LLMConnectionError:
            raise LLMConnectionError(
                "Cannot reach OpenRouter API. Check your API key and network.",
                provider="openrouter",
            )
        except LLMError:
            raise

    async def health_check(self) -> bool:
        try:
            await self._inner._client.models.list()
            return True
        except Exception as e:
            log.warning("openrouter.health_check.failed", error=str(e))
            return False
"""
brain/ollama_client.py — Ollama Local LLM Client

Supports any model running in Ollama (llama3, mistral, mixtral, gemma, etc.).
Uses the OpenAI-compatible endpoint Ollama exposes at /v1/ — so we reuse
the OpenAI SDK but point it at localhost.

Tool calling support depends on the model — llama3.1+, mistral-nemo, etc.
Models that don't support tool calling will return content only.
"""

from __future__ import annotations

from typing import Optional

import httpx
import openai
from openai import AsyncOpenAI

from brain.llm_client import BaseLLMClient, LLMConnectionError, LLMError
from brain.openai_client import OpenAIClient
from brain.types import LLMConfig, LLMResponse, Message, Provider, ToolSchema
from observability.logger import get_logger

log = get_logger(__name__)

_DEFAULT_BASE_URL = "http://localhost:11434/v1"


class OllamaClient(BaseLLMClient):
    """
    Ollama client — runs local models via Ollama's OpenAI-compatible API.

    No API key required. Requires Ollama to be running locally.
    Set base_url if Ollama is on a non-standard host/port.
    """

    def __init__(self, base_url: str = _DEFAULT_BASE_URL):
        super().__init__(api_key="ollama", base_url=base_url)
        # Reuse OpenAI client pointed at Ollama's /v1 endpoint
        self._inner = OpenAIClient(api_key="ollama", base_url=base_url)
        # Keep a raw client for health check (models list)
        self._raw_client = AsyncOpenAI(api_key="ollama", base_url=base_url)

    async def generate(
        self,
        messages: list[Message],
        config: LLMConfig,
        tools: Optional[list[ToolSchema]] = None,
    ) -> LLMResponse:
        log.debug("ollama.generate.start", model=config.model)
        try:
            result = await self._inner.generate(messages, config, tools)
            # Override provider tag
            result.provider = Provider.OLLAMA
            return result
        except LLMConnectionError:
            raise LLMConnectionError(
                f"Cannot reach Ollama at {self.base_url}. Is `ollama serve` running?",
                provider="ollama",
            )
        except LLMError:
            raise

    async def health_check(self) -> bool:
        """Check if Ollama is running and reachable."""
        try:
            models = await self._raw_client.models.list()
            model_names = [m.id for m in models.data]
            log.debug("ollama.health_check.ok", available_models=model_names)
            return True
        except Exception as e:
            log.warning("ollama.health_check.failed", error=str(e))
            return False

    async def list_models(self) -> list[str]:
        """Return names of all models available in Ollama."""
        try:
            models = await self._raw_client.models.list()
            return [m.id for m in models.data]
        except Exception as e:
            log.warning("ollama.list_models.failed", error=str(e))
            return []
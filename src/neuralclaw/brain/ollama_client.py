"""
brain/ollama_client.py — Ollama Local LLM Client

Supports any model running in Ollama (llama3, mistral, mixtral, gemma, etc.).
Uses the OpenAI-compatible endpoint Ollama exposes at /v1/ — so we reuse
the OpenAI SDK but point it at localhost.

Tool calling support is determined dynamically per model_id:
  1. Checks brain.capabilities registry (static pattern table + live overrides)
  2. On first use of a model, refreshes supports_tools on self so the orchestrator
     reads the correct value before building tool schemas
  3. If a "does not support tools" error arrives at runtime, registers the model
     as no-tools in the capability registry and retries without tools — no crash.
"""

from __future__ import annotations

from typing import Optional

import httpx
import openai
from openai import AsyncOpenAI

from neuralclaw.brain.llm_client import BaseLLMClient, LLMConnectionError, LLMError
from neuralclaw.brain.openai_client import OpenAIClient
from neuralclaw.brain.types import LLMConfig, LLMResponse, Message, Provider, ToolSchema
from neuralclaw.observability.logger import get_logger

log = get_logger(__name__)

_DEFAULT_BASE_URL = "http://localhost:11434/v1"

# Phrases that indicate the model rejected tool schemas
_TOOL_ERROR_PHRASES = (
    "does not support tools",
    "does not support tool",
    "tool_use is not supported",
    "function calling is not supported",
    "unsupported parameter: tools",
    "tools is not supported",
)


class OllamaClient(BaseLLMClient):
    """
    Ollama client — runs local models via Ollama's OpenAI-compatible API.

    No API key required. Requires Ollama to be running locally.
    Set base_url if Ollama is on a non-standard host/port.

    supports_tools is determined dynamically per model_id via the capability
    registry (brain.capabilities).  The orchestrator reads self.supports_tools
    before building tool schemas, so switching models updates this flag.
    """

    # Conservative default for the class — overridden per-instance per model
    supports_tools: bool = False

    def __init__(self, base_url: str = _DEFAULT_BASE_URL):
        super().__init__(api_key="ollama", base_url=base_url)
        self._inner = OpenAIClient(api_key="ollama", base_url=base_url)
        self._raw_client = AsyncOpenAI(api_key="ollama", base_url=base_url)
        self._last_model: str = ""
        self.supports_tools: bool = False    # instance attribute shadows class default
        self.supports_vision: bool = False

    # ─────────────────────────────────────────────────────────────────────────
    # Capability management
    # ─────────────────────────────────────────────────────────────────────────

    async def _refresh_capabilities(self, model_id: str) -> None:
        """
        Update supports_tools / supports_vision from the capability registry
        whenever the active model changes.  Called at the start of generate().
        """
        if model_id == self._last_model:
            return
            
        from neuralclaw.brain.capabilities import get_capabilities, is_explicitly_known, probe_ollama_tool_support, register_capabilities
        
        # If we don't know the model, attempt to probe it dynamically before defaulting
        if not is_explicitly_known("ollama", model_id):
            try:
                supports_tools = await probe_ollama_tool_support(model_id, self.base_url)
                register_capabilities("ollama", model_id, supports_tools=supports_tools)
                log.info("ollama.capability_probe_success", model=model_id, supports_tools=supports_tools)
            except LLMError as e:
                log.warning("ollama.capability_probe_failed", model=model_id, error=str(e))
                # Will fall back to whatever get_capabilities returns below
            except (httpx.HTTPError, OSError) as e:
                log.warning("ollama.capability_probe_failed", model=model_id, error=str(e), error_type=type(e).__name__)
                # Will fall back to whatever get_capabilities returns below

        caps = get_capabilities("ollama", model_id)
        self.supports_tools  = caps.supports_tools
        self.supports_vision = caps.supports_vision
        self._last_model = model_id
        log.debug(
            "ollama.capabilities_updated",
            model=model_id,
            supports_tools=self.supports_tools,
            supports_vision=self.supports_vision,
        )

    def _mark_no_tools(self, model_id: str) -> None:
        """
        Permanently mark a model as no-tools in the registry after a runtime
        failure.  Subsequent generate() calls will never send tool schemas.
        """
        from neuralclaw.brain.capabilities import register_capabilities
        register_capabilities("ollama", model_id, supports_tools=False)
        self.supports_tools = False
        self._last_model = model_id
        log.warning(
            "ollama.model_marked_no_tools",
            model=model_id,
            reason="runtime tool rejection — registered permanently",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Core generate
    # ─────────────────────────────────────────────────────────────────────────

    async def generate(
        self,
        messages: list[Message],
        config: LLMConfig,
        tools: Optional[list[ToolSchema]] = None,
    ) -> LLMResponse:
        # Refresh capability flags whenever the model changes
        await self._refresh_capabilities(config.model)

        # Strip tools upfront if capabilities say this model doesn't support them
        effective_tools: Optional[list[ToolSchema]] = tools if self.supports_tools else None
        if tools and not self.supports_tools:
            log.debug(
                "ollama.tools_suppressed_preemptive",
                model=config.model,
                reason="capability registry: supports_tools=False",
            )

        log.debug("ollama.generate.start", model=config.model,
                  tools_requested=bool(tools), tools_sent=bool(effective_tools))

        try:
            result = await self._inner.generate(messages, config, effective_tools)
            result.provider = Provider.OLLAMA
            return result

        except LLMConnectionError:
            raise LLMConnectionError(
                f"Cannot reach Ollama at {self.base_url}. Is `ollama serve` running?",
                provider="ollama",
            )

        except LLMError as e:
            err_lower = str(e).lower()

            # Runtime tool rejection — model refused our schemas
            if effective_tools and any(p in err_lower for p in _TOOL_ERROR_PHRASES):
                log.warning(
                    "ollama.tools_rejected_runtime",
                    model=config.model,
                    error=str(e),
                )
                self._mark_no_tools(config.model)
                # Retry the exact same call without tools — transparent to caller
                result = await self._inner.generate(messages, config, None)
                result.provider = Provider.OLLAMA
                return result

            raise

    # ─────────────────────────────────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────────────────────────────────

    async def health_check(self) -> bool:
        """Check if Ollama is running and reachable."""
        try:
            models = await self._raw_client.models.list()
            model_names = [m.id for m in models.data]
            log.debug("ollama.health_check.ok", available_models=model_names)
            return True
        except (OSError, RuntimeError, AttributeError) as e:
            log.warning("ollama.health_check.failed", error=str(e), error_type=type(e).__name__)
            return False

    async def list_models(self) -> list[str]:
        """Return names of all models available in Ollama."""
        try:
            models = await self._raw_client.models.list()
            return [m.id for m in models.data]
        except (OSError, RuntimeError, AttributeError) as e:
            log.warning("ollama.list_models.failed", error=str(e), error_type=type(e).__name__)
            return []
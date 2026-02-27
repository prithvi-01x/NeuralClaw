"""
brain/bytez_client.py — Bytez SDK Client

Fixes applied:
  - Bug 1: model.run() is synchronous; now runs in a thread-pool executor
            so it never blocks the async event loop.
  - Bug 2: tools parameter was silently ignored; now raises a clear
            LLMInvalidRequestError so callers know tool-calling is
            unsupported with this provider, rather than the agent looping
            forever waiting for tool results that never arrive.
  - Bug 3 (NEW): results.output parsing was fragile. If the SDK returns a
            list of dicts but the role check fails (wrong key, casing, etc.)
            output_text stayed "" and fell back to str(results.output),
            printing raw {'role': 'assistant', 'content': '...'} to the UI.
            Now handles list-of-dicts, single dict, and plain string/other
            output shapes robustly.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from bytez import Bytez

from neuralclaw.brain.llm_client import (
    BaseLLMClient,
    LLMConnectionError,
    LLMError,
    LLMInvalidRequestError,
)
from neuralclaw.brain.types import (
    FinishReason,
    LLMConfig,
    LLMResponse,
    Message,
    Provider,
    Role,
    TokenUsage,
    ToolSchema,
)
from neuralclaw.observability.logger import get_logger

log = get_logger(__name__)

# One shared executor — Bytez SDK is CPU/IO bound, 2 workers is plenty
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="bytez")


def _extract_text_from_output(output) -> str:
    """
    Robustly extract assistant text from whatever Bytez SDK returns.

    Handles:
      - list of dicts  → [{"role": "assistant", "content": "..."}]
      - single dict    → {"role": "assistant", "content": "..."}
      - plain string   → "Hello!"
      - anything else  → str() fallback
    """
    if isinstance(output, list):
        parts = []
        for item in output:
            if not isinstance(item, dict):
                # Unexpected item type; stringify it
                parts.append(str(item))
                continue
            role = item.get("role", "assistant")
            content = item.get("content", "")
            # Accept assistant messages or any dict that has content
            # (some SDK versions omit the role key entirely)
            if role in ("assistant", "") or "role" not in item:
                if content:
                    parts.append(content)
            elif content:
                # Non-assistant role but has content — include anyway
                # so we don't silently discard a valid response
                parts.append(content)
        return "".join(parts)

    if isinstance(output, dict):
        # Single dict — just pull content
        return output.get("content", str(output))

    if output is not None:
        return str(output)

    return ""


class BytezClient(BaseLLMClient):
    # Bytez SDK does not support tool/function calling.
    # The orchestrator checks this flag before building tool schemas,
    # so tools are never passed to this provider.
    supports_tools: bool = False

    def __init__(self, api_key: str):
        super().__init__(api_key=api_key)
        self.sdk = Bytez(api_key)

    # ─────────────────────────────────────────
    # GENERATE
    # ─────────────────────────────────────────

    async def generate(
        self,
        messages: list[Message],
        config: LLMConfig,
        tools: Optional[list[ToolSchema]] = None,
    ) -> LLMResponse:
        # Bug 2 fix: Bytez SDK does not support tool/function calling.
        # Raise explicitly so the orchestrator surfaces a clear error
        # instead of silently discarding tools and looping forever.
        if tools:
            raise LLMInvalidRequestError(
                "Bytez provider does not support tool/function calling. "
                "Switch to openai, anthropic, or ollama for tool use.",
                provider="bytez",
            )

        input_msgs = [
            {"role": m.role.value, "content": m.content or ""}
            for m in messages
            if m.role in (Role.SYSTEM, Role.USER, Role.ASSISTANT)
        ]

        log.debug("bytez.generate.start", model=config.model, message_count=len(input_msgs))

        # Bug 1 fix: run the blocking SDK call in a thread-pool executor
        # so the async event loop is never blocked.
        try:
            loop = asyncio.get_running_loop()
            model_handle = self.sdk.model(config.model)
            results = await loop.run_in_executor(
                _executor,
                model_handle.run,
                input_msgs,
            )
        except (OSError, RuntimeError) as e:
            raise LLMConnectionError(str(e), provider="bytez") from e
        except BaseException as e:
            # Bytez sync SDK raises non-standard exceptions via run_in_executor
            raise LLMConnectionError(str(e), provider="bytez") from e

        if results.error:
            raise LLMError(str(results.error), provider="bytez")

        # Bug 3 fix: use robust output extractor instead of fragile inline logic
        output_text = _extract_text_from_output(results.output)

        if not output_text:
            log.warning("bytez.generate.empty_output", model=config.model,
                        raw_output=repr(results.output)[:200])

        log.debug("bytez.generate.complete", model=config.model, chars=len(output_text))

        return LLMResponse(
            content=output_text or "(no response)",
            tool_calls=[],
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(),
            model=config.model,
            provider=Provider.BYTEZ,
        )

    # ─────────────────────────────────────────
    # HEALTH CHECK
    # ─────────────────────────────────────────

    async def health_check(self) -> bool:
        # Also run the health-check probe in the executor (sync SDK call)
        try:
            loop = asyncio.get_running_loop()
            model_handle = self.sdk.model("openai/gpt-5")
            res = await loop.run_in_executor(
                _executor,
                model_handle.run,
                [{"role": "user", "content": "ping"}],
            )
            return res.error is None
        except (OSError, RuntimeError, AttributeError) as e:
            log.warning("bytez.health_check.failed", error=str(e), error_type=type(e).__name__)
            return False
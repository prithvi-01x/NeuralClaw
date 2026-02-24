"""
brain/openai_client.py — OpenAI LLM Client

Supports: GPT-4o, GPT-4-turbo, GPT-3.5-turbo, and any OpenAI-compatible endpoint.
Handles tool calling, token counting, and error normalisation.
"""

from __future__ import annotations

import json
from typing import Optional

import openai
from openai import AsyncOpenAI

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
    ToolSchema,
)
from observability.logger import get_logger

log = get_logger(__name__)


class OpenAIClient(BaseLLMClient):
    """
    OpenAI API client (also works with any OpenAI-compatible endpoint
    e.g. LiteLLM proxy, local vLLM, etc.).
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,     # None = official OpenAI endpoint
        organization: Optional[str] = None,
    ):
        super().__init__(api_key=api_key, base_url=base_url)
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    async def generate(
        self,
        messages: list[Message],
        config: LLMConfig,
        tools: Optional[list[ToolSchema]] = None,
    ) -> LLMResponse:
        oai_messages = self._to_provider_messages(messages)
        oai_tools = self._to_provider_tools(tools) if tools else openai.NOT_GIVEN

        log.debug(
            "openai.generate.start",
            model=config.model,
            message_count=len(messages),
            has_tools=bool(tools),
        )

        try:
            response = await self._client.chat.completions.create(
                model=config.model,
                messages=oai_messages,
                tools=oai_tools,
                tool_choice="auto" if tools else openai.NOT_GIVEN,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                timeout=config.timeout_seconds,
            )
        except openai.AuthenticationError as e:
            raise LLMConnectionError(str(e), provider="openai", status_code=401) from e
        except openai.RateLimitError as e:
            raise LLMRateLimitError(str(e), provider="openai") from e
        except openai.BadRequestError as e:
            if "context" in str(e).lower() or "too long" in str(e).lower():
                raise LLMContextError(str(e), provider="openai") from e
            raise LLMInvalidRequestError(str(e), provider="openai") from e
        except openai.APIConnectionError as e:
            raise LLMConnectionError(str(e), provider="openai") from e
        except openai.APIError as e:
            raise LLMError(str(e), provider="openai", status_code=getattr(e, "status_code", None)) from e

        result = self._from_provider_response(response)
        log.debug(
            "openai.generate.complete",
            model=result.model,
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            finish_reason=result.finish_reason,
            tool_calls=len(result.tool_calls),
        )
        return result

    async def health_check(self) -> bool:
        try:
            await self._client.models.list()
            return True
        except (OSError, RuntimeError, AttributeError) as e:
            log.warning("openai.health_check.failed", error=str(e), error_type=type(e).__name__)
            return False

    # ── Private helpers ───────────────────────────────────────────────────────

    def _to_provider_messages(self, messages: list[Message]) -> list[dict]:
        """Translate internal Message list → OpenAI chat message format."""
        result = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                result.append({"role": "system", "content": msg.content or ""})

            elif msg.role == Role.USER:
                result.append({"role": "user", "content": msg.content or ""})

            elif msg.role == Role.ASSISTANT:
                entry: dict = {"role": "assistant"}
                if msg.content:
                    entry["content"] = msg.content
                if msg.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                result.append(entry)

            elif msg.role == Role.TOOL and msg.tool_result:
                result.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_result.tool_call_id,
                    "content": msg.tool_result.content,
                })

        return result

    def _to_provider_tools(self, tools: list[ToolSchema]) -> list[dict]:
        """Translate internal ToolSchema list → OpenAI function tool format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

    def _from_provider_response(self, response) -> LLMResponse:
        """Translate OpenAI ChatCompletion → internal LLMResponse."""
        choice = response.choices[0]
        msg = choice.message

        # Normalise finish reason
        raw_reason = choice.finish_reason or "stop"
        finish_map = {
            "stop": FinishReason.STOP,
            "tool_calls": FinishReason.TOOL_CALLS,
            "length": FinishReason.LENGTH,
        }
        finish_reason = finish_map.get(raw_reason, FinishReason.STOP)

        # Extract tool calls
        tool_calls: list[ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {"_raw": tc.function.arguments}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))

        usage = TokenUsage(
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )

        return LLMResponse(
            content=msg.content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            model=response.model,
            provider=Provider.OPENAI,
        )
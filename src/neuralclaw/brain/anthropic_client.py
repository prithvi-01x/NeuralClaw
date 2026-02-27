"""
brain/anthropic_client.py — Anthropic LLM Client

Supports: Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku, and future models.
Handles Anthropic's distinct message format, system prompt separation,
tool calling schema, and error normalisation.
"""

from __future__ import annotations

import json
from typing import Optional

import anthropic
from anthropic import AsyncAnthropic

from neuralclaw.brain.llm_client import (
    BaseLLMClient,
    LLMConnectionError,
    LLMContextError,
    LLMError,
    LLMInvalidRequestError,
    LLMRateLimitError,
)
from neuralclaw.brain.types import (
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
from neuralclaw.observability.logger import get_logger

log = get_logger(__name__)


class AnthropicClient(BaseLLMClient):
    """
    Anthropic Claude API client.

    Key differences from OpenAI format:
    - System prompt is a separate top-level param, not a message
    - Tool results use a distinct content block format
    - Finish reasons use Anthropic-specific strings
    """

    def __init__(self, api_key: str, base_url: Optional[str] = None):
        super().__init__(api_key=api_key, base_url=base_url)
        self._client = AsyncAnthropic(
            api_key=api_key,
            base_url=base_url,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    async def generate(
        self,
        messages: list[Message],
        config: LLMConfig,
        tools: Optional[list[ToolSchema]] = None,
    ) -> LLMResponse:
        system_prompt, ant_messages = self._to_provider_messages(messages)
        ant_tools = self._to_provider_tools(tools) if tools else anthropic.NOT_GIVEN

        log.debug(
            "anthropic.generate.start",
            model=config.model,
            message_count=len(messages),
            has_tools=bool(tools),
            has_system=bool(system_prompt),
        )

        try:
            response = await self._client.messages.create(
                model=config.model,
                system=system_prompt or anthropic.NOT_GIVEN,
                messages=ant_messages,
                tools=ant_tools,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                timeout=config.timeout_seconds,
            )
        except anthropic.AuthenticationError as e:
            raise LLMConnectionError(str(e), provider="anthropic", status_code=401) from e
        except anthropic.RateLimitError as e:
            raise LLMRateLimitError(str(e), provider="anthropic") from e
        except anthropic.BadRequestError as e:
            msg_str = str(e)
            if "too long" in msg_str.lower() or "context" in msg_str.lower():
                raise LLMContextError(msg_str, provider="anthropic") from e
            raise LLMInvalidRequestError(msg_str, provider="anthropic") from e
        except anthropic.APIConnectionError as e:
            raise LLMConnectionError(str(e), provider="anthropic") from e
        except anthropic.APIError as e:
            raise LLMError(str(e), provider="anthropic", status_code=getattr(e, "status_code", None)) from e

        result = self._from_provider_response(response)
        log.debug(
            "anthropic.generate.complete",
            model=result.model,
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            finish_reason=result.finish_reason,
            tool_calls=len(result.tool_calls),
        )
        return result

    async def health_check(self) -> bool:
        """
        Verify that the API key is valid without generating any tokens.

        Uses the models.list() endpoint instead of a messages.create() call so:
          - No tokens are consumed.
          - No hardcoded model string that could become deprecated.
        """
        try:
            await self._client.models.list()
            return True
        except anthropic.AuthenticationError:
            return False
        except (OSError, RuntimeError, AttributeError) as e:
            log.warning("anthropic.health_check.failed", error=str(e), error_type=type(e).__name__)
            return False

    # ── Private helpers ───────────────────────────────────────────────────────

    def _to_provider_messages(
        self, messages: list[Message]
    ) -> tuple[Optional[str], list[dict]]:
        """
        Translate internal Message list → Anthropic format.

        Returns (system_prompt, messages_list).
        Anthropic requires system prompt as a separate parameter.
        """
        system_prompt: Optional[str] = None
        result: list[dict] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                # Anthropic takes system as a top-level param — collect and join
                system_prompt = (system_prompt + "\n\n" + msg.content) if system_prompt else msg.content

            elif msg.role == Role.USER:
                result.append({"role": "user", "content": msg.content or ""})

            elif msg.role == Role.ASSISTANT:
                content_blocks = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        content_blocks.append({
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        })
                result.append({"role": "assistant", "content": content_blocks or msg.content or ""})

            elif msg.role == Role.TOOL and msg.tool_result:
                # Tool results must be in a user message with tool_result content blocks
                result.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_result.tool_call_id,
                            "content": msg.tool_result.content,
                            "is_error": msg.tool_result.is_error,
                        }
                    ],
                })

        return system_prompt, result

    def _to_provider_tools(self, tools: list[ToolSchema]) -> list[dict]:
        """Translate internal ToolSchema list → Anthropic tool format."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            }
            for t in tools
        ]

    def _from_provider_response(self, response) -> LLMResponse:
        """Translate Anthropic Message response → internal LLMResponse."""
        # Anthropic stop reasons
        stop_reason_map = {
            "end_turn": FinishReason.STOP,
            "tool_use": FinishReason.TOOL_CALLS,
            "max_tokens": FinishReason.LENGTH,
        }
        finish_reason = stop_reason_map.get(response.stop_reason or "end_turn", FinishReason.STOP)

        text_content: Optional[str] = None
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                text_content = block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                ))

        usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

        return LLMResponse(
            content=text_content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            model=response.model,
            provider=Provider.ANTHROPIC,
        )
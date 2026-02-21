"""
brain/gemini_client.py — Google Gemini LLM Client

Supports: gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash, etc.
Uses the NEW `google-genai` SDK (google.genai), NOT the deprecated
`google-generativeai` package.

Install: pip install google-genai
Get key: https://aistudio.google.com/app/apikey
"""

from __future__ import annotations

import uuid
from typing import Optional

from google import genai
from google.genai import types as genai_types

from brain.llm_client import (
    BaseLLMClient,
    LLMConnectionError,
    LLMContextError,
    LLMError,
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


class GeminiClient(BaseLLMClient):
    """
    Google Gemini API client using the new google-genai SDK.

    Requires: pip install google-genai
    """

    def __init__(self, api_key: str):
        super().__init__(api_key=api_key)
        self._client = genai.Client(api_key=api_key)

    async def generate(
        self,
        messages: list[Message],
        config: LLMConfig,
        tools: Optional[list[ToolSchema]] = None,
    ) -> LLMResponse:
        system_instruction, contents = self._to_provider_messages(messages)
        gemini_tools = self._to_provider_tools(tools) if tools else None

        log.debug(
            "gemini.generate.start",
            model=config.model,
            message_count=len(messages),
            has_tools=bool(tools),
        )

        gen_config = genai_types.GenerateContentConfig(
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
            top_p=config.top_p,
            system_instruction=system_instruction,
            tools=gemini_tools,
        )

        try:
            response = await self._client.aio.models.generate_content(
                model=config.model,
                contents=contents,
                config=gen_config,
            )
        except Exception as e:
            self._raise_normalised(e)

        result = self._from_provider_response(response, config.model)
        log.debug(
            "gemini.generate.complete",
            model=result.model,
            finish_reason=result.finish_reason,
            tool_calls=len(result.tool_calls),
        )
        return result

    async def health_check(self) -> bool:
        try:
            models = self._client.models.list()
            return any(True for _ in models)
        except Exception as e:
            log.warning("gemini.health_check.failed", error=str(e))
            return False

    # ── Private helpers ───────────────────────────────────────────────────────

    def _to_provider_messages(
        self, messages: list[Message]
    ) -> tuple[Optional[str], list[genai_types.Content]]:
        """Translate internal Message list → Gemini Contents + system instruction."""
        system_instruction: Optional[str] = None
        contents: list[genai_types.Content] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_instruction = (
                    system_instruction + "\n\n" + msg.content
                ) if system_instruction else msg.content

            elif msg.role == Role.USER:
                contents.append(genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text=msg.content or "")],
                ))

            elif msg.role == Role.ASSISTANT:
                parts = []
                if msg.content:
                    parts.append(genai_types.Part(text=msg.content))
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        parts.append(genai_types.Part(
                            function_call=genai_types.FunctionCall(
                                name=tc.name,
                                args=tc.arguments,
                            )
                        ))
                contents.append(genai_types.Content(role="model", parts=parts))

            elif msg.role == Role.TOOL and msg.tool_result:
                contents.append(genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(
                        function_response=genai_types.FunctionResponse(
                            name=msg.tool_result.name,
                            response={"result": msg.tool_result.content},
                        )
                    )],
                ))

        return system_instruction, contents

    def _to_provider_tools(self, tools: list[ToolSchema]) -> list[genai_types.Tool]:
        """Translate ToolSchema list → Gemini FunctionDeclaration format."""
        type_map = {
            "string": genai_types.Type.STRING,
            "integer": genai_types.Type.INTEGER,
            "number": genai_types.Type.NUMBER,
            "boolean": genai_types.Type.BOOLEAN,
            "array": genai_types.Type.ARRAY,
            "object": genai_types.Type.OBJECT,
        }
        declarations = []
        for t in tools:
            properties = {}
            for prop_name, prop_def in t.parameters.get("properties", {}).items():
                gemini_type = type_map.get(prop_def.get("type", "string"), genai_types.Type.STRING)
                properties[prop_name] = genai_types.Schema(
                    type=gemini_type,
                    description=prop_def.get("description", ""),
                )
            declarations.append(genai_types.FunctionDeclaration(
                name=t.name,
                description=t.description,
                parameters=genai_types.Schema(
                    type=genai_types.Type.OBJECT,
                    properties=properties,
                    required=t.parameters.get("required", []),
                ),
            ))
        return [genai_types.Tool(function_declarations=declarations)]

    def _from_provider_response(self, response, model_name: str) -> LLMResponse:
        """Translate Gemini GenerateContentResponse → internal LLMResponse."""
        candidate = response.candidates[0]

        finish_reason = FinishReason.STOP
        if candidate.finish_reason:
            reason_str = str(candidate.finish_reason).upper()
            if "MAX_TOKENS" in reason_str or "LENGTH" in reason_str:
                finish_reason = FinishReason.LENGTH

        text_content: Optional[str] = None
        tool_calls: list[ToolCall] = []

        for part in candidate.content.parts:
            if hasattr(part, "text") and part.text:
                text_content = part.text
            elif hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                tool_calls.append(ToolCall(
                    id=str(uuid.uuid4()),
                    name=fc.name,
                    arguments=dict(fc.args) if fc.args else {},
                ))

        if tool_calls:
            finish_reason = FinishReason.TOOL_CALLS

        usage = TokenUsage()
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            um = response.usage_metadata
            usage = TokenUsage(
                input_tokens=getattr(um, "prompt_token_count", 0) or 0,
                output_tokens=getattr(um, "candidates_token_count", 0) or 0,
            )

        return LLMResponse(
            content=text_content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            model=model_name,
            provider=Provider.GEMINI,
        )

    def _raise_normalised(self, exc: Exception) -> None:
        err_str = str(exc).lower()
        if "quota" in err_str or "rate" in err_str or "429" in err_str:
            raise LLMRateLimitError(str(exc), provider="gemini") from exc
        if "too long" in err_str or "context" in err_str:
            raise LLMContextError(str(exc), provider="gemini") from exc
        if "api key" in err_str or "invalid" in err_str or "403" in err_str or "401" in err_str:
            raise LLMConnectionError(str(exc), provider="gemini", status_code=403) from exc
        raise LLMError(str(exc), provider="gemini") from exc
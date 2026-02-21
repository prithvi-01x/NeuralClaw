"""
brain/bytez_client.py — Bytez SDK Client
"""

from __future__ import annotations

from typing import Optional

from bytez import Bytez

from brain.llm_client import (
    BaseLLMClient,
    LLMConnectionError,
    LLMError,
)
from brain.types import (
    FinishReason,
    LLMConfig,
    LLMResponse,
    Message,
    Provider,
    Role,
    TokenUsage,
    ToolSchema,
)
from observability.logger import get_logger

log = get_logger(__name__)


class BytezClient(BaseLLMClient):
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

        try:
            model = self.sdk.model(config.model)

            input_msgs = []
            for m in messages:
                if m.role in (Role.SYSTEM, Role.USER, Role.ASSISTANT):
                    input_msgs.append({
                        "role": m.role.value,
                        "content": m.content or ""
                    })

            results = model.run(input_msgs)

        except Exception as e:
            raise LLMConnectionError(str(e), provider="bytez")

        if results.error:
            raise LLMError(str(results.error), provider="bytez")

        # ─────────────────────────────────────
        # Extract text properly
        # ─────────────────────────────────────

        output_text = ""

        if isinstance(results.output, list):
            for item in results.output:
                if isinstance(item, dict):
                    if isinstance(item, dict) and item.get("role") == "assistant":
                        output_text += item.get("content", "")
        else:
            output_text = str(results.output)

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
        try:
            model = self.sdk.model("openai/gpt-5")
            res = model.run([{"role": "user", "content": "ping"}])
            return res.error is None
        except Exception as e:
            log.warning("bytez.health_check.failed", error=str(e))
            return False

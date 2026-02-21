"""
brain/types.py — NeuralClaw Brain Data Models

All shared types used across LLM clients and the agent orchestrator.
Providers (OpenAI, Anthropic, Ollama, OpenRouter, Gemini, Bytez) all map
their native response shapes into these types.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"           # tool result fed back to LLM


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"
    GEMINI = "gemini"
    BYTEZ = "bytez"          # ← NEW PROVIDER


class FinishReason(str, Enum):
    STOP = "stop"               # normal completion
    TOOL_CALLS = "tool_calls"   # LLM wants to call tools
    LENGTH = "length"           # hit max_tokens
    ERROR = "error"             # something went wrong


# ─────────────────────────────────────────────────────────────────────────────
# Tool calling types
# ─────────────────────────────────────────────────────────────────────────────


class ToolCall(BaseModel):
    """A single tool invocation requested by the LLM."""
    id: str = Field(..., description="Unique ID for this tool call (from LLM)")
    name: str = Field(..., description="Tool/function name to call")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Parsed JSON arguments")


class ToolResult(BaseModel):
    """The result of executing a tool call — fed back to the LLM."""
    tool_call_id: str = Field(..., description="Matches ToolCall.id")
    name: str = Field(..., description="Tool name (for context)")
    content: str = Field(..., description="Serialized result (JSON string or plain text)")
    is_error: bool = Field(default=False, description="True if the tool execution failed")


class ToolSchema(BaseModel):
    """
    Provider-agnostic tool definition.
    Clients translate this into provider-specific format (OpenAI function schema,
    Anthropic tool schema, etc.).
    """
    name: str
    description: str
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}, "required": []}
    )


# ─────────────────────────────────────────────────────────────────────────────
# Message types
# ─────────────────────────────────────────────────────────────────────────────


class Message(BaseModel):
    """
    A single message in the conversation.

    For tool results, set role=TOOL and populate tool_result.
    For tool calls made by the assistant, set role=ASSISTANT and populate tool_calls.
    """
    role: Role
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None     # assistant → wants to call tools
    tool_result: Optional[ToolResult] = None         # tool → result of a tool call
    name: Optional[str] = None                       # optional sender name

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: str) -> "Message":
        return cls(role=Role.ASSISTANT, content=content)

    @classmethod
    def tool_response(cls, result: ToolResult) -> "Message":
        return cls(role=Role.TOOL, tool_result=result, content=result.content)


# ─────────────────────────────────────────────────────────────────────────────
# LLM config
# ─────────────────────────────────────────────────────────────────────────────


class LLMConfig(BaseModel):
    """
    Per-request LLM configuration.
    Overrides the provider defaults for a single generate() call.
    """
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    stream: bool = False                # streaming not supported in MVP
    timeout_seconds: float = 60.0


# ─────────────────────────────────────────────────────────────────────────────
# LLM response
# ─────────────────────────────────────────────────────────────────────────────


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMResponse(BaseModel):
    """
    Normalised response from any LLM provider.
    Clients translate provider-specific responses into this shape.
    """
    content: Optional[str] = None               # text response (None if tool_calls only)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: FinishReason = FinishReason.STOP
    usage: TokenUsage = Field(default_factory=TokenUsage)
    model: str = ""                             # actual model used (may differ from requested)
    provider: Provider = Provider.OPENAI

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def is_complete(self) -> bool:
        """True when the LLM finished naturally (not mid-tool-call or truncated)."""
        return self.finish_reason == FinishReason.STOP and not self.has_tool_calls

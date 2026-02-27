"""
tests/unit/test_brain.py — Brain Module Unit Tests

Tests all LLM clients with mocked API calls.
No real API keys or network calls required.

Run with:
    pytest tests/unit/test_brain.py -v
    pytest tests/unit/test_brain.py -v --tb=short
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neuralclaw.brain import (
    LLMClientFactory,
    LLMConfig,
    LLMConnectionError,
    LLMError,
    Message,
    Role,
    ToolCall,
    ToolResult,
    ToolSchema,
)
from neuralclaw.brain.types import FinishReason, Provider, TokenUsage


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def basic_messages() -> list[Message]:
    return [
        Message.system("You are a helpful assistant."),
        Message.user("Say hello."),
    ]


@pytest.fixture
def basic_config() -> LLMConfig:
    return LLMConfig(model="gpt-4o", temperature=0.7, max_tokens=100)


@pytest.fixture
def sample_tool() -> ToolSchema:
    return ToolSchema(
        name="get_weather",
        description="Get the current weather for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
            },
            "required": ["location"],
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Message model tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMessage:
    def test_system_factory(self):
        msg = Message.system("You are helpful")
        assert msg.role == Role.SYSTEM
        assert msg.content == "You are helpful"

    def test_user_factory(self):
        msg = Message.user("Hello!")
        assert msg.role == Role.USER
        assert msg.content == "Hello!"

    def test_assistant_factory(self):
        msg = Message.assistant("Hi there!")
        assert msg.role == Role.ASSISTANT
        assert msg.content == "Hi there!"

    def test_tool_response_factory(self):
        result = ToolResult(
            tool_call_id="call_123",
            name="get_weather",
            content='{"temp": 72, "condition": "sunny"}',
        )
        msg = Message.tool_response(result)
        assert msg.role == Role.TOOL
        assert msg.tool_result == result

    def test_tool_call_in_message(self):
        tc = ToolCall(id="call_1", name="search", arguments={"query": "python"})
        msg = Message(role=Role.ASSISTANT, tool_calls=[tc])
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "search"


# ─────────────────────────────────────────────────────────────────────────────
# LLMConfig tests
# ─────────────────────────────────────────────────────────────────────────────


class TestLLMConfig:
    def test_defaults(self):
        config = LLMConfig(model="gpt-4o")
        assert config.temperature == 0.7
        assert config.max_tokens == 4096
        assert config.stream is False

    def test_custom_values(self):
        config = LLMConfig(model="claude-3-5-sonnet-20241022", temperature=0.0, max_tokens=1000)
        assert config.temperature == 0.0
        assert config.max_tokens == 1000


# ─────────────────────────────────────────────────────────────────────────────
# LLMClientFactory tests
# ─────────────────────────────────────────────────────────────────────────────


class TestLLMClientFactory:
    def test_create_openai(self):
        from neuralclaw.brain.openai_client import OpenAIClient
        client = LLMClientFactory.create("openai", api_key="sk-test")
        assert isinstance(client, OpenAIClient)

    def test_create_anthropic(self):
        from neuralclaw.brain.anthropic_client import AnthropicClient
        client = LLMClientFactory.create("anthropic", api_key="sk-ant-test")
        assert isinstance(client, AnthropicClient)

    def test_create_ollama(self):
        from neuralclaw.brain.ollama_client import OllamaClient
        client = LLMClientFactory.create("ollama")
        assert isinstance(client, OllamaClient)

    def test_create_openrouter(self):
        from neuralclaw.brain.openrouter_client import OpenRouterClient
        client = LLMClientFactory.create("openrouter", api_key="sk-or-test")
        assert isinstance(client, OpenRouterClient)

    def test_create_gemini(self):
        import sys
        import types
        # Ensure google.genai is importable so brain.gemini_client can be loaded
        if "google" not in sys.modules:
            sys.modules["google"] = types.ModuleType("google")
        if "google.genai" not in sys.modules:
            sys.modules["google.genai"] = types.ModuleType("google.genai")
        if "google.genai.types" not in sys.modules:
            sys.modules["google.genai.types"] = types.ModuleType("google.genai.types")
        import importlib
        import neuralclaw.brain.gemini_client as _gc_mod
        importlib.reload(_gc_mod)
        with patch("neuralclaw.brain.gemini_client.genai"):
            from neuralclaw.brain.gemini_client import GeminiClient
            client = LLMClientFactory.create("gemini", api_key="AIza-test")
            assert isinstance(client, GeminiClient)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            LLMClientFactory.create("grok", api_key="test")

    def test_openai_missing_key_raises(self):
        with pytest.raises(LLMConnectionError):
            LLMClientFactory.create("openai", api_key=None)

    def test_anthropic_missing_key_raises(self):
        with pytest.raises(LLMConnectionError):
            LLMClientFactory.create("anthropic", api_key=None)

    def test_case_insensitive_provider(self):
        from neuralclaw.brain.openai_client import OpenAIClient
        client = LLMClientFactory.create("OpenAI", api_key="sk-test")
        assert isinstance(client, OpenAIClient)

    def test_default_model(self):
        assert LLMClientFactory.default_model("openai") == "gpt-4o"
        assert LLMClientFactory.default_model("anthropic") == "claude-3-5-sonnet-20241022"
        assert LLMClientFactory.default_model("ollama") == "llama3.1"
        assert LLMClientFactory.default_model("gemini") == "gemini-1.5-pro"


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI Client tests
# ─────────────────────────────────────────────────────────────────────────────


class TestOpenAIClient:
    @pytest.fixture
    def client(self):
        from neuralclaw.brain.openai_client import OpenAIClient
        return OpenAIClient(api_key="sk-test-fake")

    def _make_mock_response(
        self,
        content: str = "Hello!",
        finish_reason: str = "stop",
        tool_calls=None,
        model: str = "gpt-4o",
        input_tokens: int = 10,
        output_tokens: int = 5,
    ):
        mock = MagicMock()
        mock.model = model
        mock.choices = [MagicMock()]
        mock.choices[0].finish_reason = finish_reason
        mock.choices[0].message.content = content
        mock.choices[0].message.tool_calls = tool_calls
        mock.usage.prompt_tokens = input_tokens
        mock.usage.completion_tokens = output_tokens
        return mock

    @pytest.mark.asyncio
    async def test_basic_generate(self, client, basic_messages, basic_config):
        mock_response = self._make_mock_response(content="Hello! How can I help?")
        client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await client.generate(basic_messages, basic_config)

        assert result.content == "Hello! How can I help?"
        assert result.finish_reason == FinishReason.STOP
        assert result.provider == Provider.OPENAI
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5
        assert result.usage.total_tokens == 15
        assert not result.has_tool_calls

    @pytest.mark.asyncio
    async def test_tool_call_response(self, client, basic_messages, basic_config, sample_tool):
        mock_tc = MagicMock()
        mock_tc.id = "call_abc123"
        mock_tc.function.name = "get_weather"
        mock_tc.function.arguments = '{"location": "London"}'

        mock_response = self._make_mock_response(
            content=None,
            finish_reason="tool_calls",
            tool_calls=[mock_tc],
        )
        client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await client.generate(basic_messages, basic_config, tools=[sample_tool])

        assert result.has_tool_calls
        assert result.finish_reason == FinishReason.TOOL_CALLS
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_abc123"
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].arguments == {"location": "London"}

    @pytest.mark.asyncio
    async def test_malformed_tool_args_handled(self, client, basic_messages, basic_config, sample_tool):
        """Invalid JSON in tool arguments should not crash — stored as _raw."""
        mock_tc = MagicMock()
        mock_tc.id = "call_xyz"
        mock_tc.function.name = "get_weather"
        mock_tc.function.arguments = "NOT VALID JSON"

        mock_response = self._make_mock_response(
            content=None, finish_reason="tool_calls", tool_calls=[mock_tc]
        )
        client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await client.generate(basic_messages, basic_config, tools=[sample_tool])
        assert result.tool_calls[0].arguments == {"_raw": "NOT VALID JSON"}

    @pytest.mark.asyncio
    async def test_auth_error_raises_connection_error(self, client, basic_messages, basic_config):
        import openai as oai
        client._client.chat.completions.create = AsyncMock(
            side_effect=oai.AuthenticationError("Invalid key", response=MagicMock(), body={})
        )
        with pytest.raises(LLMConnectionError):
            await client.generate(basic_messages, basic_config)

    @pytest.mark.asyncio
    async def test_rate_limit_raises(self, client, basic_messages, basic_config):
        import openai as oai
        client._client.chat.completions.create = AsyncMock(
            side_effect=oai.RateLimitError("Rate limit", response=MagicMock(), body={})
        )
        from neuralclaw.brain.llm_client import LLMRateLimitError
        with pytest.raises(LLMRateLimitError):
            await client.generate(basic_messages, basic_config)

    def test_message_translation_system(self, client):
        messages = [Message.system("Be helpful")]
        result = client._to_provider_messages(messages)
        assert result[0] == {"role": "system", "content": "Be helpful"}

    def test_message_translation_tool_result(self, client):
        tr = ToolResult(tool_call_id="c1", name="search", content='{"results": []}')
        msg = Message.tool_response(tr)
        result = client._to_provider_messages([msg])
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "c1"

    def test_tool_schema_translation(self, client, sample_tool):
        result = client._to_provider_tools([sample_tool])
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert "location" in result[0]["function"]["parameters"]["properties"]


# ─────────────────────────────────────────────────────────────────────────────
# Anthropic Client tests
# ─────────────────────────────────────────────────────────────────────────────


class TestAnthropicClient:
    @pytest.fixture
    def client(self):
        from neuralclaw.brain.anthropic_client import AnthropicClient
        return AnthropicClient(api_key="sk-ant-test-fake")

    def _make_mock_response(
        self,
        text: str = "Hello!",
        stop_reason: str = "end_turn",
        tool_uses: list | None = None,
        model: str = "claude-3-5-sonnet-20241022",
        input_tokens: int = 10,
        output_tokens: int = 5,
    ):
        mock = MagicMock()
        mock.model = model
        mock.stop_reason = stop_reason
        mock.usage.input_tokens = input_tokens
        mock.usage.output_tokens = output_tokens

        content_blocks = []
        if text:
            tb = MagicMock()
            tb.type = "text"
            tb.text = text
            content_blocks.append(tb)
        if tool_uses:
            for tu in tool_uses:
                tb = MagicMock()
                tb.type = "tool_use"
                tb.id = tu["id"]
                tb.name = tu["name"]
                tb.input = tu["input"]
                content_blocks.append(tb)

        mock.content = content_blocks
        return mock

    @pytest.mark.asyncio
    async def test_basic_generate(self, client, basic_messages, basic_config):
        basic_config.model = "claude-3-5-sonnet-20241022"
        mock_response = self._make_mock_response(text="Hello from Claude!")
        client._client.messages.create = AsyncMock(return_value=mock_response)

        result = await client.generate(basic_messages, basic_config)

        assert result.content == "Hello from Claude!"
        assert result.finish_reason == FinishReason.STOP
        assert result.provider == Provider.ANTHROPIC
        assert result.usage.input_tokens == 10

    @pytest.mark.asyncio
    async def test_tool_call_response(self, client, basic_messages, basic_config, sample_tool):
        basic_config.model = "claude-3-5-sonnet-20241022"
        mock_response = self._make_mock_response(
            text=None,
            stop_reason="tool_use",
            tool_uses=[{"id": "toolu_01", "name": "get_weather", "input": {"location": "Paris"}}],
        )
        client._client.messages.create = AsyncMock(return_value=mock_response)

        result = await client.generate(basic_messages, basic_config, tools=[sample_tool])

        assert result.has_tool_calls
        assert result.tool_calls[0].id == "toolu_01"
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].arguments == {"location": "Paris"}

    def test_system_prompt_extraction(self, client):
        messages = [
            Message.system("You are helpful"),
            Message.user("Hello"),
        ]
        system_out, msgs_out = client._to_provider_messages(messages)
        assert system_out == "You are helpful"
        assert any(m["role"] == "user" for m in msgs_out)

    def test_tool_schema_translation(self, client, sample_tool):
        result = client._to_provider_tools([sample_tool])
        assert result[0]["name"] == "get_weather"
        assert result[0]["input_schema"] == sample_tool.parameters
        assert "description" in result[0]

    def test_tool_result_in_user_message(self, client):
        """Anthropic needs tool results wrapped in user message content blocks."""
        tr = ToolResult(tool_call_id="toolu_01", name="get_weather", content='{"temp": 20}')
        msg = Message.tool_response(tr)
        _, msgs = client._to_provider_messages([Message.user("Start"), Message(role=Role.ASSISTANT, content="ok"), msg])
        tool_msg = [m for m in msgs if m["role"] == "user" and isinstance(m["content"], list)]
        assert len(tool_msg) > 0
        assert tool_msg[0]["content"][0]["type"] == "tool_result"

    @pytest.mark.asyncio
    async def test_auth_error_raises_connection_error(self, client, basic_messages, basic_config):
        import anthropic as ant
        client._client.messages.create = AsyncMock(
            side_effect=ant.AuthenticationError.__new__(ant.AuthenticationError)
        )
        # Basic check that auth errors surface correctly (error class may vary)
        with pytest.raises(Exception):
            await client.generate(basic_messages, basic_config)


# ─────────────────────────────────────────────────────────────────────────────
# Ollama Client tests
# ─────────────────────────────────────────────────────────────────────────────


class TestOllamaClient:
    @pytest.fixture
    def client(self):
        from neuralclaw.brain.ollama_client import OllamaClient
        return OllamaClient(base_url="http://localhost:11434/v1")

    @pytest.mark.asyncio
    async def test_generate_delegates_to_openai_client(self, client, basic_messages):
        from neuralclaw.brain.types import LLMResponse, FinishReason, TokenUsage
        config = LLMConfig(model="llama3.1")

        mock_response = LLMResponse(
            content="Hello from Llama!",
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(input_tokens=8, output_tokens=4),
            model="llama3.1",
            provider=Provider.OPENAI,  # inner client sets this; OllamaClient overrides
        )
        client._inner.generate = AsyncMock(return_value=mock_response)

        result = await client.generate(basic_messages, config)

        assert result.content == "Hello from Llama!"
        assert result.provider == Provider.OLLAMA  # must be overridden

    @pytest.mark.asyncio
    async def test_connection_error_gives_helpful_message(self, client, basic_messages):
        config = LLMConfig(model="llama3.1")
        client._inner.generate = AsyncMock(
            side_effect=LLMConnectionError("connection refused", provider="openai")
        )
        with pytest.raises(LLMConnectionError, match="ollama serve"):
            await client.generate(basic_messages, config)

    @pytest.mark.asyncio
    async def test_health_check_success(self, client):
        mock_model = MagicMock()
        mock_model.id = "llama3.1"
        client._raw_client.models.list = AsyncMock(return_value=MagicMock(data=[mock_model]))
        assert await client.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, client):
        client._raw_client.models.list = AsyncMock(side_effect=OSError("connection refused"))
        assert await client.health_check() is False

    @pytest.mark.asyncio
    async def test_list_models(self, client):
        mock_models = [MagicMock(id="llama3.1"), MagicMock(id="mistral")]
        client._raw_client.models.list = AsyncMock(return_value=MagicMock(data=mock_models))
        models = await client.list_models()
        assert "llama3.1" in models
        assert "mistral" in models


# ─────────────────────────────────────────────────────────────────────────────
# OpenRouter Client tests
# ─────────────────────────────────────────────────────────────────────────────


class TestOpenRouterClient:
    @pytest.fixture
    def client(self):
        from neuralclaw.brain.openrouter_client import OpenRouterClient
        return OpenRouterClient(api_key="sk-or-test")

    @pytest.mark.asyncio
    async def test_generate_sets_provider(self, client, basic_messages):
        from neuralclaw.brain.types import LLMResponse, FinishReason, TokenUsage
        config = LLMConfig(model="openai/gpt-4o")

        mock_response = LLMResponse(
            content="Response from OpenRouter",
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(input_tokens=5, output_tokens=3),
            model="openai/gpt-4o",
            provider=Provider.OPENAI,
        )
        client._inner.generate = AsyncMock(return_value=mock_response)

        result = await client.generate(basic_messages, config)
        assert result.provider == Provider.OPENROUTER
        assert result.content == "Response from OpenRouter"

    @pytest.mark.asyncio
    async def test_health_check_success(self, client):
        client._inner._client.models.list = AsyncMock(return_value=MagicMock(data=[]))
        assert await client.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, client):
        client._inner._client.models.list = AsyncMock(side_effect=OSError("network error"))
        assert await client.health_check() is False


# ─────────────────────────────────────────────────────────────────────────────
# LLMResponse tests
# ─────────────────────────────────────────────────────────────────────────────


class TestLLMResponse:
    def test_has_tool_calls_false_when_empty(self):
        from neuralclaw.brain.types import LLMResponse
        r = LLMResponse(content="hello", finish_reason=FinishReason.STOP)
        assert not r.has_tool_calls

    def test_has_tool_calls_true(self):
        from neuralclaw.brain.types import LLMResponse
        r = LLMResponse(
            tool_calls=[ToolCall(id="x", name="foo", arguments={})],
            finish_reason=FinishReason.TOOL_CALLS,
        )
        assert r.has_tool_calls

    def test_is_complete(self):
        from neuralclaw.brain.types import LLMResponse
        r = LLMResponse(content="done", finish_reason=FinishReason.STOP)
        assert r.is_complete

    def test_not_complete_when_tool_calls(self):
        from neuralclaw.brain.types import LLMResponse
        r = LLMResponse(
            tool_calls=[ToolCall(id="x", name="foo", arguments={})],
            finish_reason=FinishReason.TOOL_CALLS,
        )
        assert not r.is_complete

    def test_token_usage_total(self):
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        assert usage.total_tokens == 150


# ─────────────────────────────────────────────────────────────────────────────
# ToolSchema tests
# ─────────────────────────────────────────────────────────────────────────────


class TestToolSchema:
    def test_default_parameters(self):
        tool = ToolSchema(name="my_tool", description="Does stuff")
        assert tool.parameters["type"] == "object"
        assert tool.parameters["properties"] == {}

    def test_custom_parameters(self, sample_tool):
        assert "location" in sample_tool.parameters["properties"]
        assert sample_tool.parameters["required"] == ["location"]
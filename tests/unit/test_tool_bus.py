"""
tests/unit/test_tool_bus.py — Tool Bus and Registry Unit Tests

Tests the ToolBus dispatch pipeline and ToolRegistry
with mock handlers and a real SafetyKernel.

Run with:
    pytest tests/unit/test_tool_bus.py -v
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from safety.safety_kernel import SafetyKernel
from tools.tool_bus import ToolBus, _normalise_result, _truncate, _validate_args
from tools.tool_registry import ToolRegistry
from tools.types import (
    RiskLevel,
    SafetyDecision,
    SafetyStatus,
    ToolCall,
    ToolResult,
    ToolSchema,
    TrustLevel,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def registry():
    return ToolRegistry()


@pytest.fixture
def kernel():
    return SafetyKernel(allowed_paths=["~/agent_files", "/tmp"])


@pytest.fixture
def bus(registry, kernel):
    return ToolBus(registry, kernel, timeout_seconds=5.0)


def make_call(name: str, args: dict = None, call_id: str = "call_1") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=args or {})


def make_schema(
    name: str,
    category: str = "general",
    risk: RiskLevel = RiskLevel.LOW,
    enabled: bool = True,
) -> ToolSchema:
    return ToolSchema(
        name=name,
        description=f"Test tool: {name}",
        category=category,
        risk_level=risk,
        enabled=enabled,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ToolRegistry tests
# ─────────────────────────────────────────────────────────────────────────────


class TestToolRegistry:
    def test_register_and_lookup(self, registry):
        schema = make_schema("my_tool")

        async def handler(x: str) -> str:
            return x

        registry.register_tool(schema, handler)
        assert registry.is_registered("my_tool")
        assert registry.get_schema("my_tool") == schema
        assert registry.get_handler("my_tool") is handler

    def test_unknown_tool_returns_none(self, registry):
        assert registry.get_schema("nonexistent") is None
        assert registry.get_handler("nonexistent") is None

    def test_list_schemas(self, registry):
        registry.register_tool(make_schema("tool_a"), AsyncMock())
        registry.register_tool(make_schema("tool_b"), AsyncMock())
        names = registry.list_names()
        assert "tool_a" in names
        assert "tool_b" in names

    def test_disabled_tool_excluded_from_list(self, registry):
        registry.register_tool(make_schema("enabled_tool"), AsyncMock())
        registry.register_tool(make_schema("disabled_tool", enabled=False), AsyncMock())
        names = registry.list_names(enabled_only=True)
        assert "enabled_tool" in names
        assert "disabled_tool" not in names

    def test_enable_disable(self, registry):
        registry.register_tool(make_schema("toggle_tool"), AsyncMock())
        registry.disable("toggle_tool")
        assert "toggle_tool" not in registry.list_names(enabled_only=True)
        registry.enable("toggle_tool")
        assert "toggle_tool" in registry.list_names(enabled_only=True)

    def test_to_llm_schemas(self, registry):
        registry.register_tool(make_schema("llm_tool"), AsyncMock())
        llm_schemas = registry.to_llm_schemas()
        assert any(s["name"] == "llm_tool" for s in llm_schemas)

    def test_decorator_registration(self):
        reg = ToolRegistry()

        @reg.register(
            name="decorated_tool",
            description="A decorated tool",
            category="test",
            risk_level=RiskLevel.LOW,
        )
        async def my_tool(x: str) -> str:
            return x

        assert reg.is_registered("decorated_tool")

    def test_len(self, registry):
        assert len(registry) == 0
        registry.register_tool(make_schema("a"), AsyncMock())
        assert len(registry) == 1


# ─────────────────────────────────────────────────────────────────────────────
# ToolBus dispatch tests
# ─────────────────────────────────────────────────────────────────────────────


class TestToolBusDispatch:
    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self, bus):
        call = make_call("nonexistent_tool", {"x": 1})
        result = await bus.dispatch(call)
        assert result.is_error
        assert "Unknown tool" in result.content

    @pytest.mark.asyncio
    async def test_successful_dispatch(self, registry, kernel):
        handler = AsyncMock(return_value="search results here")
        schema = make_schema("web_search", category="search", risk=RiskLevel.LOW)
        registry.register_tool(schema, handler)
        bus = ToolBus(registry, kernel)

        call = make_call("web_search", {"query": "test"})
        result = await bus.dispatch(call, TrustLevel.LOW)

        assert not result.is_error
        assert result.content == "search results here"
        handler.assert_called_once_with(query="test")

    @pytest.mark.asyncio
    async def test_missing_required_param_returns_error(self, registry, kernel):
        handler = AsyncMock(return_value="ok")
        schema = ToolSchema(
            name="strict_tool",
            description="needs x",
            category="general",
            risk_level=RiskLevel.LOW,
            parameters={
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            },
        )
        registry.register_tool(schema, handler)
        bus = ToolBus(registry, kernel)

        call = make_call("strict_tool", {})  # missing required "x"
        result = await bus.dispatch(call)

        assert result.is_error
        assert "Missing required field" in result.content

    @pytest.mark.asyncio
    async def test_handler_exception_returns_error(self, registry, kernel):
        handler = AsyncMock(side_effect=RuntimeError("handler blew up"))
        schema = make_schema("exploding_tool", category="search", risk=RiskLevel.LOW)
        registry.register_tool(schema, handler)
        bus = ToolBus(registry, kernel)

        call = make_call("exploding_tool", {})
        result = await bus.dispatch(call)

        assert result.is_error
        assert "handler blew up" in result.content

    @pytest.mark.asyncio
    async def test_timeout_returns_error(self, registry, kernel):
        import asyncio

        async def slow_handler():
            await asyncio.sleep(100)

        schema = make_schema("slow_tool", category="search", risk=RiskLevel.LOW)
        registry.register_tool(schema, slow_handler)
        bus = ToolBus(registry, kernel, timeout_seconds=0.01)

        call = make_call("slow_tool", {})
        result = await bus.dispatch(call)

        assert result.is_error
        assert "timed out" in result.content.lower()

    @pytest.mark.asyncio
    async def test_blocked_tool_returns_error(self, registry, kernel):
        handler = AsyncMock(return_value="shouldn't run")
        schema = make_schema("blocked_tool", category="terminal", risk=RiskLevel.HIGH)
        registry.register_tool(schema, handler)
        bus = ToolBus(registry, kernel)

        # Command that triggers blocked pattern
        call = make_call("blocked_tool", {"command": "rm -rf /"})
        result = await bus.dispatch(call, TrustLevel.LOW)

        assert result.is_error
        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_confirm_needed_with_no_handler_denies(self, registry, kernel):
        handler = AsyncMock(return_value="executed")
        schema = make_schema("risky_tool", category="terminal", risk=RiskLevel.HIGH)
        registry.register_tool(schema, handler)

        # No on_confirm_needed → defaults to deny
        bus = ToolBus(registry, kernel, on_confirm_needed=None)
        call = make_call("risky_tool", {"command": "ls -la"})
        result = await bus.dispatch(call, TrustLevel.LOW)

        assert result.is_error
        assert "denied" in result.content.lower() or "blocked" in result.content.lower()

    @pytest.mark.asyncio
    async def test_confirm_needed_approved_by_handler(self, registry, kernel):
        handler = AsyncMock(return_value="executed after confirm")
        schema = make_schema("risky_tool", category="terminal", risk=RiskLevel.HIGH)
        registry.register_tool(schema, handler)

        async def approve_all(decision) -> bool:
            return True

        bus = ToolBus(registry, kernel, on_confirm_needed=approve_all)
        call = make_call("risky_tool", {"command": "ls -la"})
        result = await bus.dispatch(call, TrustLevel.LOW)

        assert not result.is_error
        assert result.content == "executed after confirm"

    @pytest.mark.asyncio
    async def test_result_has_duration(self, registry, kernel):
        handler = AsyncMock(return_value="done")
        schema = make_schema("fast_tool", category="search", risk=RiskLevel.LOW)
        registry.register_tool(schema, handler)
        bus = ToolBus(registry, kernel)

        call = make_call("fast_tool", {})
        result = await bus.dispatch(call)

        assert result.duration_ms >= 0


# ─────────────────────────────────────────────────────────────────────────────
# Helper function tests
# ─────────────────────────────────────────────────────────────────────────────


class TestHelpers:
    def test_normalise_none(self):
        assert _normalise_result(None) == "Done."

    def test_normalise_string(self):
        assert _normalise_result("hello") == "hello"

    def test_normalise_dict(self):
        result = _normalise_result({"key": "value"})
        parsed = json.loads(result)
        assert parsed["key"] == "value"

    def test_normalise_list(self):
        result = _normalise_result([1, 2, 3])
        parsed = json.loads(result)
        assert parsed == [1, 2, 3]

    def test_truncate_short_string(self):
        assert _truncate("hello", 100) == "hello"

    def test_truncate_long_string(self):
        long = "x" * 200
        result = _truncate(long, 100)
        assert len(result) > 100  # includes the notice
        assert "truncated" in result.lower()
        assert result.startswith("x" * 100)

    def test_validate_args_missing_required(self):
        error = _validate_args({}, {"required": ["x"], "properties": {"x": {"type": "str"}}})
        assert "x" in error

    def test_validate_args_all_present(self):
        error = _validate_args({"x": "hello"}, {"required": ["x"], "properties": {}})
        assert error is None

    def test_validate_args_no_required(self):
        error = _validate_args({}, {"required": [], "properties": {}})
        assert error is None


# ─────────────────────────────────────────────────────────────────────────────
# ToolResult tests
# ─────────────────────────────────────────────────────────────────────────────


class TestToolResult:
    def test_success_factory(self):
        result = ToolResult.success("call_1", "my_tool", "output", RiskLevel.LOW, 42.0)
        assert not result.is_error
        assert result.content == "output"
        assert result.duration_ms == 42.0

    def test_error_factory(self):
        result = ToolResult.error("call_1", "my_tool", "something went wrong")
        assert result.is_error
        assert "something went wrong" in result.content
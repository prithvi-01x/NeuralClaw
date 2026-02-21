"""
tools/tool_registry.py — Tool Registry

Central registry for all tools available to the agent.
Tools register themselves via the @registry.register() decorator.

The registry stores:
  - ToolSchema (metadata, risk level, JSON schema for LLM)
  - The async handler function to call for execution

Usage:
    registry = ToolRegistry()

    @registry.register(
        name="file_read",
        description="Read a file",
        category="filesystem",
        risk_level=RiskLevel.LOW,
        parameters={...}
    )
    async def file_read(path: str) -> str:
        ...

    # Lookup
    schema = registry.get_schema("file_read")
    handler = registry.get_handler("file_read")
    all_schemas = registry.list_schemas()
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Optional

from observability.logger import get_logger
from tools.types import RiskLevel, ToolSchema

log = get_logger(__name__)


class ToolRegistry:
    """
    Registry that maps tool names to their schemas and async handlers.

    Thread-safe for reads (dict lookups). Not designed for concurrent writes.
    """

    def __init__(self):
        self._schemas: dict[str, ToolSchema] = {}
        self._handlers: dict[str, Callable] = {}

    def register(
        self,
        name: str,
        description: str,
        category: str = "general",
        risk_level: RiskLevel = RiskLevel.LOW,
        requires_confirmation: bool = False,
        parameters: Optional[dict[str, Any]] = None,
        enabled: bool = True,
    ) -> Callable:
        """
        Decorator to register an async tool handler.

        Example:
            @registry.register(
                name="file_read",
                description="Read a text file",
                category="filesystem",
                risk_level=RiskLevel.LOW,
                parameters={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]
                }
            )
            async def file_read(path: str) -> str:
                ...
        """
        def decorator(fn: Callable) -> Callable:
            schema = ToolSchema(
                name=name,
                description=description,
                category=category,
                risk_level=risk_level,
                requires_confirmation=requires_confirmation,
                parameters=parameters or {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
                enabled=enabled,
            )
            self._schemas[name] = schema
            self._handlers[name] = fn
            log.debug("tool.registered", tool=name, category=category, risk=risk_level.value)

            @functools.wraps(fn)
            async def wrapper(*args, **kwargs):
                return await fn(*args, **kwargs)

            return wrapper

        return decorator

    def register_tool(self, schema: ToolSchema, handler: Callable) -> None:
        """Programmatic registration (alternative to decorator)."""
        self._schemas[schema.name] = schema
        self._handlers[schema.name] = handler
        log.debug(
            "tool.registered",
            tool=schema.name,
            category=schema.category,
            risk=schema.risk_level.value,
        )

    def get_schema(self, name: str) -> Optional[ToolSchema]:
        """Return the ToolSchema for a tool, or None if not found."""
        return self._schemas.get(name)

    def get_handler(self, name: str) -> Optional[Callable]:
        """Return the async handler function for a tool, or None if not found."""
        return self._handlers.get(name)

    def is_registered(self, name: str) -> bool:
        return name in self._schemas

    def list_schemas(self, enabled_only: bool = True) -> list[ToolSchema]:
        """Return all registered tool schemas."""
        schemas = list(self._schemas.values())
        if enabled_only:
            schemas = [s for s in schemas if s.enabled]
        return schemas

    def list_names(self, enabled_only: bool = True) -> list[str]:
        """Return all registered tool names."""
        return [s.name for s in self.list_schemas(enabled_only)]

    def to_llm_schemas(self) -> list[dict[str, Any]]:
        """Return all enabled tools in the format the LLM brain expects."""
        return [s.to_llm_schema() for s in self.list_schemas(enabled_only=True)]

    def enable(self, name: str) -> None:
        if name in self._schemas:
            self._schemas[name].enabled = True

    def disable(self, name: str) -> None:
        if name in self._schemas:
            self._schemas[name].enabled = False

    def __len__(self) -> int:
        return len(self._schemas)

    def __repr__(self) -> str:
        return f"<ToolRegistry tools={list(self._schemas.keys())}>"


# ─────────────────────────────────────────────────────────────────────────────
# Global registry singleton
# ─────────────────────────────────────────────────────────────────────────────

# Import this in tool modules to self-register
registry = ToolRegistry()
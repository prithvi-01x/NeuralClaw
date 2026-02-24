"""
mcp/manager.py — MCP Manager

Orchestrates all MCP server connections.
On startup, connects to enabled servers, fetches their tool lists,
and registers those tools on the global ToolRegistry so the agent
can call them like any native tool.

Usage:
    manager = MCPManager.from_settings(settings)
    await manager.start()           # connects all enabled servers
    # tools are now in global_registry
    await manager.call_tool("blender__render_scene", {...})
    await manager.stop()            # graceful shutdown

Architecture:
    MCPManager
      └── per server: BaseTransport (StdioTransport | HttpTransport)
            └── JSON-RPC ↔ MCP server process
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from mcp.transports.base import BaseTransport, MCPConnectionError
from mcp.transports.stdio import StdioTransport
from mcp.transports.http import HttpTransport
from mcp.types import MCPServerConfig, MCPToolResult
from observability.logger import get_logger
from skills.types import RiskLevel

log = get_logger(__name__)


class MCPManager:
    """
    Manages connections to one or more MCP servers.

    Registered MCP tools appear as regular tools in the ToolRegistry,
    so the Orchestrator doesn't need to know they come from MCP.
    """

    def __init__(self, server_configs: list[MCPServerConfig]):
        self._configs = {c.name: c for c in server_configs}
        self._transports: dict[str, BaseTransport] = {}
        self._tool_to_server: dict[str, str] = {}  # tool_name → server_name

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self, skill_registry=None) -> None:
        """Connect to all enabled MCP servers and register their tools.

        Args:
            skill_registry: Optional SkillRegistry to register MCP tools into.
                            When None, tools are registered as metadata only
                            (useful during migration / testing).
        """
        enabled = [c for c in self._configs.values() if c.enabled]
        if not enabled:
            log.info("mcp.manager.no_servers_enabled")
            return

        log.info("mcp.manager.starting", servers=[c.name for c in enabled])

        results = await asyncio.gather(
            *[self._connect_server(config, skill_registry=skill_registry) for config in enabled],
            return_exceptions=True,
        )

        connected = sum(1 for r in results if not isinstance(r, Exception))
        log.info("mcp.manager.started",
                 connected=connected,
                 total=len(enabled),
                 tools_registered=len(self._tool_to_server))

    async def stop(self) -> None:
        """Disconnect from all MCP servers."""
        if not self._transports:
            return

        log.info("mcp.manager.stopping", servers=list(self._transports.keys()))
        await asyncio.gather(
            *[t.disconnect() for t in self._transports.values()],
            return_exceptions=True,
        )
        self._transports.clear()
        self._tool_to_server.clear()
        log.info("mcp.manager.stopped")

    # ── Tool calling ──────────────────────────────────────────────────────────

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        timeout: Optional[float] = None,
    ) -> MCPToolResult:
        """
        Call a tool on the appropriate MCP server.

        Args:
            tool_name:  Full namespaced tool name (e.g. "blender__render_scene").
            arguments:  Tool arguments dict.
            timeout:    Per-call timeout override.

        Returns:
            MCPToolResult with content or error.
        """
        server_name = self._tool_to_server.get(tool_name)
        if not server_name:
            return MCPToolResult(
                tool_call_id="",
                name=tool_name,
                content=f"Error: Tool '{tool_name}' not found in any connected MCP server.",
                is_error=True,
            )

        transport = self._transports.get(server_name)
        if not transport or not transport.is_connected:
            return MCPToolResult(
                tool_call_id="",
                name=tool_name,
                content=f"Error: MCP server '{server_name}' is not connected.",
                is_error=True,
            )

        config = self._configs[server_name]
        call_timeout = timeout or config.timeout_seconds

        try:
            content = await transport.call_tool(tool_name, arguments, timeout=call_timeout)
            return MCPToolResult(
                tool_call_id="",
                name=tool_name,
                content=content,
                server_name=server_name,
            )
        except asyncio.TimeoutError:
            return MCPToolResult(
                tool_call_id="",
                name=tool_name,
                content=f"Error: Tool '{tool_name}' timed out after {call_timeout}s.",
                is_error=True,
                server_name=server_name,
            )
        except (MCPConnectionError, OSError, RuntimeError, ValueError) as e:
            log.error("mcp.call_tool.error", tool=tool_name, server=server_name, error=str(e), error_type=type(e).__name__)
            return MCPToolResult(
                tool_call_id="",
                name=tool_name,
                content=f"Error: {e}",
                is_error=True,
                server_name=server_name,
            )

    # ── Queries ───────────────────────────────────────────────────────────────

    def list_tools(self) -> list[str]:
        """Return names of all registered MCP tools."""
        return list(self._tool_to_server.keys())

    def list_servers(self) -> dict[str, bool]:
        """Return server name → is_connected mapping."""
        return {
            name: (name in self._transports and self._transports[name].is_connected)
            for name in self._configs
        }

    def is_connected(self, server_name: str) -> bool:
        t = self._transports.get(server_name)
        return t is not None and t.is_connected

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls, settings) -> "MCPManager":
        """Create MCPManager from NeuralClaw Settings."""
        mcp_cfg = settings.mcp or {}
        servers_raw = mcp_cfg.get("servers", {})

        configs = [
            MCPServerConfig.from_dict(name, data)
            for name, data in servers_raw.items()
        ]
        return cls(configs)

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _connect_server(self, config: MCPServerConfig, skill_registry=None) -> None:
        """Connect to one MCP server, fetch tools, and register them."""
        transport = self._make_transport(config)

        try:
            await transport.connect()
        except MCPConnectionError as e:
            log.warning("mcp.server.connect_failed", server=config.name, error=str(e))
            return

        self._transports[config.name] = transport

        # Fetch and register tools
        try:
            schemas = await transport.list_tools()
        except (MCPConnectionError, OSError, RuntimeError, ValueError) as e:
            log.warning("mcp.server.list_tools_failed", server=config.name, error=str(e), error_type=type(e).__name__)
            return

        for schema in schemas:
            self._tool_to_server[schema.name] = config.name
            if skill_registry is not None:
                self._register_tool(schema, config, skill_registry)
            else:
                log.debug("mcp.tool.metadata_only", tool=schema.name, server=config.name)

        log.info("mcp.server.ready",
                 server=config.name,
                 tool_count=len(schemas),
                 tools=[s.name for s in schemas])

    def _make_transport(self, config: MCPServerConfig) -> BaseTransport:
        if config.transport == "http":
            return HttpTransport(config)
        return StdioTransport(config)  # default: stdio

    def _register_tool(self, schema, config: MCPServerConfig, skill_registry) -> None:
        """Register an MCP tool on the provided SkillRegistry."""
        # Capture in closure
        tool_name = schema.name
        manager = self

        async def _mcp_handler(**kwargs) -> str:
            result = await manager.call_tool(tool_name, kwargs)
            if result.is_error:
                raise RuntimeError(result.content)
            return result.content

        # Determine risk based on server config or tool name heuristics
        risk = _infer_risk(schema.name, config)

        skill_registry.register(
            name=schema.name,
            description=f"[MCP: {config.name}] {schema.description}",
            category=f"mcp.{config.name}",
            risk_level=risk,
            parameters=schema.parameters,
        )(_mcp_handler)

        log.debug("mcp.tool.registered", tool=schema.name, server=config.name)


def _infer_risk(tool_name: str, config: MCPServerConfig) -> RiskLevel:
    """Heuristically determine risk level from tool name."""
    name_lower = tool_name.lower()
    if any(word in name_lower for word in ("delete", "remove", "destroy", "format", "drop")):
        return RiskLevel.CRITICAL
    if any(word in name_lower for word in ("write", "create", "update", "execute", "run")):
        return RiskLevel.HIGH
    if any(word in name_lower for word in ("render", "generate", "modify", "edit")):
        return RiskLevel.MEDIUM
    return RiskLevel.LOW
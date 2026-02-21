"""
tools/__init__.py — NeuralClaw Tool System

Public interface for the tool system.

Usage:
    from tools import setup_tools, registry, ToolBus
    from safety.safety_kernel import SafetyKernel

    # Set up everything
    kernel = SafetyKernel(allowed_paths=["~/agent_files"])
    setup_tools()  # registers all built-in tools
    bus = ToolBus(registry, kernel)

    # Dispatch a tool call
    result = await bus.dispatch(tool_call, trust_level=TrustLevel.LOW)
"""

from __future__ import annotations

from tools.tool_registry import registry
from tools.types import (
    RiskLevel,
    SafetyDecision,
    SafetyStatus,
    ToolCall,
    ToolResult,
    ToolSchema,
    TrustLevel,
)

__all__ = [
    "registry",
    "setup_tools",
    # Types
    "RiskLevel",
    "TrustLevel",
    "SafetyStatus",
    "SafetyDecision",
    "ToolCall",
    "ToolResult",
    "ToolSchema",
]


def setup_tools(
    enable_filesystem: bool = True,
    enable_terminal: bool = True,
    enable_search: bool = True,
) -> None:
    """
    Import and register all built-in tools.

    Call once at startup before creating the ToolBus.
    Only import the modules you want active — importing registers them.

    Args:
        enable_filesystem: Register file_read, file_write, list_dir tools.
        enable_terminal:   Register terminal_exec tool.
        enable_search:     Register web_search tool.
    """
    if enable_filesystem:
        import tools.filesystem  # noqa: F401 — side effect: registers tools

    if enable_terminal:
        import tools.terminal  # noqa: F401 — side effect: registers tools

    if enable_search:
        import tools.search  # noqa: F401 — side effect: registers tools
"""
kernel/ — NeuralClaw Agent Kernel Public API

This package is the single, stable public entry-point for wiring and starting
the NeuralClaw agent kernel. External callers (main.py, tests, future
integrations) import from here; they never need to reach into the individual
sub-packages directly.

Exports:
    AgentKernel  — assembled kernel instance ready to handle turns
    KernelConfig — typed config surface derived from config.Settings
"""

from kernel.kernel import AgentKernel, KernelConfig

__all__ = ["AgentKernel", "KernelConfig"]
"""
skills/__init__.py — NeuralClaw Skills System

Public interface for the skills module.

The skills system is the new, unified successor to the tools/ bus.
Skills are self-describing, risk-annotated capabilities that the agent
can invoke through the SkillBus pipeline.

Usage:
    from neuralclaw.skills import SkillRegistry, SkillBus, SkillLoader
    from neuralclaw.skills.types import RiskLevel, TrustLevel, SkillCall, SkillResult

    loader = SkillLoader(registry)
    loader.load_builtins()
    loader.load_plugins()

    bus = SkillBus(registry, safety_kernel)
    result = await bus.dispatch(skill_call, trust_level=TrustLevel.LOW)
"""

from neuralclaw.skills.registry import SkillRegistry
from neuralclaw.skills.bus import SkillBus
from neuralclaw.skills.loader import SkillLoader
from neuralclaw.skills.base import SkillBase
from neuralclaw.skills.types import (
    RiskLevel,
    TrustLevel,
    SkillCall,
    SkillResult,
    SkillManifest,
    SkillNotFoundError,
)

__all__ = [
    "SkillRegistry",
    "SkillBus",
    "SkillLoader",
    "SkillBase",
    # Types
    "RiskLevel",
    "TrustLevel",
    "SkillCall",
    "SkillResult",
    "SkillManifest",
    "SkillNotFoundError",
]
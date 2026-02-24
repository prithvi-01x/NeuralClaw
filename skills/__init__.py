"""
skills/__init__.py — NeuralClaw Skills System

Public interface for the skills module.

The skills system is the new, unified successor to the tools/ bus.
Skills are self-describing, risk-annotated capabilities that the agent
can invoke through the SkillBus pipeline.

Usage:
    from skills import SkillRegistry, SkillBus, SkillLoader
    from skills.types import RiskLevel, TrustLevel, SkillCall, SkillResult

    loader = SkillLoader(registry)
    loader.load_builtins()
    loader.load_plugins()

    bus = SkillBus(registry, safety_kernel)
    result = await bus.dispatch(skill_call, trust_level=TrustLevel.LOW)
"""

from skills.registry import SkillRegistry
from skills.bus import SkillBus
from skills.loader import SkillLoader
from skills.base import SkillBase
from skills.types import (
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
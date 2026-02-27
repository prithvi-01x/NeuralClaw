"""
kernel/bootstrap.py — Agent Stack Factory

Shared factory function that wires up the full agent stack pipeline
from settings. Used by both CLI and Telegram interfaces to avoid
duplicating the SkillLoader → MdSkillLoader → SkillBus → SafetyKernel →
Orchestrator chain.

Usage:
    from neuralclaw.kernel.bootstrap import bootstrap_agent_stack
    stack = bootstrap_agent_stack(settings, llm_client, on_response=callback)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from neuralclaw.skills.bus import SkillBus
from neuralclaw.skills.loader import SkillLoader
from neuralclaw.skills.md_loader import MarkdownSkillLoader
from neuralclaw.skills.clawhub.bridge_loader import ClawhubBridgeLoader
from neuralclaw.skills.registry import SkillRegistry
from neuralclaw.safety.safety_kernel import SafetyKernel
from neuralclaw.agent.orchestrator import Orchestrator


@dataclass
class AgentStack:
    """All wired components returned by bootstrap_agent_stack()."""
    orchestrator: Orchestrator
    skill_bus: SkillBus
    skill_registry: SkillRegistry
    safety_kernel: SafetyKernel


def bootstrap_agent_stack(
    settings,
    llm_client,
    memory_manager,
    on_response: Optional[Callable] = None,
    *,
    strict_skills: bool = True,
    strict_md_skills: bool = False,
) -> AgentStack:
    """
    Wire up the full agent stack from settings.

    Args:
        settings:          Loaded NeuralClaw Settings object.
        llm_client:        Pre-created LLM client instance.
        memory_manager:    Initialised MemoryManager instance.
        on_response:       Optional callback for mid-turn streaming updates.
        strict_skills:     If True, broken Python skill files raise at startup.
        strict_md_skills:  If True, broken markdown skill files raise at startup.

    Returns:
        AgentStack with all wired components.
    """
    import warnings
    warnings.warn(
        "bootstrap_agent_stack() is deprecated and will be removed in a future release. "
        "Use AgentKernel.build() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Safety kernel
    allowed_paths = settings.tools.filesystem.allowed_paths
    extra_commands = settings.tools.terminal.whitelist_extra
    safety_kernel = SafetyKernel(
        allowed_paths=allowed_paths,
        whitelist_extra=extra_commands,
        require_confirmation_for=settings.safety.require_confirmation_for,
    )

    # Skill registry + bus — loaded from builtin/ and plugins/
    _base = Path(__file__).parent.parent
    skill_registry = SkillLoader().load_all(
        [
            _base / "skills" / "builtin",
            _base / "skills" / "plugins",
        ],
        strict=strict_skills,
    )
    # Also load markdown skills (OpenClaw-compatible SKILL.md format)
    MarkdownSkillLoader().load_all(
        [_base / "skills" / "plugins"],
        registry=skill_registry,
        strict=strict_md_skills,
    )
    skill_bus = SkillBus(
        registry=skill_registry,
        safety_kernel=safety_kernel,
        default_timeout_seconds=settings.tools.terminal.default_timeout_seconds,
    )
    # Load ClawHub skills (needs skill_bus injected)
    ClawhubBridgeLoader().load_all(
        skills_dir=Path(settings.clawhub.skills_dir),
        registry=skill_registry,
        settings=settings,
        llm_client=llm_client,
        skill_bus=skill_bus,
    )

    # Orchestrator — wired to SkillBus + SkillRegistry
    orchestrator = Orchestrator.from_settings(
        settings=settings,
        llm_client=llm_client,
        tool_bus=skill_bus,
        tool_registry=skill_registry,
        memory_manager=memory_manager,
        on_response=on_response,
    )

    return AgentStack(
        orchestrator=orchestrator,
        skill_bus=skill_bus,
        skill_registry=skill_registry,
        safety_kernel=safety_kernel,
    )


@dataclass
class SharedSkills:
    """Shared skill infrastructure returned by load_shared_skills()."""
    skill_registry: SkillRegistry
    safety_kernel: SafetyKernel
    skill_bus: SkillBus


def load_shared_skills(
    settings,
    *,
    strict_skills: bool = True,
    strict_md_skills: bool = False,
) -> SharedSkills:
    """
    Load SkillRegistry, SafetyKernel, and SkillBus once.

    These are stateless / thread-safe and can be shared across
    multiple Orchestrator instances (e.g. per-chat in Telegram).
    """
    allowed_paths = settings.tools.filesystem.allowed_paths
    extra_commands = settings.tools.terminal.whitelist_extra
    safety_kernel = SafetyKernel(
        allowed_paths=allowed_paths,
        whitelist_extra=extra_commands,
        require_confirmation_for=settings.safety.require_confirmation_for,
    )

    _base = Path(__file__).parent.parent
    skill_registry = SkillLoader().load_all(
        [
            _base / "skills" / "builtin",
            _base / "skills" / "plugins",
        ],
        strict=strict_skills,
    )
    MarkdownSkillLoader().load_all(
        [_base / "skills" / "plugins"],
        registry=skill_registry,
        strict=strict_md_skills,
    )
    skill_bus = SkillBus(
        registry=skill_registry,
        safety_kernel=safety_kernel,
        default_timeout_seconds=settings.tools.terminal.default_timeout_seconds,
    )
    # Note: HttpApiExecutor needs an LLM client, but load_shared_skills doesn't have one.
    # We pass None; HttpApiExecutor handles it gracefully.
    ClawhubBridgeLoader().load_all(
        skills_dir=Path(settings.clawhub.skills_dir),
        registry=skill_registry,
        settings=settings,
        llm_client=None,
        skill_bus=skill_bus,
    )

    return SharedSkills(
        skill_registry=skill_registry,
        safety_kernel=safety_kernel,
        skill_bus=skill_bus,
    )

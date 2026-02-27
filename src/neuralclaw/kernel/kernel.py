"""
kernel/kernel.py — AgentKernel

Single assembly point that wires every NeuralClaw sub-system into a ready-to-use
kernel object. Interfaces (CLI, Telegram) and tests obtain an Orchestrator by
calling AgentKernel.build(settings) rather than importing from the individual
sub-packages directly.

This keeps main.py, interfaces, and integration tests decoupled from the internal
module layout so the kernel can be refactored without touching callers.

Design principles
-----------------
* No business logic here — pure wiring.
* All sub-systems receive their dependencies via constructor injection.
* AgentKernel.build() is the only place that reads from Settings; every other
  module is settings-unaware.
* Fail loudly at startup: missing dependencies → typed exception, never silent.

Usage::

    from neuralclaw.kernel import AgentKernel
    from neuralclaw.config.settings import load_settings

    settings = load_settings()
    kernel   = await AgentKernel.build(settings)
    turn     = await kernel.orchestrator.run_turn(session, "Hello")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from neuralclaw.observability.logger import get_logger
from neuralclaw.exceptions import NeuralClawError

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# KernelConfig — typed subset of Settings consumed by the kernel
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class KernelConfig:
    """
    Typed configuration surface for the agent kernel.

    Extracted from Settings so the kernel never imports Pydantic models or
    touches config.yaml keys directly.
    """
    # Agent loop
    max_iterations_per_turn: int
    max_turn_timeout_seconds: int
    confirmation_timeout_seconds: int
    default_trust_level: str

    # LLM
    llm_provider: str
    llm_model: str
    llm_temperature: float
    llm_max_tokens: int

    # Memory
    chroma_persist_dir: str
    sqlite_path: str
    embedding_model: str
    max_short_term_turns: int
    relevance_threshold: float

    # Safety
    allowed_paths: list
    whitelist_extra: list

    # Skills
    skill_dirs: list

    # Scheduler
    scheduler_max_concurrent: int = 3
    scheduler_timezone: str = "UTC"

    # Optional
    serpapi_key: Optional[str] = None
    ollama_base_url: Optional[str] = None

    @classmethod
    def from_settings(cls, settings) -> "KernelConfig":
        """Build a KernelConfig from a config.Settings instance."""
        skill_dirs = [
            str(Path(__file__).parent.parent / "skills" / "builtin"),
            str(Path(__file__).parent.parent / "skills" / "plugins"),
        ]
        return cls(
            max_iterations_per_turn=settings.agent.max_iterations_per_turn,
            max_turn_timeout_seconds=settings.agent.max_turn_timeout_seconds,
            confirmation_timeout_seconds=settings.agent.confirmation_timeout_seconds,
            default_trust_level=settings.agent.default_trust_level,
            llm_provider=settings.llm.default_provider,
            llm_model=settings.llm.default_model,
            llm_temperature=settings.llm.temperature,
            llm_max_tokens=settings.llm.max_tokens,
            chroma_persist_dir=settings.memory.chroma_persist_dir,
            sqlite_path=settings.memory.sqlite_path,
            embedding_model=settings.memory.embedding_model,
            max_short_term_turns=settings.memory.max_short_term_turns,
            relevance_threshold=settings.memory.relevance_threshold,
            allowed_paths=settings.tools.filesystem.allowed_paths,
            whitelist_extra=settings.tools.terminal.whitelist_extra,
            skill_dirs=skill_dirs,
            scheduler_max_concurrent=settings.scheduler.max_concurrent_tasks,
            scheduler_timezone=settings.scheduler.timezone,
            serpapi_key=settings.serpapi_key,
            ollama_base_url=getattr(settings, "ollama_base_url", None),
        )


# ─────────────────────────────────────────────────────────────────────────────
# AgentKernel
# ─────────────────────────────────────────────────────────────────────────────

class AgentKernel:
    """
    Fully assembled NeuralClaw agent kernel.

    Holds references to every sub-system. The Orchestrator is the primary
    entry-point for interfaces; the other attributes are exposed for testing
    and introspection.

    Do not instantiate directly — use AgentKernel.build(settings).
    """

    def __init__(
        self,
        config: KernelConfig,
        llm_client,
        memory_manager,
        skill_registry,
        skill_bus,
        safety_kernel,
        orchestrator,
        scheduler=None,
    ) -> None:
        self.config         = config
        self.llm_client     = llm_client
        self.memory_manager = memory_manager
        self.skill_registry = skill_registry
        self.skill_bus      = skill_bus
        self.safety_kernel  = safety_kernel
        self.orchestrator   = orchestrator
        self.scheduler      = scheduler

    # ─────────────────────────────────────────────────────────────────────────
    # Factory
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    async def build(cls, settings) -> "AgentKernel":
        """
        Assemble the full kernel from a Settings instance.

        Raises NeuralClawError (or a typed subclass) if any sub-system fails
        to initialise. Never swallows errors silently.
        """
        log.info("kernel.build.start")

        cfg = KernelConfig.from_settings(settings)

        # ── LLM client ───────────────────────────────────────────────────────
        from neuralclaw.brain import LLMClientFactory
        llm_client = LLMClientFactory.from_settings(settings)
        log.info("kernel.llm_ready", provider=cfg.llm_provider, model=cfg.llm_model)

        # ── Memory manager ───────────────────────────────────────────────────
        from neuralclaw.memory.memory_manager import MemoryManager
        memory_manager = MemoryManager.from_settings(settings)
        await memory_manager.init()
        log.info("kernel.memory_ready")

        # ── Skill registry + loader ──────────────────────────────────────────
        from neuralclaw.skills.registry import SkillRegistry
        from neuralclaw.skills.loader import SkillLoader
        skill_registry = SkillRegistry()
        loader = SkillLoader()
        skill_dirs = [Path(d) for d in cfg.skill_dirs if Path(d).exists()]
        loaded_registry = loader.load_all(skill_dirs)
        # Merge loaded skills into our registry instance
        for name, skill_cls in loaded_registry._skills.items():
            skill_registry.register(skill_cls)
        log.info("kernel.skills_ready", count=len(skill_registry._skills))

        # ── Safety kernel ────────────────────────────────────────────────────
        from neuralclaw.safety.safety_kernel import SafetyKernel
        safety_kernel = SafetyKernel(
            allowed_paths=cfg.allowed_paths,
            whitelist_extra=cfg.whitelist_extra,
            require_confirmation_for=settings.safety.require_confirmation_for,
        )
        log.info("kernel.safety_ready")

        # ── Skill bus ────────────────────────────────────────────────────────
        from neuralclaw.skills.bus import SkillBus, configure_retry_policy
        skill_bus = SkillBus(
            registry=skill_registry,
            safety_kernel=safety_kernel,
        )
        # Apply config.yaml skills.retry.* to the bus's retry policy
        configure_retry_policy(settings)
        log.info("kernel.bus_ready")

        # ── ClawHub bridge loader ────────────────────────────────────────────
        if hasattr(settings, "clawhub") and settings.clawhub.enabled:
            try:
                from neuralclaw.skills.clawhub.bridge_loader import ClawhubBridgeLoader
                from neuralclaw.brain.types import LLMConfig as _BrainCfg
                clawhub_loader = ClawhubBridgeLoader()
                _clawhub_llm_cfg = _BrainCfg(
                    model=cfg.llm_model,
                    temperature=cfg.llm_temperature,
                    max_tokens=cfg.llm_max_tokens,
                )
                clawhub_loader.load_all(
                    skills_dir=Path(settings.clawhub.skills_dir),
                    registry=skill_registry,
                    settings=settings,
                    llm_client=llm_client,
                    llm_config=_clawhub_llm_cfg,
                    skill_bus=skill_bus,
                )
                log.info("kernel.clawhub_ready",
                         skills_dir=settings.clawhub.skills_dir)
            except Exception as e:
                log.warning("kernel.clawhub_load_failed", error=str(e))

        # ── Orchestrator (wires planner, executor, reflector internally) ─────
        from neuralclaw.brain.types import LLMConfig as BrainLLMConfig
        from neuralclaw.agent.orchestrator import Orchestrator
        llm_config = BrainLLMConfig(
            model=cfg.llm_model,
            temperature=cfg.llm_temperature,
            max_tokens=cfg.llm_max_tokens,
        )
        orchestrator = Orchestrator(
            llm_client=llm_client,
            llm_config=llm_config,
            tool_bus=skill_bus,
            tool_registry=skill_registry,
            memory_manager=memory_manager,
            max_iterations=cfg.max_iterations_per_turn,
            max_turn_timeout=cfg.max_turn_timeout_seconds,
        )
        log.info("kernel.orchestrator_ready")

        # ── Scheduler ────────────────────────────────────────────────────────
        from neuralclaw.scheduler.scheduler import TaskScheduler
        scheduler = TaskScheduler(
            orchestrator=orchestrator,
            memory_manager=memory_manager,
            max_concurrent_tasks=cfg.scheduler_max_concurrent,
            timezone=cfg.scheduler_timezone,
        )
        log.info("kernel.scheduler_ready")

        log.info("kernel.build.complete")
        return cls(
            config=cfg,
            llm_client=llm_client,
            memory_manager=memory_manager,
            skill_registry=skill_registry,
            skill_bus=skill_bus,
            safety_kernel=safety_kernel,
            orchestrator=orchestrator,
            scheduler=scheduler,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    async def shutdown(self) -> None:
        """Gracefully shut down all sub-systems that require cleanup."""
        log.info("kernel.shutdown.start")
        if self.scheduler:
            try:
                await self.scheduler.stop()
            except Exception as e:
                log.warning("kernel.shutdown.scheduler_stop_failed", error=str(e))
        try:
            await self.memory_manager.close()
        except NeuralClawError as e:
            log.warning("kernel.shutdown.memory_close_failed", error=str(e))
        log.info("kernel.shutdown.complete")
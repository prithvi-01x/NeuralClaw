"""
skills/clawhub/bridge_loader.py — ClawHub Skill Discovery & Registration

Discovers ClawHub skill folders in the configured skills_dir and registers
each one as a ClawhubSkill instance in the shared SkillRegistry.

Called at NeuralClaw startup alongside SkillLoader and MarkdownSkillLoader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from neuralclaw.skills.clawhub.bridge_parser import parse_clawhub_skill_md, _sanitize_name
from neuralclaw.skills.clawhub.clawhub_skill import ClawhubSkill, build_neuralclaw_manifest
from neuralclaw.skills.clawhub.bridge_executor import create_executor
from neuralclaw.skills.registry import SkillRegistry

try:
    from neuralclaw.observability.logger import get_logger as _get_logger
    _log_raw = _get_logger(__name__)
    _STRUCTLOG = True
except ImportError:
    import logging as _logging
    _log_raw = _logging.getLogger(__name__)
    _STRUCTLOG = False


def _log(level: str, event: str, **kwargs) -> None:
    if _STRUCTLOG:
        getattr(_log_raw, level)(event, **kwargs)
    else:
        extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
        getattr(_log_raw, level)("%s %s", event, extra)


class ClawhubBridgeLoader:
    """
    Discovers ClawHub skill folders in `skills_dir` and registers them
    as ClawhubSkill instances in the shared SkillRegistry.
    """

    def load_all(
        self,
        skills_dir: Path,
        registry: SkillRegistry,
        settings=None,
        llm_client=None,
        llm_config=None,
        skill_bus=None,
    ) -> SkillRegistry:
        """
        Walk skills_dir. For each subdirectory containing a SKILL.md:
          1. Parse with bridge_parser.py → ClawhubSkillManifest
          2. Build SkillManifest via build_neuralclaw_manifest()
          3. Instantiate ClawhubSkill with the manifest + executor
          4. Register in SkillRegistry
          5. Log: "clawhub_bridge.loaded skill=todoist_cli tier=2"

        Skips dirs with no SKILL.md (warns but doesn't crash).
        Skips skills with names that clash with existing registered skills (warns).
        """
        skills_dir = Path(skills_dir)
        if not skills_dir.exists():
            _log("debug", "clawhub_loader.dir_not_found", path=str(skills_dir))
            return registry

        # Determine auto-install setting
        auto_install = False
        if settings and hasattr(settings, "clawhub"):
            auto_install = settings.clawhub.execution.auto_install_deps

        loaded = 0
        skipped = 0

        for subdir in sorted(skills_dir.iterdir()):
            if not subdir.is_dir():
                continue
            if subdir.name.startswith(("_", ".")):
                continue

            # Check for SKILL.md
            skill_md = subdir / "SKILL.md"
            if not skill_md.exists():
                skill_md = subdir / "skill.md"
            if not skill_md.exists():
                continue

            try:
                # Parse
                cm = parse_clawhub_skill_md(subdir)

                # Build NeuralClaw manifest
                nc_manifest = build_neuralclaw_manifest(cm, settings)

                # Check for name clashes
                if registry.is_registered(nc_manifest.name):
                    _log("warning", "clawhub_loader.name_clash",
                         skill=nc_manifest.name, dir=str(subdir))
                    skipped += 1
                    continue

                # Create executor for this tier
                executor = create_executor(
                    tier=cm.execution_tier,
                    llm_client=llm_client,
                    llm_config=llm_config,
                    skill_bus=skill_bus,
                    auto_install=auto_install,
                )

                # Build a unique subclass for this skill
                class_name = "".join(
                    part.capitalize() for part in nc_manifest.name.split("_")
                ) + "ClawhubSkill"

                skill_cls = type(
                    class_name,
                    (ClawhubSkill,),
                    {"manifest": nc_manifest},
                )

                # Create instance and attach per-instance fields
                instance = skill_cls()
                instance._clawhub_manifest = cm
                instance._executor = executor

                # Register
                registry.register(instance)
                loaded += 1
                _log("info", "clawhub_loader.loaded",
                     skill=nc_manifest.name,
                     tier=cm.execution_tier,
                     dir=str(subdir))

            except Exception as e:
                _log("warning", "clawhub_loader.skip",
                     dir=str(subdir), error=str(e))
                skipped += 1

        _log("info", "clawhub_loader.complete",
             loaded=loaded, skipped=skipped,
             skills_dir=str(skills_dir))
        return registry

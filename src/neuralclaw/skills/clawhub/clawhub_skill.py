"""
skills/clawhub/clawhub_skill.py — ClawhubSkill SkillBase Subclass

A single SkillBase wrapper that adapts any ClawHub skill to NeuralClaw.
One class, different behavior based on execution_tier, driven by the
assigned BridgeExecutor.
"""

from __future__ import annotations

import re
from typing import ClassVar, Optional, TYPE_CHECKING

from neuralclaw.skills.base import SkillBase
from neuralclaw.skills.types import RiskLevel, SkillManifest, SkillResult

if TYPE_CHECKING:
    from neuralclaw.skills.clawhub.bridge_parser import ClawhubSkillManifest
    from neuralclaw.skills.clawhub.bridge_executor import BridgeExecutor


# ─────────────────────────────────────────────────────────────────────────────
# Name sanitizer
# ─────────────────────────────────────────────────────────────────────────────

def _sanitize_name(raw: str) -> str:
    """Convert 'todoist-cli' → 'todoist_cli' for NeuralClaw skill names."""
    name = re.sub(r"[^a-zA-Z0-9_]", "_", raw).lower().strip("_")
    name = re.sub(r"_+", "_", name)
    return name or "unknown_skill"


# ─────────────────────────────────────────────────────────────────────────────
# Manifest builder
# ─────────────────────────────────────────────────────────────────────────────

def _derive_capabilities(cm: "ClawhubSkillManifest") -> frozenset:
    """
    Only tier-3 (binary/shell) ClawHub skills require a capability gate.
    Tier-1 (prompt-only) and tier-2 (HTTP/API) skills are safe enough to be
    visible and callable without an explicit /grant. The safety kernel still
    enforces risk×trust gates at execution time regardless of this setting.
    """
    if cm.execution_tier == 3:
        return frozenset({"shell:run"})
    return frozenset()


def build_neuralclaw_manifest(
    cm: "ClawhubSkillManifest",
    settings=None,
) -> SkillManifest:
    """
    Build a NeuralClaw SkillManifest from a parsed ClawhubSkillManifest.

    Uses the tier-to-risk mapping from settings if available,
    otherwise uses sensible defaults.
    """
    risk_map = {
        1: RiskLevel.LOW,
        2: RiskLevel.LOW,
        3: RiskLevel.HIGH,
    }

    # Override from settings if available
    if settings and hasattr(settings, "clawhub"):
        rd = settings.clawhub.risk_defaults
        _risk_str_map = {
            "LOW": RiskLevel.LOW,
            "MEDIUM": RiskLevel.MEDIUM,
            "HIGH": RiskLevel.HIGH,
            "CRITICAL": RiskLevel.CRITICAL,
        }
        risk_map[1] = _risk_str_map.get(rd.prompt_only.upper(), RiskLevel.LOW)
        risk_map[2] = _risk_str_map.get(rd.api_http.upper(), RiskLevel.LOW)
        risk_map[3] = _risk_str_map.get(rd.binary_execution.upper(), RiskLevel.HIGH)

    # Sanitize version
    version = cm.version or "1.0.0"
    if not re.match(r"^\d+\.\d+\.\d+$", version):
        version = "1.0.0"

    return SkillManifest(
        name=_sanitize_name(cm.name),
        version=version,
        description=cm.description or f"ClawHub skill: {cm.name}",
        category="clawhub",
        risk_level=risk_map.get(cm.execution_tier, RiskLevel.LOW),
        capabilities=_derive_capabilities(cm),
        parameters={
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "What you want the skill to do",
                },
            },
            "required": ["request"],
        },
        requires_confirmation=(cm.execution_tier == 3),
        enabled=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ClawhubSkill
# ─────────────────────────────────────────────────────────────────────────────

class ClawhubSkill(SkillBase):
    """
    A NeuralClaw SkillBase wrapper for a ClawHub SKILL.md skill.

    Dynamically created per-skill by bridge_loader.py.
    The manifest is built from the parsed ClawhubSkillManifest.
    The execute() method dispatches to the appropriate tier executor.
    """

    manifest: ClassVar[SkillManifest]

    # Set per-instance by bridge_loader (not ClassVar)
    _clawhub_manifest: Optional["ClawhubSkillManifest"] = None
    _executor: Optional["BridgeExecutor"] = None

    async def execute(self, **kwargs) -> SkillResult:
        """Dispatch to the tier-appropriate executor."""
        call_id = kwargs.get("_skill_call_id", "")

        if self._executor is None or self._clawhub_manifest is None:
            return SkillResult.fail(
                skill_name=self.manifest.name,
                skill_call_id=call_id,
                error="ClawhubSkill not properly initialized (missing executor)",
                error_type="ClawhubConfigError",
            )

        try:
            return await self._executor.run(
                self._clawhub_manifest,
                kwargs,
            )
        except Exception as e:
            return SkillResult.fail(
                skill_name=self.manifest.name,
                skill_call_id=call_id,
                error=f"ClawHub execution error: {e}",
                error_type=type(e).__name__,
            )

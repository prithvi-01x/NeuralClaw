from neuralclaw.skills.base import SkillBase
from neuralclaw.skills.types import SkillManifest, SkillResult, RiskLevel

class EchoSkill(SkillBase):
    manifest = SkillManifest(
        name="echo_test",
        version="1.0.0",
        description="Echo text",
        category="test",
        risk_level=RiskLevel.LOW,
        capabilities=frozenset(),
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string"}
            },
            "required": ["text"],
        },
    )

    async def execute(self, text: str) -> SkillResult:
        return SkillResult(success=True, output=text)

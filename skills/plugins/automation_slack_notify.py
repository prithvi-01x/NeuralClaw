"""
skills/plugins/automation_slack_notify.py — Automation: Slack Notify

Sends a message to a Slack channel via an Incoming Webhook URL.
The webhook URL is the only credential needed — no bot token required.

Risk: LOW — net:post
"""
from __future__ import annotations
import time
from typing import ClassVar
from skills.base import SkillBase
from skills.types import RiskLevel, SkillManifest, SkillResult, SkillValidationError

class AutomationSlackNotifySkill(SkillBase):
    manifest: ClassVar[SkillManifest] = SkillManifest(
        name="automation_slack_notify",
        version="1.0.0",
        description="Send a notification to a Slack channel using an Incoming Webhook URL. Supports plain text and Block Kit attachments.",
        category="automation",
        risk_level=RiskLevel.LOW,
        capabilities=frozenset({"net:post"}),
        timeout_seconds=20,
        parameters={"type":"object","properties":{
            "webhook_url":{"type":"string","description":"Slack Incoming Webhook URL (starts with https://hooks.slack.com/)."},
            "message":{"type":"string","description":"Message text to send."},
            "channel":{"type":"string","description":"Override channel (e.g. '#alerts'). Optional — webhook default is used if omitted.","default":""},
            "username":{"type":"string","description":"Bot display name override.","default":"NeuralClaw"},
            "icon_emoji":{"type":"string","description":"Emoji icon (e.g. ':robot_face:').","default":":robot_face:"},
        },"required":["webhook_url","message"]},
    )

    async def validate(self, webhook_url: str, **_) -> None:
        if not webhook_url.startswith("https://hooks.slack.com/"):
            raise SkillValidationError("webhook_url must start with 'https://hooks.slack.com/'.")

    async def execute(self, webhook_url: str, message: str, channel: str="",
                      username: str="NeuralClaw", icon_emoji: str=":robot_face:", **kwargs) -> SkillResult:
        call_id = kwargs.get("_skill_call_id","")
        t_start = time.monotonic()
        payload: dict = {"text":message,"username":username,"icon_emoji":icon_emoji}
        if channel: payload["channel"] = channel

        try:
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.post(webhook_url, json=payload)
                success = r.status_code == 200 and r.text == "ok"
            duration_ms = (time.monotonic()-t_start)*1000
            return SkillResult.ok(self.manifest.name, call_id,
                {"sent":success,"status_code":r.status_code,"response":r.text}, duration_ms=duration_ms)
        except BaseException as e:
            return SkillResult.fail(self.manifest.name, call_id, f"{type(e).__name__}: {e}", type(e).__name__,
                                    duration_ms=(time.monotonic()-t_start)*1000)

"""
skills/plugins/automation_webhook_sender.py — Automation: Webhook Sender

Sends an HTTP POST/GET/PUT request to a webhook URL with a JSON payload.

Risk: MED — net:post
"""
from __future__ import annotations
import json, time
from typing import ClassVar
from neuralclaw.skills.base import SkillBase
from neuralclaw.skills.types import RiskLevel, SkillManifest, SkillResult, SkillValidationError

class AutomationWebhookSenderSkill(SkillBase):
    manifest: ClassVar[SkillManifest] = SkillManifest(
        name="automation_webhook_sender",
        version="1.0.0",
        description="Send an HTTP request to a webhook URL with a JSON or form payload. Supports POST, GET, PUT. Returns response status and body.",
        category="automation",
        risk_level=RiskLevel.MEDIUM,
        capabilities=frozenset({"net:post"}),
        timeout_seconds=30,
        parameters={"type":"object","properties":{
            "url":{"type":"string","description":"Webhook URL to send to."},
            "method":{"type":"string","enum":["POST","GET","PUT","PATCH"],"description":"HTTP method (default POST).","default":"POST"},
            "payload":{"type":"object","description":"JSON payload to send.","default":{}},
            "headers":{"type":"object","description":"Extra HTTP headers.","default":{}},
            "timeout":{"type":"integer","description":"Request timeout seconds (default 15).","default":15},
        },"required":["url"]},
    )

    async def validate(self, url: str, **_) -> None:
        if not url.startswith(("http://","https://")):
            raise SkillValidationError(f"URL must start with http:// or https://, got: '{url}'")

    async def execute(self, url: str, method: str="POST", payload: dict|None=None,
                      headers: dict|None=None, timeout: int=15, **kwargs) -> SkillResult:
        call_id = kwargs.get("_skill_call_id","")
        t_start = time.monotonic()
        timeout = min(int(timeout), 25)
        payload = payload or {}
        headers = {"Content-Type":"application/json","User-Agent":"NeuralClaw/1.0", **(headers or {})}

        try:
            import httpx
            async with httpx.AsyncClient(timeout=float(timeout), follow_redirects=True) as client:
                r = await client.request(method.upper(), url, json=payload, headers=headers)
                try: resp_body = r.json()
                except Exception: resp_body = r.text[:2000]
            duration_ms = (time.monotonic()-t_start)*1000
            return SkillResult.ok(self.manifest.name, call_id, {
                "url":url,"method":method,"status_code":r.status_code,
                "success":200<=r.status_code<300,"response":resp_body
            }, duration_ms=duration_ms)
        except BaseException as e:
            return SkillResult.fail(self.manifest.name, call_id, f"{type(e).__name__}: {e}", type(e).__name__,
                                    duration_ms=(time.monotonic()-t_start)*1000)

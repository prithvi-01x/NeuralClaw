"""
skills/plugins/cyber_http_probe.py — Cyber: HTTP Probe

Probes an HTTP/HTTPS target and returns headers, status code, redirect chain,
server fingerprint, and security header audit.

Risk: MEDIUM — net:fetch capability required.
"""

from __future__ import annotations

import time
from typing import ClassVar
from urllib.parse import urlparse

from neuralclaw.skills.base import SkillBase
from neuralclaw.skills.types import RiskLevel, SkillManifest, SkillResult, SkillValidationError

# Security headers we check for
_SECURITY_HEADERS = [
    "strict-transport-security",
    "content-security-policy",
    "x-frame-options",
    "x-content-type-options",
    "referrer-policy",
    "permissions-policy",
    "x-xss-protection",
]


class CyberHttpProbeSkill(SkillBase):
    manifest: ClassVar[SkillManifest] = SkillManifest(
        name="cyber_http_probe",
        version="1.0.0",
        description=(
            "Probe an HTTP/HTTPS target. Returns status code, response headers, "
            "server fingerprint, redirect chain, TLS info, and a security header audit. "
            "Use for authorized web reconnaissance and vulnerability assessment."
        ),
        category="cybersecurity",
        risk_level=RiskLevel.MEDIUM,
        capabilities=frozenset({"net:fetch"}),
        requires_confirmation=False,
        timeout_seconds=30,
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Target URL to probe (must start with http:// or https://).",
                },
                "follow_redirects": {
                    "type": "boolean",
                    "description": "Follow HTTP redirects (default true).",
                    "default": True,
                },
                "method": {
                    "type": "string",
                    "enum": ["GET", "HEAD"],
                    "description": "HTTP method to use (default HEAD for minimal footprint).",
                    "default": "HEAD",
                },
            },
            "required": ["url"],
        },
    )

    async def validate(self, url: str, **_) -> None:
        if not url.startswith(("http://", "https://")):
            raise SkillValidationError(
                f"URL must start with http:// or https://, got: '{url}'"
            )

    async def execute(
        self,
        url: str,
        follow_redirects: bool = True,
        method: str = "HEAD",
        **kwargs,
    ) -> SkillResult:
        call_id = kwargs.get("_skill_call_id", "")
        t_start = time.monotonic()

        try:
            import httpx

            redirect_chain: list[dict] = []

            async with httpx.AsyncClient(
                timeout=15.0,
                follow_redirects=follow_redirects,
                max_redirects=10,
            ) as client:
                response = await client.request(method, url)

                # Capture redirect history
                for hist in response.history:
                    redirect_chain.append({
                        "url": str(hist.url),
                        "status_code": hist.status_code,
                    })

                headers = dict(response.headers)
                final_url = str(response.url)

                # Security header audit
                present = [h for h in _SECURITY_HEADERS if h in headers]
                missing = [h for h in _SECURITY_HEADERS if h not in headers]

                # Server fingerprint
                server = headers.get("server", "")
                x_powered = headers.get("x-powered-by", "")
                tech = headers.get("x-generator", headers.get("x-drupal-cache", ""))

                # TLS info (only available if httpx exposes it)
                tls_info: dict = {}
                parsed = urlparse(final_url)
                if parsed.scheme == "https":
                    tls_info["tls"] = True

                duration_ms = (time.monotonic() - t_start) * 1000
                return SkillResult.ok(
                    skill_name=self.manifest.name,
                    skill_call_id=call_id,
                    output={
                        "url": url,
                        "final_url": final_url,
                        "status_code": response.status_code,
                        "method": method,
                        "redirect_chain": redirect_chain,
                        "headers": {k: v for k, v in headers.items()},
                        "fingerprint": {
                            "server": server,
                            "x_powered_by": x_powered,
                            "tech_hints": tech,
                        },
                        "security_headers": {
                            "present": present,
                            "missing": missing,
                            "score": f"{len(present)}/{len(_SECURITY_HEADERS)}",
                        },
                        "tls": tls_info,
                    },
                    duration_ms=duration_ms,
                )

        except (OSError, ValueError) as e:
            return SkillResult.fail(
                self.manifest.name, call_id,
                f"{type(e).__name__}: {e}", type(e).__name__,
                duration_ms=(time.monotonic() - t_start) * 1000,
            )
        except BaseException as e:
            return SkillResult.fail(
                self.manifest.name, call_id,
                f"{type(e).__name__}: {e}", type(e).__name__,
                duration_ms=(time.monotonic() - t_start) * 1000,
            )
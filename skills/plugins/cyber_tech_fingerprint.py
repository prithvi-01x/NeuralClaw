"""
skills/plugins/cyber_tech_fingerprint.py — Cyber: Technology Fingerprinter

Identifies web technologies from HTTP headers, cookies, HTML meta tags, and JS files.
No Wappalyzer dependency — pure pattern matching.

Risk: MED — net:fetch
"""
from __future__ import annotations
import re, time
from typing import ClassVar
from skills.base import SkillBase
from skills.types import RiskLevel, SkillManifest, SkillResult, SkillValidationError

_SIGNATURES: dict[str, list[str]] = {
    "nginx":       ["server: nginx","x-powered-by: nginx"],
    "apache":      ["server: apache"],
    "cloudflare":  ["server: cloudflare","cf-ray"],
    "wordpress":   ["x-pingback","wp-content","wp-includes","wordpress"],
    "drupal":      ["x-drupal-cache","x-generator: drupal","drupal.settings"],
    "joomla":      ["joomla","/components/com_"],
    "react":       ["__react","data-reactroot","_react"],
    "next.js":     ["x-powered-by: next.js","__next","_next/static"],
    "django":      ["x-frame-options: sameorigin","csrfmiddlewaretoken","django"],
    "laravel":     ["laravel_session","x-powered-by: php"],
    "rails":       ["x-powered-by: phusion passenger","x-request-id","_rails_"],
    "php":         ["x-powered-by: php","phpsessid"],
    "asp.net":     ["x-aspnet-version","x-aspnetmvc-version","__viewstate"],
    "jquery":      ["jquery"],
    "bootstrap":   ["bootstrap"],
    "google-analytics": ["google-analytics.com/analytics.js","gtag(","ga("],
    "aws":         ["x-amz-","amazonaws.com","s3."],
    "fastly":      ["x-served-by: cache-","x-cache: hit","fastly"],
    "varnish":     ["via: varnish","x-varnish"],
}

class CyberTechFingerprintSkill(SkillBase):
    manifest: ClassVar[SkillManifest] = SkillManifest(
        name="cyber_tech_fingerprint",
        version="1.0.0",
        description="Identify web technologies from HTTP response headers, HTML source, cookies, and meta tags. Returns detected tech stack with confidence indicators.",
        category="cybersecurity",
        risk_level=RiskLevel.MEDIUM,
        capabilities=frozenset({"net:fetch"}),
        timeout_seconds=30,
        parameters={"type":"object","properties":{
            "url":{"type":"string","description":"Target URL to fingerprint."},
        },"required":["url"]},
    )

    async def validate(self, url: str, **_) -> None:
        if not url.startswith(("http://","https://")):
            raise SkillValidationError(f"URL must start with http:// or https://, got: '{url}'")

    async def execute(self, url: str, **kwargs) -> SkillResult:
        call_id = kwargs.get("_skill_call_id","")
        t_start = time.monotonic()

        try:
            import httpx
            from bs4 import BeautifulSoup

            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True, max_redirects=5) as client:
                r = await client.get(url, headers={"User-Agent":"Mozilla/5.0 (compatible; NeuralClaw/1.0)"})
                headers_str = "\n".join(f"{k}: {v}" for k,v in r.headers.items()).lower()
                cookies_str = "; ".join(r.cookies.keys()).lower()
                body = r.text[:50_000].lower()
                status_code = r.status_code

            detected: dict[str, list[str]] = {}
            combined = headers_str + "\n" + cookies_str + "\n" + body

            for tech, sigs in _SIGNATURES.items():
                matches = [s for s in sigs if s.lower() in combined]
                if matches: detected[tech] = matches

            # Extract meta generator
            try:
                soup = BeautifulSoup(r.text[:20_000], "html.parser")
                gen = soup.find("meta", attrs={"name":"generator"})
                if gen and gen.get("content"):
                    detected.setdefault("meta-generator",[]).append(gen["content"])
            except Exception: pass

            # Extract explicit server header
            server = r.headers.get("server","")
            x_powered = r.headers.get("x-powered-by","")

            duration_ms = (time.monotonic()-t_start)*1000
            return SkillResult.ok(self.manifest.name, call_id, {
                "url":url,"status_code":status_code,"detected_technologies":list(detected.keys()),
                "details":detected,"server_header":server,"x_powered_by":x_powered,
                "tech_count":len(detected)
            }, duration_ms=duration_ms)
        except BaseException as e:
            return SkillResult.fail(self.manifest.name, call_id, f"{type(e).__name__}: {e}", type(e).__name__,
                                    duration_ms=(time.monotonic()-t_start)*1000)

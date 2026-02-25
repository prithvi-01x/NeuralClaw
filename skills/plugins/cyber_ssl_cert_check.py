"""
skills/plugins/cyber_ssl_cert_check.py — Cyber: SSL Certificate Check

Retrieves and analyses the TLS certificate of a host.
Returns validity, expiry, SANs, issuer, and chain info.

Risk: LOW — net:fetch
"""
from __future__ import annotations
import asyncio, ssl, socket, time
from datetime import datetime, timezone
from typing import ClassVar
from skills.base import SkillBase
from skills.types import RiskLevel, SkillManifest, SkillResult, SkillValidationError

class CyberSslCertCheckSkill(SkillBase):
    manifest: ClassVar[SkillManifest] = SkillManifest(
        name="cyber_ssl_cert_check",
        version="1.0.0",
        description="Retrieve and analyse the TLS/SSL certificate of a host. Returns validity period, expiry countdown, issuer, subject, SANs, and chain details.",
        category="cybersecurity",
        risk_level=RiskLevel.LOW,
        capabilities=frozenset({"net:fetch"}),
        timeout_seconds=20,
        parameters={"type":"object","properties":{
            "host":{"type":"string","description":"Hostname or IP to check (e.g. 'example.com')."},
            "port":{"type":"integer","description":"Port to connect to (default 443).","default":443},
        },"required":["host"]},
    )

    async def validate(self, host: str, **_) -> None:
        if not host or not host.strip(): raise SkillValidationError("host must be non-empty.")

    async def execute(self, host: str, port: int=443, **kwargs) -> SkillResult:
        call_id = kwargs.get("_skill_call_id","")
        t_start = time.monotonic()
        host = host.strip().lstrip("https://").lstrip("http://").split("/")[0]

        def _get_cert():
            ctx = ssl.create_default_context()
            try:
                with ctx.wrap_socket(socket.create_connection((host, port), timeout=10), server_hostname=host) as s:
                    cert = s.getpeercert()
                    # Also get raw DER for additional info
                    raw = s.getpeercert(binary_form=True)
            except ssl.SSLCertVerificationError as e:
                # Still try to get cert info with unverified context
                ctx2 = ssl.create_default_context()
                ctx2.check_hostname = False
                ctx2.verify_mode = ssl.CERT_NONE
                with ctx2.wrap_socket(socket.create_connection((host, port), timeout=10), server_hostname=host) as s:
                    cert = s.getpeercert()
                    raw = s.getpeercert(binary_form=True)
                    cert["_verification_error"] = str(e)

            not_before_str = cert.get("notBefore","")
            not_after_str  = cert.get("notAfter","")

            def _parse_dt(s):
                try: return datetime.strptime(s, "%b %d %H:%M:%S %Y %Z").replace(tzinfo=timezone.utc)
                except Exception: return None

            not_before = _parse_dt(not_before_str)
            not_after  = _parse_dt(not_after_str)
            now = datetime.now(tz=timezone.utc)
            days_remaining = (not_after - now).days if not_after else None
            valid = not_before <= now <= not_after if (not_before and not_after) else None

            sans = [v for _, v in cert.get("subjectAltName",[])]
            subject = dict(x[0] for x in cert.get("subject",()))
            issuer  = dict(x[0] for x in cert.get("issuer",()))

            return {"host":host,"port":port,"valid":valid,
                    "not_before":not_before_str,"not_after":not_after_str,
                    "days_remaining":days_remaining,"expiring_soon":days_remaining is not None and days_remaining < 30,
                    "subject":subject,"issuer":issuer,"sans":sans,"serial":cert.get("serialNumber",""),
                    "verification_error":cert.get("_verification_error",None)}

        try:
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(loop.run_in_executor(None, _get_cert), timeout=15)
            duration_ms = (time.monotonic()-t_start)*1000
            return SkillResult.ok(self.manifest.name, call_id, result, duration_ms=duration_ms)
        except BaseException as e:
            return SkillResult.fail(self.manifest.name, call_id, f"{type(e).__name__}: {e}", type(e).__name__,
                                    duration_ms=(time.monotonic()-t_start)*1000)

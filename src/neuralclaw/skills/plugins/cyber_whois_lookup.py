"""
skills/plugins/cyber_whois_lookup.py — Cyber: WHOIS Lookup

Performs a WHOIS lookup for a domain or IP address.
Uses python-whois if available, falls back to subprocess whois command.

Risk: LOW — net:fetch
"""
from __future__ import annotations
import asyncio, time
from typing import ClassVar
from neuralclaw.skills.base import SkillBase
from neuralclaw.skills.types import RiskLevel, SkillManifest, SkillResult, SkillValidationError

class CyberWhoisLookupSkill(SkillBase):
    manifest: ClassVar[SkillManifest] = SkillManifest(
        name="cyber_whois_lookup",
        version="1.0.0",
        description="Perform a WHOIS lookup for a domain or IP address. Returns registrar, creation/expiry dates, nameservers, and registrant info.",
        category="cybersecurity",
        risk_level=RiskLevel.LOW,
        capabilities=frozenset({"net:fetch"}),
        timeout_seconds=30,
        parameters={"type":"object","properties":{
            "target":{"type":"string","description":"Domain name or IP address to look up."},
        },"required":["target"]},
    )

    async def validate(self, target: str, **_) -> None:
        if not target or not target.strip(): raise SkillValidationError("target must be non-empty.")

    async def execute(self, target: str, **kwargs) -> SkillResult:
        call_id = kwargs.get("_skill_call_id","")
        t_start = time.monotonic()
        target = target.strip().lower()

        def _whois_python():
            import whois
            w = whois.whois(target)
            return {"target":target,"source":"python-whois",
                    "domain_name":str(w.domain_name),"registrar":str(w.registrar),
                    "creation_date":str(w.creation_date),"expiration_date":str(w.expiration_date),
                    "updated_date":str(w.updated_date),"nameservers":list(w.name_servers or []),
                    "status":str(w.status),"emails":list(w.emails or [])}

        async def _whois_subprocess():
            proc = await asyncio.create_subprocess_exec("whois", target,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=20)
            raw = out.decode(errors="replace")
            result = {"target":target,"source":"whois command","raw":raw[:3000]}
            for line in raw.splitlines():
                l = line.lower()
                if "registrar:" in l: result["registrar"] = line.split(":",1)[1].strip()
                elif "creation date:" in l or "created:" in l: result["creation_date"] = line.split(":",1)[1].strip()
                elif "expiry date:" in l or "expiration date:" in l or "expires:" in l: result["expiration_date"] = line.split(":",1)[1].strip()
                elif "name server:" in l or "nserver:" in l:
                    result.setdefault("nameservers",[]).append(line.split(":",1)[1].strip())
            return result

        try:
            loop = asyncio.get_event_loop()
            try:
                result = await loop.run_in_executor(None, _whois_python)
            except (ImportError, Exception):
                result = await _whois_subprocess()
            duration_ms = (time.monotonic()-t_start)*1000
            return SkillResult.ok(self.manifest.name, call_id, result, duration_ms=duration_ms)
        except BaseException as e:
            return SkillResult.fail(self.manifest.name, call_id, f"{type(e).__name__}: {e}", type(e).__name__,
                                    duration_ms=(time.monotonic()-t_start)*1000)

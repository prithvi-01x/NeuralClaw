"""
skills/plugins/cyber_subdomain_enum.py — Cyber: Subdomain Enumeration

Enumerates subdomains using DNS resolution against a wordlist.
No external tools needed — pure async DNS probing.

Risk: MED — net:fetch
"""
from __future__ import annotations
import asyncio, socket, time
from typing import ClassVar
from skills.base import SkillBase
from skills.types import RiskLevel, SkillManifest, SkillResult, SkillValidationError

_DEFAULT_WORDLIST = ["www","mail","ftp","admin","api","dev","staging","test","app","portal",
    "vpn","remote","blog","shop","store","cdn","static","assets","img","images",
    "m","mobile","secure","login","auth","help","support","docs","wiki","git"]

class CyberSubdomainEnumSkill(SkillBase):
    manifest: ClassVar[SkillManifest] = SkillManifest(
        name="cyber_subdomain_enum",
        version="1.0.0",
        description="Enumerate subdomains of a target domain by DNS resolution. Uses a built-in wordlist or accepts custom entries. Returns resolved subdomains with IP addresses.",
        category="cybersecurity",
        risk_level=RiskLevel.MEDIUM,
        capabilities=frozenset({"net:fetch"}),
        timeout_seconds=60,
        parameters={"type":"object","properties":{
            "domain":{"type":"string","description":"Target domain (e.g. 'example.com')."},
            "wordlist":{"type":"array","items":{"type":"string"},"description":"Custom subdomain wordlist (default: built-in 30-word list).","default":[]},
            "concurrency":{"type":"integer","description":"Concurrent DNS probes (default 50, max 100).","default":50},
        },"required":["domain"]},
    )

    async def validate(self, domain: str, **_) -> None:
        if not domain or not domain.strip(): raise SkillValidationError("domain must be non-empty.")
        if domain.startswith(("http://","https://")): raise SkillValidationError("domain should be bare (e.g. 'example.com'), not a URL.")

    async def execute(self, domain: str, wordlist: list|None=None, concurrency: int=50, **kwargs) -> SkillResult:
        call_id = kwargs.get("_skill_call_id","")
        t_start = time.monotonic()
        concurrency = min(int(concurrency), 100)
        probes = wordlist if wordlist else _DEFAULT_WORDLIST
        domain = domain.strip().lower().rstrip(".")

        found: list[dict] = []
        sem = asyncio.Semaphore(concurrency)

        async def probe(sub: str) -> None:
            fqdn = f"{sub}.{domain}"
            async with sem:
                try:
                    loop = asyncio.get_event_loop()
                    infos = await asyncio.wait_for(
                        loop.run_in_executor(None, socket.getaddrinfo, fqdn, None), timeout=3)
                    ips = list({i[4][0] for i in infos})
                    found.append({"subdomain":fqdn,"ips":ips})
                except (socket.gaierror, asyncio.TimeoutError, OSError):
                    pass

        try:
            await asyncio.gather(*[probe(s) for s in probes])
            found.sort(key=lambda x: x["subdomain"])
            duration_ms = (time.monotonic()-t_start)*1000
            return SkillResult.ok(self.manifest.name, call_id,
                {"domain":domain,"probed":len(probes),"found_count":len(found),"subdomains":found}, duration_ms=duration_ms)
        except BaseException as e:
            return SkillResult.fail(self.manifest.name, call_id, f"{type(e).__name__}: {e}", type(e).__name__,
                                    duration_ms=(time.monotonic()-t_start)*1000)

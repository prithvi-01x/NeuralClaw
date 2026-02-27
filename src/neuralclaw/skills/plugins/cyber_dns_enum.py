"""
skills/plugins/cyber_dns_enum.py — Cyber: DNS Enumeration

Enumerates DNS records for a target domain: A, AAAA, MX, NS, TXT, CNAME, SOA.
Uses Python's stdlib socket + dnspython if available, falls back to socket only.

Risk: MEDIUM — net:fetch capability required.
"""

from __future__ import annotations

import asyncio
import socket
import time
from typing import ClassVar

from neuralclaw.skills.base import SkillBase
from neuralclaw.skills.types import RiskLevel, SkillManifest, SkillResult, SkillValidationError

_RECORD_TYPES = ["A", "AAAA", "MX", "NS", "TXT", "CNAME", "SOA"]


class CyberDnsEnumSkill(SkillBase):
    manifest: ClassVar[SkillManifest] = SkillManifest(
        name="cyber_dns_enum",
        version="1.0.0",
        description=(
            "Enumerate DNS records for a target domain. "
            "Returns A, AAAA, MX, NS, TXT, CNAME, and SOA records. "
            "Use for authorized reconnaissance and infrastructure mapping."
        ),
        category="cybersecurity",
        risk_level=RiskLevel.MEDIUM,
        capabilities=frozenset({"net:fetch"}),
        requires_confirmation=False,
        timeout_seconds=30,
        parameters={
            "type": "object",
            "properties": {
                "domain": {
                    "type": "string",
                    "description": "Target domain to enumerate (e.g. 'example.com').",
                },
                "record_types": {
                    "type": "array",
                    "items": {"type": "string", "enum": _RECORD_TYPES},
                    "description": f"Record types to query. Default: all ({', '.join(_RECORD_TYPES)}).",
                    "default": _RECORD_TYPES,
                },
            },
            "required": ["domain"],
        },
    )

    async def validate(self, domain: str, **_) -> None:
        if not domain or not domain.strip():
            raise SkillValidationError("domain must be a non-empty string.")
        if domain.startswith("http://") or domain.startswith("https://"):
            raise SkillValidationError(
                "domain should be a bare domain name (e.g. 'example.com'), not a URL."
            )

    async def execute(
        self,
        domain: str,
        record_types: list[str] | None = None,
        **kwargs,
    ) -> SkillResult:
        call_id = kwargs.get("_skill_call_id", "")
        t_start = time.monotonic()
        domain = domain.strip().lower().rstrip(".")
        record_types = [r.upper() for r in (record_types or _RECORD_TYPES)]

        try:
            results: dict[str, list] = {}

            # Try dnspython first (richer output), fall back to socket
            try:
                import dns.resolver  # type: ignore
                import dns.exception  # type: ignore

                def _query_sync(rtype: str) -> list[str]:
                    try:
                        answers = dns.resolver.resolve(domain, rtype, lifetime=8)
                        return [r.to_text() for r in answers]
                    except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN,
                            dns.resolver.NoNameservers, dns.exception.Timeout):
                        return []
                    except BaseException:
                        return []

                loop = asyncio.get_event_loop()
                for rtype in record_types:
                    records = await loop.run_in_executor(None, _query_sync, rtype)
                    if records:
                        results[rtype] = records

            except ImportError:
                # Fallback: stdlib socket — A records only
                def _socket_resolve() -> list[str]:
                    try:
                        infos = socket.getaddrinfo(domain, None)
                        return list({info[4][0] for info in infos})
                    except socket.gaierror:
                        return []

                loop = asyncio.get_event_loop()
                a_records = await loop.run_in_executor(None, _socket_resolve)
                if "A" in record_types or "AAAA" in record_types:
                    ipv4 = [ip for ip in a_records if ":" not in ip]
                    ipv6 = [ip for ip in a_records if ":" in ip]
                    if ipv4:
                        results["A"] = ipv4
                    if ipv6:
                        results["AAAA"] = ipv6
                if not results:
                    results["note"] = ["dnspython not installed — only A/AAAA via socket available"]

            duration_ms = (time.monotonic() - t_start) * 1000
            return SkillResult.ok(
                skill_name=self.manifest.name,
                skill_call_id=call_id,
                output={
                    "domain": domain,
                    "records": results,
                    "record_count": sum(len(v) for v in results.values() if isinstance(v, list)),
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
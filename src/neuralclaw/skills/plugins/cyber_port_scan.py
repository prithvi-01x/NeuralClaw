"""
skills/plugins/cyber_port_scan.py — Cyber: Port Scanner

Performs a TCP port scan on a target host over a specified port range.
Uses asyncio for concurrent connection attempts — no nmap dependency required.

Risk: HIGH — net:scan capability required.
Capability gate: must have 'net:scan' in session.granted_capabilities.
requires_confirmation: True — always prompts before scanning.
"""

from __future__ import annotations

import asyncio
import time
from typing import ClassVar

from neuralclaw.skills.base import SkillBase
from neuralclaw.skills.types import RiskLevel, SkillManifest, SkillResult, SkillValidationError

# Hard ceiling — prevent runaway scans
_MAX_PORTS = 1000
_CONNECT_TIMEOUT = 1.0   # seconds per port
_MAX_CONCURRENT = 100    # semaphore width


class CyberPortScanSkill(SkillBase):
    manifest: ClassVar[SkillManifest] = SkillManifest(
        name="cyber_port_scan",
        version="1.0.0",
        description=(
            "Perform a TCP port scan on a target host. "
            "Returns a list of open ports with service banners where available. "
            "Use for authorized penetration testing and network reconnaissance only."
        ),
        category="cybersecurity",
        risk_level=RiskLevel.HIGH,
        capabilities=frozenset({"net:scan"}),
        requires_confirmation=True,
        timeout_seconds=120,
        parameters={
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Hostname or IP address to scan.",
                },
                "ports": {
                    "type": "string",
                    "description": (
                        "Port specification. Examples: '80', '22,80,443', "
                        "'1-1024', '80,443,8000-8100'. Max 1000 ports per scan."
                    ),
                    "default": "1-1024",
                },
                "timeout": {
                    "type": "number",
                    "description": "Per-port connection timeout in seconds (default 1.0, max 5.0).",
                    "default": 1.0,
                },
            },
            "required": ["target"],
        },
    )

    async def validate(self, target: str, ports: str = "1-1024", timeout: float = 1.0, **_) -> None:
        if not target or not target.strip():
            raise SkillValidationError("target must be a non-empty hostname or IP address.")
        if timeout > 5.0:
            raise SkillValidationError("timeout must be ≤ 5.0 seconds.")
        port_list = _parse_ports(ports)
        if not port_list:
            raise SkillValidationError(f"Could not parse port specification: '{ports}'")
        if len(port_list) > _MAX_PORTS:
            raise SkillValidationError(
                f"Port range too large: {len(port_list)} ports requested, max is {_MAX_PORTS}."
            )

    async def execute(
        self,
        target: str,
        ports: str = "1-1024",
        timeout: float = 1.0,
        **kwargs,
    ) -> SkillResult:
        call_id = kwargs.get("_skill_call_id", "")
        t_start = time.monotonic()

        try:
            timeout = min(float(timeout), 5.0)
            port_list = _parse_ports(ports)
            if not port_list or len(port_list) > _MAX_PORTS:
                return SkillResult.fail(
                    self.manifest.name, call_id,
                    f"Invalid or oversized port specification: '{ports}'",
                )

            open_ports: list[dict] = []
            semaphore = asyncio.Semaphore(_MAX_CONCURRENT)

            async def probe(port: int) -> None:
                async with semaphore:
                    try:
                        conn = asyncio.open_connection(target, port)
                        reader, writer = await asyncio.wait_for(conn, timeout=timeout)
                        # Attempt banner grab (100ms window)
                        banner = ""
                        try:
                            data = await asyncio.wait_for(reader.read(256), timeout=0.1)
                            banner = data.decode("utf-8", errors="replace").strip()
                        except (asyncio.TimeoutError, OSError):
                            pass
                        writer.close()
                        try:
                            await writer.wait_closed()
                        except OSError:
                            pass
                        open_ports.append({
                            "port": port,
                            "state": "open",
                            "banner": banner[:200] if banner else "",
                        })
                    except (asyncio.TimeoutError, OSError, ConnectionRefusedError):
                        pass

            await asyncio.gather(*[probe(p) for p in port_list])
            open_ports.sort(key=lambda x: x["port"])

            duration_ms = (time.monotonic() - t_start) * 1000
            return SkillResult.ok(
                skill_name=self.manifest.name,
                skill_call_id=call_id,
                output={
                    "target": target,
                    "ports_scanned": len(port_list),
                    "open_count": len(open_ports),
                    "open_ports": open_ports,
                    "scan_time_ms": round(duration_ms),
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


# ─────────────────────────────────────────────────────────────────────────────
# Port parsing helper
# ─────────────────────────────────────────────────────────────────────────────

def _parse_ports(spec: str) -> list[int]:
    """
    Parse a port specification string into a sorted, deduplicated list of ints.
    Supports: '80', '22,80,443', '1-1024', '80,443,8000-8100'.
    Returns empty list on parse failure.
    """
    ports: set[int] = set()
    try:
        for part in spec.replace(" ", "").split(","):
            if "-" in part:
                lo, hi = part.split("-", 1)
                lo_i, hi_i = int(lo), int(hi)
                if lo_i < 1 or hi_i > 65535 or lo_i > hi_i:
                    return []
                ports.update(range(lo_i, hi_i + 1))
            else:
                p = int(part)
                if p < 1 or p > 65535:
                    return []
                ports.add(p)
    except (ValueError, TypeError):
        return []
    return sorted(ports)
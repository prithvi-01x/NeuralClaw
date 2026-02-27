"""
skills/plugins/cyber_banner_grab.py — Cyber: Banner Grab

Connects to a host:port and reads the service banner.
Supports raw TCP and HTTP banner grabbing.

Risk: MED — net:scan
"""
from __future__ import annotations
import asyncio, time
from typing import ClassVar
from neuralclaw.skills.base import SkillBase
from neuralclaw.skills.types import RiskLevel, SkillManifest, SkillResult, SkillValidationError

class CyberBannerGrabSkill(SkillBase):
    manifest: ClassVar[SkillManifest] = SkillManifest(
        name="cyber_banner_grab",
        version="1.0.0",
        description="Grab the service banner from a host and port. Sends a probe and reads the initial response. Useful for service identification during reconnaissance.",
        category="cybersecurity",
        risk_level=RiskLevel.MEDIUM,
        capabilities=frozenset({"net:scan"}),
        timeout_seconds=20,
        parameters={"type":"object","properties":{
            "host":{"type":"string","description":"Target hostname or IP."},
            "port":{"type":"integer","description":"Target port."},
            "probe":{"type":"string","description":"Data to send before reading (default: HTTP HEAD request). Use '' for passive grab.","default":""},
            "timeout":{"type":"number","description":"Connection timeout seconds (default 5.0, max 10.0).","default":5.0},
            "max_bytes":{"type":"integer","description":"Max bytes to read from banner (default 512).","default":512},
        },"required":["host","port"]},
    )

    async def validate(self, host: str, port: int, timeout: float=5.0, **_) -> None:
        if not host or not host.strip(): raise SkillValidationError("host must be non-empty.")
        if not (1 <= int(port) <= 65535): raise SkillValidationError(f"port must be 1-65535, got: {port}")
        if float(timeout) > 10.0: raise SkillValidationError("timeout must be ≤ 10.0")

    async def execute(self, host: str, port: int, probe: str="", timeout: float=5.0, max_bytes: int=512, **kwargs) -> SkillResult:
        call_id = kwargs.get("_skill_call_id","")
        t_start = time.monotonic()
        timeout = min(float(timeout), 10.0)
        max_bytes = min(int(max_bytes), 4096)

        try:
            reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=timeout)

            # Send probe
            if probe:
                writer.write(probe.encode(errors="replace"))
            elif port == 80:
                writer.write(f"HEAD / HTTP/1.0\r\nHost: {host}\r\n\r\n".encode())
            elif port == 443:
                writer.write(f"HEAD / HTTP/1.0\r\nHost: {host}\r\n\r\n".encode())
            await writer.drain()

            try:
                banner_bytes = await asyncio.wait_for(reader.read(max_bytes), timeout=timeout)
                banner = banner_bytes.decode("utf-8", errors="replace").strip()
            except asyncio.TimeoutError:
                banner = ""

            writer.close()
            try: await writer.wait_closed()
            except OSError: pass

            duration_ms = (time.monotonic()-t_start)*1000
            return SkillResult.ok(self.manifest.name, call_id, {
                "host":host,"port":port,"banner":banner,"banner_length":len(banner),
                "has_banner":bool(banner)
            }, duration_ms=duration_ms)
        except (asyncio.TimeoutError, OSError, ConnectionRefusedError) as e:
            return SkillResult.fail(self.manifest.name, call_id, f"{type(e).__name__}: {e}", type(e).__name__,
                                    duration_ms=(time.monotonic()-t_start)*1000)
        except BaseException as e:
            return SkillResult.fail(self.manifest.name, call_id, f"{type(e).__name__}: {e}", type(e).__name__,
                                    duration_ms=(time.monotonic()-t_start)*1000)

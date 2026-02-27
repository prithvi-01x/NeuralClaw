"""
gateway/gateway_server.py — WebSocket Gateway Server

The core WebSocket server that acts as a control plane between interfaces
and the NeuralClaw orchestrator. Uses the `websockets` library.

Usage:
    server = GatewayServer(settings, orchestrator, session_store, ...)
    await server.start()          # starts listening
    await server.wait_closed()    # blocks until shutdown
"""

from __future__ import annotations

import asyncio
from typing import Optional, Any

import websockets
from websockets.asyncio.server import ServerConnection

from neuralclaw.agent.orchestrator import Orchestrator, TurnResult
from neuralclaw.agent.response_synthesizer import AgentResponse, ResponseKind
from neuralclaw.agent.session import Session
from neuralclaw.gateway.protocol import (
    GatewayMessage,
    MessageType,
    make_response,
    make_confirm_request,
    make_error,
    make_pong,
    make_session_created,
    make_session_updated,
)
from neuralclaw.gateway.session_store import SessionStore
from neuralclaw.skills.types import TrustLevel, KNOWN_CAPABILITIES
from neuralclaw.observability.logger import get_logger

log = get_logger(__name__)


class GatewayServer:
    """
    WebSocket gateway server.

    Accepts WebSocket connections and routes JSON messages to the orchestrator.
    Streams AgentResponse objects back to connected clients.
    """

    def __init__(
        self,
        orchestrator: Orchestrator,
        session_store: SessionStore,
        skill_registry,
        *,
        host: str = "127.0.0.1",
        port: int = 9090,
        auth_token: Optional[str] = None,
        max_connections: int = 10,
    ):
        self._orchestrator = orchestrator
        self._sessions = session_store
        self._registry = skill_registry
        self._host = host
        self._port = port
        self._auth_token = auth_token
        self._max_connections = max_connections
        self._server = None

        # Track connected clients: websocket → set of session_ids subscribed
        self._connections: dict[ServerConnection, set[str]] = {}
        self._running_tasks: dict[str, asyncio.Task] = {}
        # Per-session lock to serialise ask/run calls that mutate _on_response
        self._session_locks: dict[str, asyncio.Lock] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the WebSocket server."""
        self._server = await websockets.serve(
            self._handler,
            self._host,
            self._port,
            max_size=2**20,  # 1 MB max message
        )
        log.info(
            "gateway.started",
            host=self._host,
            port=self._port,
            max_connections=self._max_connections,
        )

    async def wait_closed(self) -> None:
        """Block until the server is closed."""
        if self._server:
            await self._server.wait_closed()

    async def shutdown(self) -> None:
        """Gracefully shut down the server."""
        # Cancel all running tasks
        for task in self._running_tasks.values():
            task.cancel()

        if self._server:
            self._server.close()
            await self._server.wait_closed()
        log.info("gateway.stopped")

    # ─────────────────────────────────────────────────────────────────────────
    # Connection handler
    # ─────────────────────────────────────────────────────────────────────────

    async def _handler(self, websocket: ServerConnection) -> None:
        """Handle a single WebSocket connection."""
        # Check max connections
        if len(self._connections) >= self._max_connections:
            err = make_error("max_connections", "Server at connection limit.")
            await websocket.send(err.to_json())
            await websocket.close()
            return

        # Optional auth
        if self._auth_token:
            try:
                auth_msg = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                parsed = GatewayMessage.from_json(auth_msg)
                token = parsed.data.get("token", "")
                if token != self._auth_token:
                    err = make_error("auth_failed", "Invalid auth token.")
                    await websocket.send(err.to_json())
                    await websocket.close()
                    return
            except (asyncio.TimeoutError, Exception):
                await websocket.close()
                return

        self._connections[websocket] = set()
        remote = getattr(websocket, "remote_address", ("?", 0))
        log.info("gateway.client_connected", remote=str(remote))

        try:
            async for raw in websocket:
                try:
                    msg = GatewayMessage.from_json(raw)
                    await self._route(websocket, msg)
                except Exception as e:
                    err = make_error("parse_error", str(e))
                    await websocket.send(err.to_json())
        except websockets.ConnectionClosed:
            pass
        finally:
            self._connections.pop(websocket, None)
            log.info("gateway.client_disconnected", remote=str(remote))

    # ─────────────────────────────────────────────────────────────────────────
    # Message router
    # ─────────────────────────────────────────────────────────────────────────

    async def _route(self, ws: ServerConnection, msg: GatewayMessage) -> None:
        """Route an incoming message to the appropriate handler."""
        mtype = msg.type

        if mtype == MessageType.PING.value:
            await ws.send(make_pong().to_json())

        elif mtype == MessageType.SESSION_CREATE.value:
            await self._handle_session_create(ws, msg)

        elif mtype == MessageType.ASK.value:
            await self._handle_ask(ws, msg)

        elif mtype == MessageType.RUN.value:
            await self._handle_run(ws, msg)

        elif mtype == MessageType.CANCEL.value:
            await self._handle_cancel(ws, msg)

        elif mtype == MessageType.CONFIRM.value:
            await self._handle_confirm(ws, msg)

        elif mtype == MessageType.SESSION_TRUST.value:
            await self._handle_trust(ws, msg)

        elif mtype == MessageType.SESSION_GRANT.value:
            await self._handle_grant(ws, msg)

        elif mtype == MessageType.SESSION_REVOKE.value:
            await self._handle_revoke(ws, msg)

        elif mtype == MessageType.SESSION_STATUS.value:
            await self._handle_status(ws, msg)

        elif mtype == MessageType.SKILLS_LIST.value:
            await self._handle_skills_list(ws, msg)

        elif mtype == MessageType.SKILLS_RELOAD.value:
            await self._handle_skills_reload(ws, msg)

        elif mtype == MessageType.CONFIG_GET.value:
            await self._handle_config_get(ws, msg)

        elif mtype == MessageType.CONFIG_SET.value:
            await self._handle_config_set(ws, msg)

        elif mtype == MessageType.ENV_LIST.value:
            await self._handle_env_list(ws, msg)

        elif mtype == MessageType.ENV_SET.value:
            await self._handle_env_set(ws, msg)

        elif mtype == MessageType.SYSTEM_INFO.value:
            await self._handle_system_info(ws, msg)

        else:
            err = make_error("unknown_type", f"Unknown message type: {mtype}",
                             reply_to=msg.id)
            await ws.send(err.to_json())

    # ─────────────────────────────────────────────────────────────────────────
    # Session management
    # ─────────────────────────────────────────────────────────────────────────

    async def _handle_session_create(
        self, ws: ServerConnection, msg: GatewayMessage
    ) -> None:
        user_id = msg.data.get("user_id", "gateway")
        trust_str = msg.data.get("trust_level", "low")
        trust_map = {t.value: t for t in TrustLevel}
        trust = trust_map.get(trust_str, TrustLevel.LOW)

        session = await self._sessions.create(user_id=user_id, trust_level=trust)
        self._connections[ws].add(session.id)

        resp = make_session_created(session.id)
        resp.id = msg.id  # echo request ID for correlation
        await ws.send(resp.to_json())

    async def _resolve_session(
        self, ws: ServerConnection, msg: GatewayMessage
    ) -> Session:
        """Resolve the session for a message, auto-creating one if needed."""
        sid = msg.session_id
        if sid:
            session = await self._sessions.get(sid)
            if session:
                self._connections[ws].add(sid)
                return session
        # Auto-create if no session_id or not found
        session = await self._sessions.create()
        self._connections[ws].add(session.id)
        return session

    # ─────────────────────────────────────────────────────────────────────────
    # Ask (interactive turn)
    # ─────────────────────────────────────────────────────────────────────────

    async def _handle_ask(self, ws: ServerConnection, msg: GatewayMessage) -> None:
        message = msg.data.get("message", "")
        if not message:
            err = make_error("missing_field", "Ask requires 'message' in data.",
                             reply_to=msg.id, session_id=msg.session_id)
            await ws.send(err.to_json())
            return

        session = await self._resolve_session(ws, msg)

        # Acquire a per-session lock so concurrent asks don't race on _on_response
        lock = self._session_locks.setdefault(session.id, asyncio.Lock())
        async with lock:
            # Stream response callback (scoped to this call)
            async def on_response(resp: AgentResponse) -> None:
                gw_msg = self._agent_response_to_gw(session.id, resp, reply_to=msg.id)
                await ws.send(gw_msg.to_json())

            # Set the on_response callback for streaming
            old_callback = self._orchestrator._on_response
            self._orchestrator._on_response = lambda r: asyncio.ensure_future(on_response(r))

            try:
                result: TurnResult = await self._orchestrator.run_turn(session, message)
                # Send the final response
                final = make_response(
                    session_id=session.id,
                    kind=result.response.kind.value,
                    text=result.response.text,
                    is_final=True,
                    reply_to=msg.id,
                    extra={
                        "status": result.status.value,
                        "steps_taken": result.steps_taken,
                        "duration_ms": round(result.duration_ms, 1),
                    },
                )
                await ws.send(final.to_json())
            except Exception as e:
                err = make_error("turn_error", str(e),
                                 reply_to=msg.id, session_id=session.id)
                await ws.send(err.to_json())
            finally:
                self._orchestrator._on_response = old_callback

    # ─────────────────────────────────────────────────────────────────────────
    # Run (autonomous mode)
    # ─────────────────────────────────────────────────────────────────────────

    async def _handle_run(self, ws: ServerConnection, msg: GatewayMessage) -> None:
        goal = msg.data.get("goal", "")
        if not goal:
            err = make_error("missing_field", "Run requires 'goal' in data.",
                             reply_to=msg.id, session_id=msg.session_id)
            await ws.send(err.to_json())
            return

        session = await self._resolve_session(ws, msg)

        async def _run_task() -> None:
            try:
                async for resp in self._orchestrator.run_autonomous(session, goal):
                    gw_msg = self._agent_response_to_gw(
                        session.id, resp, reply_to=msg.id
                    )
                    await ws.send(gw_msg.to_json())
            except asyncio.CancelledError:
                cancel_resp = make_response(
                    session_id=session.id,
                    kind="text",
                    text="🛑 Autonomous task cancelled.",
                    is_final=True,
                    reply_to=msg.id,
                )
                await ws.send(cancel_resp.to_json())
            except Exception as e:
                err = make_error("run_error", str(e),
                                 reply_to=msg.id, session_id=session.id)
                await ws.send(err.to_json())
            finally:
                self._running_tasks.pop(session.id, None)

        task = asyncio.create_task(_run_task())
        self._running_tasks[session.id] = task

    # ─────────────────────────────────────────────────────────────────────────
    # Cancel
    # ─────────────────────────────────────────────────────────────────────────

    async def _handle_cancel(self, ws: ServerConnection, msg: GatewayMessage) -> None:
        session = await self._resolve_session(ws, msg)
        task = self._running_tasks.get(session.id)

        if task and not task.done():
            session.cancel()
            task.cancel()
            resp = make_response(
                session_id=session.id,
                kind="text",
                text="🛑 Cancel signal sent.",
                reply_to=msg.id,
            )
        else:
            session._cancel_event.clear()
            resp = make_response(
                session_id=session.id,
                kind="text",
                text="Nothing is currently running to cancel.",
                reply_to=msg.id,
            )
        await ws.send(resp.to_json())

    # ─────────────────────────────────────────────────────────────────────────
    # Confirmation
    # ─────────────────────────────────────────────────────────────────────────

    async def _handle_confirm(
        self, ws: ServerConnection, msg: GatewayMessage
    ) -> None:
        session = await self._resolve_session(ws, msg)
        tool_call_id = msg.data.get("tool_call_id", "")
        approved = msg.data.get("approved", False)

        if not tool_call_id:
            err = make_error("missing_field", "Confirm requires 'tool_call_id'.",
                             reply_to=msg.id)
            await ws.send(err.to_json())
            return

        session.resolve_confirmation(tool_call_id, approved)
        ack = make_response(
            session_id=session.id,
            kind="text",
            text=f"✓ {'Approved' if approved else 'Denied'}: {tool_call_id}",
            reply_to=msg.id,
        )
        await ws.send(ack.to_json())

    # ─────────────────────────────────────────────────────────────────────────
    # Trust / Grant / Revoke / Status
    # ─────────────────────────────────────────────────────────────────────────

    async def _handle_trust(self, ws: ServerConnection, msg: GatewayMessage) -> None:
        session = await self._resolve_session(ws, msg)
        level_str = msg.data.get("level", "").lower()
        trust_map = {t.value: t for t in TrustLevel}

        if level_str not in trust_map:
            err = make_error(
                "invalid_value",
                f"Unknown trust level: '{level_str}'. Valid: {list(trust_map.keys())}",
                reply_to=msg.id,
            )
            await ws.send(err.to_json())
            return

        session.set_trust_level(trust_map[level_str])
        resp = make_session_updated(session.id, trust_level=level_str)
        resp.id = msg.id
        await ws.send(resp.to_json())

    async def _handle_grant(self, ws: ServerConnection, msg: GatewayMessage) -> None:
        session = await self._resolve_session(ws, msg)
        cap = msg.data.get("capability", "")

        if cap not in KNOWN_CAPABILITIES:
            err = make_error(
                "invalid_value",
                f"Unknown capability: '{cap}'. Known: {sorted(KNOWN_CAPABILITIES)}",
                reply_to=msg.id,
            )
            await ws.send(err.to_json())
            return

        session.grant_capability(cap)
        resp = make_session_updated(
            session.id,
            granted=sorted(session.granted_capabilities),
        )
        resp.id = msg.id
        await ws.send(resp.to_json())

    async def _handle_revoke(self, ws: ServerConnection, msg: GatewayMessage) -> None:
        session = await self._resolve_session(ws, msg)
        cap = msg.data.get("capability", "")
        session.revoke_capability(cap)

        resp = make_session_updated(
            session.id,
            granted=sorted(session.granted_capabilities),
        )
        resp.id = msg.id
        await ws.send(resp.to_json())

    async def _handle_status(self, ws: ServerConnection, msg: GatewayMessage) -> None:
        session = await self._resolve_session(ws, msg)
        summary = session.status_summary()
        resp = make_response(
            session_id=session.id,
            kind="status",
            text="Session status",
            reply_to=msg.id,
            extra=summary,
        )
        await ws.send(resp.to_json())

    # ─────────────────────────────────────────────────────────────────────────
    # Skills list
    # ─────────────────────────────────────────────────────────────────────────

    async def _handle_skills_list(
        self, ws: ServerConnection, msg: GatewayMessage
    ) -> None:
        schemas = self._registry.list_schemas(enabled_only=True, granted=None)
        skills = [
            {
                "name": s.name,
                "description": s.description,
                "category": getattr(s, "category", "general"),
                "risk_level": getattr(s, "risk_level", "LOW"),
            }
            for s in schemas
        ]
        resp = make_response(
            session_id=msg.session_id,
            kind="status",
            text=f"{len(skills)} skills available",
            reply_to=msg.id,
            extra={"skills": skills},
        )
        await ws.send(resp.to_json())

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _agent_response_to_gw(
        session_id: str,
        resp: AgentResponse,
        *,
        reply_to: Optional[str] = None,
    ) -> GatewayMessage:
        """Convert an AgentResponse to a GatewayMessage."""
        extra: dict[str, Any] = {}
        if resp.tool_name:
            extra["tool_name"] = resp.tool_name
        if resp.tool_call_id:
            extra["tool_call_id"] = resp.tool_call_id
        if resp.risk_level:
            extra["risk_level"] = resp.risk_level.value
        if resp.metadata:
            extra["metadata"] = resp.metadata

        # Confirmation requests get a special message type
        if resp.kind == ResponseKind.CONFIRMATION:
            return GatewayMessage(
                type=MessageType.CONFIRM_REQUEST.value,
                session_id=session_id,
                data={
                    "tool_call_id": resp.tool_call_id or "",
                    "skill_name": resp.tool_name or "",
                    "risk_level": resp.risk_level.value if resp.risk_level else "HIGH",
                    "reason": resp.metadata.get("reason", ""),
                    "arguments": resp.metadata.get("arguments", {}),
                    "text": resp.text,
                },
            )

        return make_response(
            session_id=session_id,
            kind=resp.kind.value,
            text=resp.text,
            is_final=resp.is_final,
            reply_to=reply_to,
            extra=extra,
        )

    # ─────────────────────────────────────────────────────────────────────────────
    # Admin: Config management
    # ─────────────────────────────────────────────────────────────────────────────

    async def _handle_config_get(
        self, ws: ServerConnection, msg: GatewayMessage
    ) -> None:
        """Return the full config.yaml as a dict."""
        import yaml
        from pathlib import Path

        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        try:
            if config_path.exists():
                with config_path.open("r") as f:
                    config = yaml.safe_load(f) or {}
            else:
                config = {}

            resp = GatewayMessage(
                type=MessageType.RESPONSE.value,
                session_id=msg.session_id,
                data={"kind": "config", "config": config, "reply_to": msg.id},
            )
            await ws.send(resp.to_json())
        except Exception as e:
            err = make_error("config_error", str(e), reply_to=msg.id)
            await ws.send(err.to_json())

    async def _handle_config_set(
        self, ws: ServerConnection, msg: GatewayMessage
    ) -> None:
        """Update config.yaml with the provided key-value pairs."""
        import yaml
        from pathlib import Path

        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        updates = msg.data.get("config", {})
        if not updates:
            err = make_error("missing_field", "config.set requires 'config' dict",
                             reply_to=msg.id)
            await ws.send(err.to_json())
            return

        try:
            # Read existing
            if config_path.exists():
                with config_path.open("r") as f:
                    config = yaml.safe_load(f) or {}
            else:
                config = {}

            # Deep merge updates
            def deep_merge(base, new):
                for k, v in new.items():
                    if isinstance(v, dict) and isinstance(base.get(k), dict):
                        deep_merge(base[k], v)
                    else:
                        base[k] = v

            deep_merge(config, updates)

            # Write back
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with config_path.open("w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False,
                          allow_unicode=True)

            resp = GatewayMessage(
                type=MessageType.RESPONSE.value,
                session_id=msg.session_id,
                data={"kind": "config_saved", "config": config, "reply_to": msg.id},
            )
            await ws.send(resp.to_json())
            log.info("admin.config_updated", keys=list(updates.keys()))
        except Exception as e:
            err = make_error("config_error", str(e), reply_to=msg.id)
            await ws.send(err.to_json())

    # ─────────────────────────────────────────────────────────────────────────────
    # Admin: Environment variables
    # ─────────────────────────────────────────────────────────────────────────────

    async def _handle_env_list(
        self, ws: ServerConnection, msg: GatewayMessage
    ) -> None:
        """Return env vars from .env file (keys only, values masked)."""
        from pathlib import Path

        env_path = Path(__file__).parent.parent / ".env"
        env_vars: list[dict[str, str]] = []

        try:
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip()
                        # Mask values for security (show first/last 4 chars)
                        if len(value) > 10:
                            masked = value[:4] + "*" * (len(value) - 8) + value[-4:]
                        elif len(value) > 4:
                            masked = value[:2] + "*" * (len(value) - 2)
                        elif value in ("", "..."):
                            masked = "(not set)"
                        else:
                            masked = value
                        env_vars.append({"key": key, "value": value, "masked": masked})

            resp = GatewayMessage(
                type=MessageType.RESPONSE.value,
                session_id=msg.session_id,
                data={"kind": "env_list", "env_vars": env_vars, "reply_to": msg.id},
            )
            await ws.send(resp.to_json())
        except Exception as e:
            err = make_error("env_error", str(e), reply_to=msg.id)
            await ws.send(err.to_json())

    async def _handle_env_set(
        self, ws: ServerConnection, msg: GatewayMessage
    ) -> None:
        """Update an env var in .env file."""
        from pathlib import Path

        env_path = Path(__file__).parent.parent / ".env"
        key = msg.data.get("key", "").strip()
        value = msg.data.get("value", "").strip()

        if not key:
            err = make_error("missing_field", "env.set requires 'key'",
                             reply_to=msg.id)
            await ws.send(err.to_json())
            return

        try:
            lines = []
            found = False
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    stripped = line.strip()
                    if stripped and not stripped.startswith("#") and "=" in stripped:
                        line_key = stripped.split("=", 1)[0].strip()
                        if line_key == key:
                            lines.append(f"{key}={value}")
                            found = True
                            continue
                    lines.append(line)

            if not found:
                lines.append(f"{key}={value}")

            env_path.write_text("\n".join(lines) + "\n")

            # Also update os.environ for the running process
            import os
            os.environ[key] = value

            resp = GatewayMessage(
                type=MessageType.RESPONSE.value,
                session_id=msg.session_id,
                data={"kind": "env_saved", "key": key, "reply_to": msg.id},
            )
            await ws.send(resp.to_json())
            log.info("admin.env_updated", key=key)
        except Exception as e:
            err = make_error("env_error", str(e), reply_to=msg.id)
            await ws.send(err.to_json())

    # ─────────────────────────────────────────────────────────────────────────────
    # Admin: Skills reload
    # ─────────────────────────────────────────────────────────────────────────────

    async def _handle_skills_reload(
        self, ws: ServerConnection, msg: GatewayMessage
    ) -> None:
        """Hot-reload all skills and invalidate caches."""
        from pathlib import Path
        from neuralclaw.skills.loader import SkillLoader

        try:
            _base = Path(__file__).parent.parent
            registry = self._registry
            old_count = len(registry)
            registry._skills.clear()
            registry._manifests.clear()

            dirs = [_base / "skills" / "builtin", _base / "skills" / "plugins"]
            SkillLoader().load_all(dirs, strict=False, registry=registry)

            # ClawHub skills
            try:
                from neuralclaw.skills.clawhub.bridge_loader import ClawhubBridgeLoader
                clawhub_dir = _base / "data" / "clawhub" / "skills"
                if clawhub_dir.exists():
                    ClawhubBridgeLoader().load_all(
                        skills_dir=clawhub_dir,
                        registry=registry,
                        settings=self._settings if hasattr(self, '_settings') else None,
                    )
            except Exception:
                pass  # ClawHub is optional

            new_count = len(registry)

            # Invalidate orchestrator caches
            self._orchestrator._cached_tool_schemas = None
            self._orchestrator._cached_md_instructions = None

            resp = GatewayMessage(
                type=MessageType.RESPONSE.value,
                session_id=msg.session_id,
                data={
                    "kind": "skills_reloaded",
                    "old_count": old_count,
                    "new_count": new_count,
                    "reply_to": msg.id,
                },
            )
            await ws.send(resp.to_json())
            log.info("admin.skills_reloaded", old=old_count, new=new_count)
        except Exception as e:
            err = make_error("skills_error", str(e), reply_to=msg.id)
            await ws.send(err.to_json())

    # ─────────────────────────────────────────────────────────────────────────────
    # Admin: System info
    # ─────────────────────────────────────────────────────────────────────────────

    async def _handle_system_info(
        self, ws: ServerConnection, msg: GatewayMessage
    ) -> None:
        """Return system information."""
        import platform
        import sys

        skills_count = len(self._registry)
        sessions_count = self._sessions.count
        connections_count = len(self._connections)

        info = {
            "kind": "system_info",
            "version": "1.0.0",
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": platform.system(),
            "skills_count": skills_count,
            "sessions_count": sessions_count,
            "connections_count": connections_count,
            "reply_to": msg.id,
        }

        resp = GatewayMessage(
            type=MessageType.RESPONSE.value,
            session_id=msg.session_id,
            data=info,
        )
        await ws.send(resp.to_json())

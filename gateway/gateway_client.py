"""
gateway/gateway_client.py — Async Gateway Client

Thin async client that connects to the gateway WebSocket server.
Used by interfaces (gateway-cli, future web UI) to communicate
with the agent without importing any agent internals.

Usage:
    async with GatewayClient("ws://localhost:9090") as client:
        session_id = await client.create_session()
        async for resp in client.ask(session_id, "Hello"):
            print(resp.data["text"])
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Optional

import websockets

from gateway.protocol import GatewayMessage, MessageType

from observability.logger import get_logger

log = get_logger(__name__)


class GatewayClient:
    """
    Async WebSocket client for the NeuralClaw gateway.

    Async context manager — auto-connects on enter, disconnects on exit.
    """

    def __init__(
        self,
        url: str = "ws://127.0.0.1:9090",
        auth_token: Optional[str] = None,
    ):
        self._url = url
        self._auth_token = auth_token
        self._ws = None
        self._pending: dict[str, asyncio.Queue] = {}  # msg_id → queue

    async def __aenter__(self) -> "GatewayClient":
        await self.connect()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.disconnect()

    # ─────────────────────────────────────────────────────────────────────────
    # Connection lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Connect to the gateway server."""
        self._ws = await websockets.connect(self._url, max_size=2**20)

        # Send auth if required
        if self._auth_token:
            auth = GatewayMessage(
                type="auth",
                data={"token": self._auth_token},
            )
            await self._ws.send(auth.to_json())

        # Start the reader task
        self._reader_task = asyncio.create_task(self._reader_loop())
        log.info("gateway_client.connected", url=self._url)

    async def disconnect(self) -> None:
        """Disconnect from the gateway server."""
        if hasattr(self, "_reader_task"):
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
        if self._ws:
            await self._ws.close()
            self._ws = None
        log.info("gateway_client.disconnected")

    # ─────────────────────────────────────────────────────────────────────────
    # Reader loop — dispatches incoming messages to pending queues
    # ─────────────────────────────────────────────────────────────────────────

    async def _reader_loop(self) -> None:
        """Read messages from the server and dispatch to waiting callers."""
        try:
            async for raw in self._ws:
                msg = GatewayMessage.from_json(raw)

                # Route by reply_to if present
                reply_to = msg.data.get("reply_to")
                if reply_to and reply_to in self._pending:
                    await self._pending[reply_to].put(msg)
                    continue

                # Route by msg.id
                if msg.id in self._pending:
                    await self._pending[msg.id].put(msg)
                    continue

                # Unsolicited messages (e.g. confirm_request)
                # go to the session queue
                sid = msg.session_id or "__global__"
                if sid in self._pending:
                    await self._pending[sid].put(msg)

        except websockets.ConnectionClosed:
            log.warning("gateway_client.connection_lost")
        except asyncio.CancelledError:
            return

    # ─────────────────────────────────────────────────────────────────────────
    # Public API — request/response patterns
    # ─────────────────────────────────────────────────────────────────────────

    async def _send_and_wait(self, msg: GatewayMessage) -> GatewayMessage:
        """Send a message and wait for a single response with matching ID."""
        queue: asyncio.Queue = asyncio.Queue()
        self._pending[msg.id] = queue
        try:
            await self._ws.send(msg.to_json())
            resp = await asyncio.wait_for(queue.get(), timeout=30.0)
            return resp
        finally:
            self._pending.pop(msg.id, None)

    async def _send_and_stream(
        self, msg: GatewayMessage
    ) -> AsyncGenerator[GatewayMessage, None]:
        """Send a message and yield all responses until is_final=True."""
        queue: asyncio.Queue = asyncio.Queue()
        self._pending[msg.id] = queue
        try:
            await self._ws.send(msg.to_json())
            while True:
                try:
                    resp = await asyncio.wait_for(queue.get(), timeout=600.0)
                except asyncio.TimeoutError:
                    break
                yield resp
                # Stop streaming when we get a final response
                if resp.data.get("is_final", True):
                    break
        finally:
            self._pending.pop(msg.id, None)

    # ─────────────────────────────────────────────────────────────────────────
    # High-level API
    # ─────────────────────────────────────────────────────────────────────────

    async def ping(self) -> bool:
        """Send a ping and wait for pong. Returns True if healthy."""
        try:
            msg = GatewayMessage(type=MessageType.PING.value)
            resp = await self._send_and_wait(msg)
            return resp.type == MessageType.PONG.value
        except Exception:
            return False

    async def create_session(
        self,
        user_id: str = "gateway-cli",
        trust_level: str = "low",
    ) -> str:
        """Create a new session and return its ID."""
        msg = GatewayMessage(
            type=MessageType.SESSION_CREATE.value,
            data={"user_id": user_id, "trust_level": trust_level},
        )
        resp = await self._send_and_wait(msg)
        return resp.session_id or ""

    async def ask(
        self, session_id: str, message: str
    ) -> AsyncGenerator[GatewayMessage, None]:
        """Send a message and stream all responses."""
        msg = GatewayMessage(
            type=MessageType.ASK.value,
            session_id=session_id,
            data={"message": message},
        )
        async for resp in self._send_and_stream(msg):
            yield resp

    async def run(
        self, session_id: str, goal: str
    ) -> AsyncGenerator[GatewayMessage, None]:
        """Start autonomous mode and stream progress."""
        msg = GatewayMessage(
            type=MessageType.RUN.value,
            session_id=session_id,
            data={"goal": goal},
        )
        async for resp in self._send_and_stream(msg):
            yield resp

    async def cancel(self, session_id: str) -> GatewayMessage:
        """Cancel a running task."""
        msg = GatewayMessage(
            type=MessageType.CANCEL.value,
            session_id=session_id,
        )
        return await self._send_and_wait(msg)

    async def confirm(
        self, session_id: str, tool_call_id: str, approved: bool
    ) -> GatewayMessage:
        """Respond to a confirmation request."""
        msg = GatewayMessage(
            type=MessageType.CONFIRM.value,
            session_id=session_id,
            data={"tool_call_id": tool_call_id, "approved": approved},
        )
        return await self._send_and_wait(msg)

    async def set_trust(self, session_id: str, level: str) -> GatewayMessage:
        """Change the session trust level."""
        msg = GatewayMessage(
            type=MessageType.SESSION_TRUST.value,
            session_id=session_id,
            data={"level": level},
        )
        return await self._send_and_wait(msg)

    async def grant(self, session_id: str, capability: str) -> GatewayMessage:
        """Grant a capability to the session."""
        msg = GatewayMessage(
            type=MessageType.SESSION_GRANT.value,
            session_id=session_id,
            data={"capability": capability},
        )
        return await self._send_and_wait(msg)

    async def revoke(self, session_id: str, capability: str) -> GatewayMessage:
        """Revoke a capability from the session."""
        msg = GatewayMessage(
            type=MessageType.SESSION_REVOKE.value,
            session_id=session_id,
            data={"capability": capability},
        )
        return await self._send_and_wait(msg)

    async def status(self, session_id: str) -> GatewayMessage:
        """Get session status."""
        msg = GatewayMessage(
            type=MessageType.SESSION_STATUS.value,
            session_id=session_id,
        )
        return await self._send_and_wait(msg)

    async def list_skills(self, session_id: Optional[str] = None) -> GatewayMessage:
        """List available skills."""
        msg = GatewayMessage(
            type=MessageType.SKILLS_LIST.value,
            session_id=session_id,
        )
        return await self._send_and_wait(msg)

    def subscribe(self, session_id: str) -> asyncio.Queue:
        """
        Subscribe to unsolicited messages for a session (e.g. confirm_request).

        Returns a queue that receives GatewayMessage objects pushed by
        the server for this session.
        """
        queue: asyncio.Queue = asyncio.Queue()
        self._pending[session_id] = queue
        return queue

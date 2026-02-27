"""
tests/unit/test_gateway_protocol.py — Gateway Protocol Tests

Tests for the gateway message protocol, session store, and server/client
message routing.
"""

import asyncio
import json
import pytest

from gateway.protocol import (
    GatewayMessage,
    MessageType,
    make_response,
    make_confirm_request,
    make_error,
    make_pong,
    make_session_created,
    make_session_updated,
)
from gateway.session_store import SessionStore
from skills.types import TrustLevel


# ─────────────────────────────────────────────────────────────────────────────
# Protocol: Message serialization
# ─────────────────────────────────────────────────────────────────────────────

class TestGatewayMessage:
    def test_roundtrip_json(self):
        msg = GatewayMessage(
            type="ask",
            session_id="sess-1",
            data={"message": "hello"},
        )
        raw = msg.to_json()
        parsed = GatewayMessage.from_json(raw)
        assert parsed.type == "ask"
        assert parsed.session_id == "sess-1"
        assert parsed.data["message"] == "hello"

    def test_none_fields_omitted(self):
        msg = GatewayMessage(type="ping")
        raw = msg.to_json()
        d = json.loads(raw)
        assert "session_id" not in d
        assert "type" in d

    def test_from_json_missing_type(self):
        msg = GatewayMessage.from_json('{"id": "abc"}')
        assert msg.type == "error"  # default fallback

    def test_auto_generated_id(self):
        msg = GatewayMessage(type="ping")
        assert len(msg.id) == 8  # first 8 chars of uuid

    def test_data_default_empty(self):
        msg = GatewayMessage(type="ping")
        assert msg.data == {}


# ─────────────────────────────────────────────────────────────────────────────
# Protocol: Factory helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestFactories:
    def test_make_response(self):
        msg = make_response(
            session_id="s1",
            kind="text",
            text="Hello!",
            reply_to="req-1",
        )
        assert msg.type == MessageType.RESPONSE.value
        assert msg.session_id == "s1"
        assert msg.data["kind"] == "text"
        assert msg.data["text"] == "Hello!"
        assert msg.data["reply_to"] == "req-1"
        assert msg.data["is_final"] is True

    def test_make_response_streaming(self):
        msg = make_response("s1", "progress", "Thinking…", is_final=False)
        assert msg.data["is_final"] is False

    def test_make_confirm_request(self):
        msg = make_confirm_request(
            session_id="s1",
            tool_call_id="tc-1",
            skill_name="terminal_exec",
            risk_level="HIGH",
            reason="Shell command execution",
            arguments={"command": "ls"},
        )
        assert msg.type == MessageType.CONFIRM_REQUEST.value
        assert msg.data["tool_call_id"] == "tc-1"
        assert msg.data["skill_name"] == "terminal_exec"
        assert msg.data["risk_level"] == "HIGH"
        assert msg.data["arguments"]["command"] == "ls"

    def test_make_error(self):
        msg = make_error("auth_failed", "Bad token", reply_to="req-2")
        assert msg.type == MessageType.ERROR.value
        assert msg.data["code"] == "auth_failed"
        assert msg.data["reply_to"] == "req-2"

    def test_make_pong(self):
        msg = make_pong()
        assert msg.type == MessageType.PONG.value

    def test_make_session_created(self):
        msg = make_session_created("sess-abc")
        assert msg.type == MessageType.SESSION_CREATED.value
        assert msg.session_id == "sess-abc"

    def test_make_session_updated(self):
        msg = make_session_updated("sess-abc", trust_level="high")
        assert msg.type == MessageType.SESSION_UPDATED.value
        assert msg.data["trust_level"] == "high"


# ─────────────────────────────────────────────────────────────────────────────
# Protocol: MessageType enum
# ─────────────────────────────────────────────────────────────────────────────

class TestMessageType:
    def test_all_types_are_strings(self):
        for t in MessageType:
            assert isinstance(t.value, str)

    def test_client_types(self):
        client_types = {"ask", "run", "cancel", "confirm", "ping",
                        "session.create", "session.trust", "session.grant",
                        "session.revoke", "session.status", "skills.list"}
        for t in client_types:
            assert t in [m.value for m in MessageType]

    def test_server_types(self):
        server_types = {"response", "confirm_request", "session.created",
                        "session.updated", "error", "pong"}
        for t in server_types:
            assert t in [m.value for m in MessageType]


# ─────────────────────────────────────────────────────────────────────────────
# SessionStore
# ─────────────────────────────────────────────────────────────────────────────

class TestSessionStore:
    @pytest.mark.asyncio
    async def test_create_returns_session(self):
        store = SessionStore()
        session = await store.create(user_id="test-user")
        assert session.id is not None
        assert session.user_id == "test-user"

    @pytest.mark.asyncio
    async def test_get_returns_none_for_unknown(self):
        store = SessionStore()
        assert await store.get("nonexistent") is None

    @pytest.mark.asyncio
    async def test_get_returns_existing(self):
        store = SessionStore()
        session = await store.create()
        fetched = await store.get(session.id)
        assert fetched is session

    @pytest.mark.asyncio
    async def test_get_or_create_creates_new(self):
        store = SessionStore()
        session = await store.get_or_create()
        assert session.id is not None
        assert store.count == 1

    @pytest.mark.asyncio
    async def test_get_or_create_returns_existing(self):
        store = SessionStore()
        session = await store.create()
        fetched = await store.get_or_create(session_id=session.id)
        assert fetched is session
        assert store.count == 1

    @pytest.mark.asyncio
    async def test_remove_returns_true_for_existing(self):
        store = SessionStore()
        session = await store.create()
        assert await store.remove(session.id) is True
        assert store.count == 0

    @pytest.mark.asyncio
    async def test_remove_returns_false_for_unknown(self):
        store = SessionStore()
        assert await store.remove("nope") is False

    @pytest.mark.asyncio
    async def test_list_sessions(self):
        store = SessionStore()
        s1 = await store.create()
        s2 = await store.create()
        ids = await store.list_sessions()
        assert s1.id in ids
        assert s2.id in ids

    @pytest.mark.asyncio
    async def test_default_trust_level(self):
        store = SessionStore(default_trust=TrustLevel.MEDIUM)
        session = await store.create()
        assert session.trust_level == TrustLevel.MEDIUM

    @pytest.mark.asyncio
    async def test_override_trust_level(self):
        store = SessionStore(default_trust=TrustLevel.LOW)
        session = await store.create(trust_level=TrustLevel.HIGH)
        assert session.trust_level == TrustLevel.HIGH


# ─────────────────────────────────────────────────────────────────────────────
# Settings integration
# ─────────────────────────────────────────────────────────────────────────────

class TestGatewayConfig:
    def test_default_settings(self):
        from config.settings import GatewayConfig
        cfg = GatewayConfig()
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 9090
        assert cfg.auth_token is None
        assert cfg.max_connections == 10

    def test_invalid_port(self):
        from config.settings import GatewayConfig
        with pytest.raises(ValueError, match="gateway.port"):
            GatewayConfig(port=99999)

    def test_custom_values(self):
        from config.settings import GatewayConfig
        cfg = GatewayConfig(host="0.0.0.0", port=8080, auth_token="secret")
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 8080
        assert cfg.auth_token == "secret"

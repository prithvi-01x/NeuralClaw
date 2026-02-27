"""
gateway/protocol.py — Gateway WebSocket Message Protocol

Typed message schema for all client↔server communication.
Every message is JSON with a `type` field and a unique `id`.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional

import json


# ─────────────────────────────────────────────────────────────────────────────
# Message types
# ─────────────────────────────────────────────────────────────────────────────

class MessageType(str, Enum):
    """All supported message types in the gateway protocol."""

    # Client → Server
    ASK              = "ask"
    RUN              = "run"
    CANCEL           = "cancel"
    CONFIRM          = "confirm"
    SESSION_CREATE   = "session.create"
    SESSION_TRUST    = "session.trust"
    SESSION_GRANT    = "session.grant"
    SESSION_REVOKE   = "session.revoke"
    SESSION_STATUS   = "session.status"
    SKILLS_LIST      = "skills.list"
    SKILLS_RELOAD    = "skills.reload"
    CONFIG_GET       = "config.get"
    CONFIG_SET       = "config.set"
    ENV_LIST         = "env.list"
    ENV_SET          = "env.set"
    SYSTEM_INFO      = "system.info"
    PING             = "ping"

    # Server → Client
    RESPONSE         = "response"
    CONFIRM_REQUEST  = "confirm_request"
    SESSION_CREATED  = "session.created"
    SESSION_UPDATED  = "session.updated"
    ERROR            = "error"
    PONG             = "pong"


# ─────────────────────────────────────────────────────────────────────────────
# Base message
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GatewayMessage:
    """
    Universal message envelope for the gateway protocol.

    All fields are optional except `type`. Extra payload goes in `data`.
    """
    type: str
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    session_id: Optional[str] = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to JSON string, dropping None fields."""
        d = {k: v for k, v in asdict(self).items() if v is not None}
        return json.dumps(d)

    @classmethod
    def from_json(cls, raw: str) -> "GatewayMessage":
        """Parse a JSON string into a GatewayMessage."""
        d = json.loads(raw)
        return cls(
            type=d.get("type", "error"),
            id=d.get("id", str(uuid.uuid4())[:8]),
            session_id=d.get("session_id"),
            data=d.get("data", {}),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Factory helpers — Server → Client messages
# ─────────────────────────────────────────────────────────────────────────────

def make_response(
    session_id: str,
    kind: str,
    text: str,
    *,
    is_final: bool = True,
    reply_to: Optional[str] = None,
    extra: Optional[dict] = None,
) -> GatewayMessage:
    """Build a RESPONSE message to send back to the client."""
    data: dict[str, Any] = {
        "kind": kind,
        "text": text,
        "is_final": is_final,
    }
    if reply_to:
        data["reply_to"] = reply_to
    if extra:
        data.update(extra)
    return GatewayMessage(
        type=MessageType.RESPONSE.value,
        session_id=session_id,
        data=data,
    )


def make_confirm_request(
    session_id: str,
    tool_call_id: str,
    skill_name: str,
    risk_level: str,
    reason: str,
    arguments: dict,
) -> GatewayMessage:
    """Build a CONFIRM_REQUEST message."""
    return GatewayMessage(
        type=MessageType.CONFIRM_REQUEST.value,
        session_id=session_id,
        data={
            "tool_call_id": tool_call_id,
            "skill_name": skill_name,
            "risk_level": risk_level,
            "reason": reason,
            "arguments": arguments,
        },
    )


def make_error(
    code: str,
    message: str,
    *,
    reply_to: Optional[str] = None,
    session_id: Optional[str] = None,
) -> GatewayMessage:
    """Build an ERROR message."""
    data: dict[str, Any] = {"code": code, "message": message}
    if reply_to:
        data["reply_to"] = reply_to
    return GatewayMessage(
        type=MessageType.ERROR.value,
        session_id=session_id,
        data=data,
    )


def make_session_created(session_id: str) -> GatewayMessage:
    """Build a SESSION_CREATED acknowledgement."""
    return GatewayMessage(
        type=MessageType.SESSION_CREATED.value,
        session_id=session_id,
    )


def make_session_updated(session_id: str, **changes) -> GatewayMessage:
    """Build a SESSION_UPDATED acknowledgement with changed fields."""
    return GatewayMessage(
        type=MessageType.SESSION_UPDATED.value,
        session_id=session_id,
        data=changes,
    )


def make_pong() -> GatewayMessage:
    """Build a PONG keepalive response."""
    return GatewayMessage(type=MessageType.PONG.value)

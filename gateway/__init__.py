"""
gateway/ — WebSocket Gateway Control Plane

Provides a WebSocket server that acts as a control plane between interfaces
(CLI, Telegram, web, mobile) and the NeuralClaw agent orchestrator.

Interfaces become thin rendering shells that send/receive JSON messages
over a WebSocket connection.
"""

from gateway.protocol import MessageType, GatewayMessage
from gateway.session_store import SessionStore
from gateway.gateway_server import GatewayServer
from gateway.gateway_client import GatewayClient

__all__ = [
    "MessageType",
    "GatewayMessage",
    "SessionStore",
    "GatewayServer",
    "GatewayClient",
]

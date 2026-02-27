"""
gateway/ — WebSocket Gateway Control Plane

Provides a WebSocket server that acts as a control plane between interfaces
(CLI, Telegram, web, mobile) and the NeuralClaw agent orchestrator.

Interfaces become thin rendering shells that send/receive JSON messages
over a WebSocket connection.
"""

from neuralclaw.gateway.protocol import MessageType, GatewayMessage
from neuralclaw.gateway.session_store import SessionStore
from neuralclaw.gateway.gateway_server import GatewayServer
from neuralclaw.gateway.gateway_client import GatewayClient

__all__ = [
    "MessageType",
    "GatewayMessage",
    "SessionStore",
    "GatewayServer",
    "GatewayClient",
]

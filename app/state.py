"""
app/state.py — NeuralClaw Thread-Safe Application State (Phase H)

AgentAppState is a single source of truth for the current runtime status of
the NeuralClaw application. It is written by the asyncio voice pipeline and
read by the Qt GUI thread.

All mutations go through set_*() methods that acquire a threading.Lock,
so reads from the Qt thread never see a torn state.

Fields
------
    agent_status    : str       — human-readable status label
    overlay_state   : str       — one of the 6 overlay states (see OverlayState)
    last_command    : str       — last transcribed user utterance
    last_response   : str       — last agent response (truncated to 200 chars)
    active_task     : str|None  — currently running scheduler task name
    wake_confidence : float     — confidence of last wake detection (0–1)
    error_text      : str       — last error message (cleared on next turn)
    is_running      : bool      — True while the voice daemon is active
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional


class OverlayState:
    """Constants for the 6 overlay states defined in the Phase H spec."""
    IDLE         = "idle"           # small pulsing ring, 10% opacity
    LISTENING    = "listening"      # full waveform, cyan glow, mic icon
    THINKING     = "thinking"       # spinning arc, blue
    SPEAKING     = "speaking"       # waveform synced to TTS
    ERROR        = "error"          # red pulse + brief error text
    TASK_RUNNING = "task_running"   # small green progress indicator


@dataclass
class AgentAppState:
    """
    Thread-safe application state container.

    Usage::

        state = AgentAppState()

        # asyncio thread
        state.set_overlay("listening")
        state.set_last_command("scan for open ports")

        # Qt thread (reads are safe without lock for display purposes)
        label.setText(state.agent_status)
    """

    # ── Mutable state fields ──────────────────────────────────────────────────
    agent_status:    str   = "idle"
    overlay_state:   str   = OverlayState.IDLE
    last_command:    str   = ""
    last_response:   str   = ""
    active_task:     Optional[str] = None
    wake_confidence: float = 0.0
    error_text:      str   = ""
    is_running:      bool  = False

    # ── Internal lock ─────────────────────────────────────────────────────────
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    # ── Setters (thread-safe) ─────────────────────────────────────────────────

    def set_overlay(self, state: str) -> None:
        """Set overlay_state. state must be one of OverlayState constants."""
        with self._lock:
            self.overlay_state = state

    def set_status(self, status: str) -> None:
        with self._lock:
            self.agent_status = status

    def set_last_command(self, text: str) -> None:
        with self._lock:
            self.last_command = text[:500]

    def set_last_response(self, text: str) -> None:
        with self._lock:
            self.last_response = text[:200]

    def set_active_task(self, name: Optional[str]) -> None:
        with self._lock:
            self.active_task = name

    def set_wake_confidence(self, confidence: float) -> None:
        with self._lock:
            self.wake_confidence = confidence

    def set_error(self, text: str) -> None:
        with self._lock:
            self.error_text = text[:300]

    def set_running(self, running: bool) -> None:
        with self._lock:
            self.is_running = running

    def snapshot(self) -> dict:
        """Return a point-in-time copy of all fields as a plain dict."""
        with self._lock:
            return {
                "agent_status":    self.agent_status,
                "overlay_state":   self.overlay_state,
                "last_command":    self.last_command,
                "last_response":   self.last_response,
                "active_task":     self.active_task,
                "wake_confidence": self.wake_confidence,
                "error_text":      self.error_text,
                "is_running":      self.is_running,
            }


# ── Module-level singleton ────────────────────────────────────────────────────

_state: AgentAppState | None = None


def get_state() -> AgentAppState:
    """Return the application-wide state singleton."""
    global _state
    if _state is None:
        _state = AgentAppState()
    return _state
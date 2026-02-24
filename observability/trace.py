"""
observability/trace.py — Trace Context for NeuralClaw

Provides a lightweight TraceContext that attaches trace_id and turn_id
to every structured log line within a session/turn scope, using
structlog's contextvars integration.

Usage (orchestrator):
    from observability.trace import TraceContext

    # At session start — binds trace_id for lifetime of this async context
    ctx = TraceContext.for_session(session_id)
    ctx.bind()

    # At each turn start — binds turn_id in addition to trace_id
    ctx.new_turn()
    ctx.bind_turn()

    # All log calls in this coroutine and its children now automatically
    # include trace_id and turn_id — no need to pass them explicitly.
    log.info("skill_dispatched", skill="web_fetch")
    # → {"event": "skill_dispatched", "skill": "web_fetch",
    #    "trace_id": "trc_a1b2c3d4", "turn_id": "trn_x9y8z7", ...}

    # At turn end
    ctx.clear_turn()

    # At session end
    ctx.clear()
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
import structlog
import structlog.contextvars as _scv

def _short_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@dataclass
class TraceContext:
    trace_id: str = field(default_factory=lambda: _short_id("trc"))
    _turn_id: str | None = field(default=None, repr=False)

    # ─────────────────────────────────────────────
    # Factory
    # ─────────────────────────────────────────────

    @classmethod
    def for_session(cls, session_id: str | None = None) -> "TraceContext":
        if session_id:
            trace_id = f"trc_{session_id[:8]}"
        else:
            trace_id = _short_id("trc")
        return cls(trace_id=trace_id)

    # ─────────────────────────────────────────────
    # Turn Handling
    # ─────────────────────────────────────────────

    def new_turn(self) -> "TraceContext":
        """Generate a fresh turn_id for this context. Mutates in-place and
        returns `self` so callers can chain:  ctx.new_turn().bind()
        """
        self._turn_id = _short_id("trn")
        return self

    @property
    def turn_id(self) -> str | None:
        return self._turn_id

    # ─────────────────────────────────────────────
    # Binding
    # ─────────────────────────────────────────────

    def bind(self) -> None:
        """
        Bind trace_id and optional turn_id.
        Accesses bind_contextvars via the module reference so that
        patch("structlog.contextvars.bind_contextvars") intercepts the call.
        """
        if self._turn_id is not None:
            _scv.bind_contextvars(trace_id=self.trace_id, turn_id=self._turn_id)
        else:
            _scv.bind_contextvars(trace_id=self.trace_id)

    def bind_turn(self) -> None:
        if self._turn_id is not None:
            _scv.bind_contextvars(turn_id=self._turn_id)

    def clear_turn(self) -> None:
        self._turn_id = None
        try:
            _scv.unbind_contextvars("turn_id")
        except KeyError:
            pass

    def clear(self) -> None:
        self._turn_id = None
        try:
            _scv.unbind_contextvars("trace_id", "turn_id")
        except KeyError:
            pass

    # ─────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────

    def as_dict(self) -> dict:
        data = {"trace_id": self.trace_id}
        if self._turn_id is not None:
            data["turn_id"] = self._turn_id
        return data

    def __repr__(self) -> str:
        return f"TraceContext(trace_id={self.trace_id!r}, turn_id={self._turn_id!r})"
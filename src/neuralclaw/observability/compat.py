"""
observability/compat.py — Portable Logger Fallback

Provides a safe logger factory that works whether or not structlog is
installed.  Modules that live at the edge of the dependency tree
(task memory, ClawHub bridge, ClawHub installer) import from here
instead of duplicating the try/except block.

Usage::

    from neuralclaw.observability.compat import get_safe_logger, safe_log
    _log_raw = get_safe_logger(__name__)
    safe_log(_log_raw, "info", "my_event", key="value")
"""

from __future__ import annotations

import logging
from typing import Any


def get_safe_logger(name: str) -> Any:
    """Return a structlog logger if available, otherwise a stdlib logger."""
    try:
        from neuralclaw.observability.logger import get_logger
        return get_logger(name)
    except ImportError:
        return logging.getLogger(name)


_STRUCTLOG: bool | None = None


def _has_structlog() -> bool:
    global _STRUCTLOG
    if _STRUCTLOG is None:
        try:
            from neuralclaw.observability.logger import get_logger  # noqa: F401
            _STRUCTLOG = True
        except ImportError:
            _STRUCTLOG = False
    return _STRUCTLOG


def safe_log(logger: Any, level: str, event: str, **kwargs: Any) -> None:
    """
    Emit a structured log message that works with both structlog and stdlib.

    With structlog:   logger.info("event", key=val)
    With stdlib:      logger.info("event key=val")
    """
    if _has_structlog():
        getattr(logger, level)(event, **kwargs)
    else:
        extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
        msg = f"{event} {extra}" if extra else event
        getattr(logger, level)(msg)

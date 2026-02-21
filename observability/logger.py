"""
observability/logger.py — NeuralClaw Structured Logger

Sets up structlog with:
  - JSON output to rotating log files
  - Human-readable output to console (dev mode) or JSON (prod mode)
  - Consistent fields on every log line: timestamp, level, event, session_id

Usage:
    from observability.logger import get_logger, setup_logging
    from config.settings import get_settings

    setup_logging(get_settings())          # call once at startup
    log = get_logger(__name__)
    log.info("tool.call.start", tool="browser_navigate", url="https://example.com")
    log.warning("safety.blocked", tool="terminal_exec", reason="rm pattern matched")
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any, Optional

import structlog

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def setup_logging(
    level: str = "INFO",
    log_dir: str | Path = "./data/logs",
    json_format: bool = True,
    console_output: bool = True,
    max_bytes: int = 100 * 1024 * 1024,   # 100 MB
    backup_count: int = 5,
) -> None:
    """
    Configure structlog and stdlib logging. Call once at application startup.

    Args:
        level:          Log level string — DEBUG | INFO | WARNING | ERROR | CRITICAL
        log_dir:        Directory for rotating log files.
        json_format:    If True, console also emits JSON (production mode).
                        If False, console uses coloured human-readable format (dev mode).
        console_output: Whether to emit logs to stdout at all.
        max_bytes:      Max size of each log file before rotation.
        backup_count:   Number of rotated log files to keep.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # ── Shared structlog processors ───────────────────────────────────────────
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    # ── File handler (always JSON) ────────────────────────────────────────────
    handlers: list[logging.Handler] = []

    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_dir / "neuralclaw.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(numeric_level)
    handlers.append(file_handler)

    # ── Console handler (JSON or pretty) ─────────────────────────────────────
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        handlers.append(console_handler)

    # ── Configure stdlib logging (structlog routes through it) ────────────────
    logging.basicConfig(
        format="%(message)s",
        level=numeric_level,
        handlers=handlers,
        force=True,
    )

    # ── Configure structlog ───────────────────────────────────────────────────
    if json_format:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Attach renderer to the stdlib formatter used by all handlers
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=shared_processors,
    )

    for handler in handlers:
        handler.setFormatter(formatter)


def get_logger(name: str = "neuralclaw", **initial_values: Any) -> structlog.stdlib.BoundLogger:
    """
    Get a bound logger with optional initial context values.

    Args:
        name:           Logger name, typically __name__ of the calling module.
        **initial_values: Key-value pairs permanently bound to this logger instance.

    Returns:
        A structlog BoundLogger.

    Example:
        log = get_logger(__name__, component="tool_bus")
        log.info("tool.dispatched", tool="browser_navigate")
        # → {"event": "tool.dispatched", "tool": "browser_navigate",
        #    "component": "tool_bus", "logger": "tools.tool_bus", ...}
    """
    logger = structlog.get_logger(name)
    if initial_values:
        logger = logger.bind(**initial_values)
    return logger


def bind_session(session_id: str, user_id: str) -> None:
    """
    Bind session context to all subsequent log calls in this async context.

    Call this at the start of handling a user request. structlog's contextvars
    integration ensures the values are attached to every log line in this
    coroutine and its children, without passing them explicitly.

    Example:
        bind_session(session.id, session.user_id)
        log.info("agent.loop.start")
        # → includes session_id and user_id automatically
    """
    structlog.contextvars.bind_contextvars(session_id=session_id, user_id=user_id)


def clear_session() -> None:
    """Clear session context vars at the end of a request."""
    structlog.contextvars.clear_contextvars()


# ─────────────────────────────────────────────────────────────────────────────
# Module-level logger (for internal use within this module)
# ─────────────────────────────────────────────────────────────────────────────

_log = get_logger(__name__)
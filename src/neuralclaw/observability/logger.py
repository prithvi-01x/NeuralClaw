"""
observability/logger.py — NeuralClaw Structured Logger

Sets up structlog with:
  - JSON output to rotating log files only (stdout stays clean for the chat UI)
  - Optional human-readable console output (dev mode) or JSON (prod/pipe mode)
  - Consistent fields on every log line: timestamp, level, event, session_id
  - Third-party noisy loggers fully muted via NullHandler so they never
    reach stdout regardless of log level (chromadb telemetry, sentence_transformers,
    posthog, huggingface_hub, torch).

Usage:
    from neuralclaw.observability.logger import get_logger, setup_logging
    setup_logging(level="INFO", log_dir="./data/logs", console_output=False)
    log = get_logger(__name__)
    log.info("tool.call.start", tool="browser_navigate", url="https://example.com")
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any, Optional

import structlog

# ─────────────────────────────────────────────────────────────────────────────
# Third-party loggers that produce noise on stdout at INFO level.
# These are silenced globally to WARNING so they only appear in the log file
# and only when something actually goes wrong.
# ─────────────────────────────────────────────────────────────────────────────
# Third-party loggers that must never reach stdout.
# We attach a NullHandler AND set the level to CRITICAL so that even
# ERROR-level messages (like chromadb posthog telemetry failures) are
# completely swallowed — they still go to the log file via the root handler
# if the file handler is configured, but never pollute the terminal.
_MUTED_LOGGERS = [
    "sentence_transformers",
    "sentence_transformers.SentenceTransformer",
    "chromadb",
    "chromadb.telemetry",
    "chromadb.telemetry.product.posthog",
    "huggingface_hub",
    "transformers",
    "torch",
]


def _mute_noisy_loggers() -> None:
    """
    Fully suppress all known chatty third-party loggers.

    Sets propagate=False so their records never reach the root logger
    (and therefore never reach any stdout handler), then attaches a
    NullHandler so Python does not emit the 'No handlers could be found'
    warning. Level is set to CRITICAL as an extra belt-and-braces guard.
    """
    null = logging.NullHandler()
    for name in _MUTED_LOGGERS:
        lgr = logging.getLogger(name)
        lgr.setLevel(logging.CRITICAL)
        lgr.propagate = False
        # Avoid adding duplicate NullHandlers on repeated setup_logging calls
        if not any(isinstance(h, logging.NullHandler) for h in lgr.handlers):
            lgr.addHandler(null)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def setup_logging(
    level: str = "INFO",
    log_dir: str | Path = "./data/logs",
    json_format: Optional[bool] = None,  # None = auto-detect from tty
    console_output: bool = True,
    max_bytes: int = 100 * 1024 * 1024,   # 100 MB
    backup_count: int = 5,
) -> None:
    """
    Configure structlog and stdlib logging. Call once at application startup.

    Args:
        level:          Log level string — DEBUG | INFO | WARNING | ERROR | CRITICAL
        log_dir:        Directory for rotating log files.
        json_format:    If True, console emits JSON (production/pipe mode).
                        If False, console uses coloured human-readable format (dev mode).
                        If None (default), auto-detects: pretty when stdout is a TTY,
                        JSON when stdout is a pipe/file (e.g. systemd, Docker).
        console_output: Whether to emit logs to stdout at all.
        max_bytes:      Max size of each log file before rotation.
        backup_count:   Number of rotated log files to keep.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Auto-detect JSON vs pretty based on whether stdout is a real terminal
    if json_format is None:
        json_format = not sys.stdout.isatty()

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

    # ── Mute noisy third-party loggers AFTER basicConfig ───────────────────────
    # propagate=False + NullHandler means their records never reach the root
    # logger and therefore never reach any stdout handler, regardless of level.
    # Must be called after basicConfig so the root logger is already set up.
    _mute_noisy_loggers()

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

    # File always uses JSON regardless of console format
    file_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
        foreign_pre_chain=shared_processors,
    )

    for i, handler in enumerate(handlers):
        if isinstance(handler, logging.handlers.RotatingFileHandler):
            handler.setFormatter(file_formatter)
        else:
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


# ─────────────────────────────────────────────────────────────────────────────
# Portable logger shim — works with or without structlog
# ─────────────────────────────────────────────────────────────────────────────

def portable_log(name: str):
    """
    Return a callable ``log(level, event, **kwargs)`` that works regardless
    of whether structlog is available.

    Usage (replaces the copy-pasted try/except + _log shim in skills/)::

        from neuralclaw.observability.logger import portable_log
        _log = portable_log(__name__)
        _log("info", "skill_bus.dispatching", skill="terminal_exec")
    """
    try:
        bound = get_logger(name)
        _is_structlog = True
    except Exception:
        import logging as _fallback_logging
        bound = _fallback_logging.getLogger(name)
        _is_structlog = False

    def _emit(level: str, event: str, **kwargs) -> None:
        if _is_structlog:
            getattr(bound, level)(event, **kwargs)
        else:
            extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
            getattr(bound, level)("%s %s", event, extra)

    return _emit
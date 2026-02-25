"""
app/__init__.py — NeuralClaw Qt Application Entry Point (Phase H)

Bootstraps the Qt application alongside the asyncio voice pipeline.

Threading model
--------------
    Main thread  : Qt event loop (QApplication.exec())
    Worker thread: asyncio event loop running the voice pipeline

Cross-thread communication:
    asyncio → Qt  : Qt signals (NeuralClawSignals) — always thread-safe
    Qt → asyncio  : asyncio.run_coroutine_threadsafe(coro, loop)

NEVER call Qt widget methods from the asyncio thread.
NEVER call await from a Qt slot.

Usage::

    from app import run_qt_app
    run_qt_app(settings)        # blocks until Quit
"""

from __future__ import annotations

import asyncio
import threading
from typing import Optional


def run_qt_app(settings=None) -> int:
    """
    Start the Qt application and asyncio voice pipeline concurrently.

    Parameters
    ----------
    settings : Settings | None
        NeuralClaw settings. If None, loads from config.yaml.

    Returns
    -------
    int
        Exit code from QApplication.exec().
    """
    import sys
    from PyQt6.QtWidgets import QApplication

    # ── Qt application ────────────────────────────────────────────────────────
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("NeuralClaw")
    app.setQuitOnLastWindowClosed(False)   # tray keeps the app alive

    # ── Import widgets here (after QApplication exists) ───────────────────────
    from app.overlay import OverlayWidget
    from app.tray import NeuralClawTray
    from app.state import get_state

    overlay = OverlayWidget()
    overlay.show()

    tray = NeuralClawTray(overlay=overlay)
    tray.show()

    get_state().set_running(True)

    # ── asyncio voice pipeline in background thread ───────────────────────────
    _loop: Optional[asyncio.AbstractEventLoop] = None
    _thread: Optional[threading.Thread] = None

    def _run_asyncio() -> None:
        nonlocal _loop
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
        try:
            if settings is not None:
                _loop.run_until_complete(_voice_pipeline(settings))
        except Exception as exc:
            from app.signals import get_signals
            get_signals().error_occurred.emit(str(exc)[:80])
        finally:
            _loop.close()

    if settings is not None:
        _thread = threading.Thread(target=_run_asyncio, daemon=True, name="asyncio-voice")
        _thread.start()

    exit_code = app.exec()

    # ── Shutdown ──────────────────────────────────────────────────────────────
    get_state().set_running(False)
    if _loop and not _loop.is_closed():
        _loop.call_soon_threadsafe(_loop.stop)
    if _thread:
        _thread.join(timeout=3.0)

    return exit_code


async def _voice_pipeline(settings) -> None:
    """
    Asyncio coroutine: wire signals from voice pipeline into Qt.
    Runs in the background thread's event loop.
    """
    from app.signals import get_signals
    from app.state import get_state

    sigs = get_signals()

    try:
        from interfaces.voice import VoiceInterface

        vi = VoiceInterface(settings)

        def _on_wake(wake_event):
            get_state().set_wake_confidence(wake_event.confidence)
            sigs.wake_detected.emit(wake_event.model_name, wake_event.confidence)

        vi._on_wake_detected = _on_wake

        sigs.agent_status_changed.emit("voice pipeline starting…")
        await vi.start()

    except Exception as exc:
        sigs.error_occurred.emit(f"Voice pipeline error: {exc!s:.60}")
        sigs.agent_status_changed.emit("error — check logs")
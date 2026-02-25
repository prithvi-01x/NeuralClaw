"""
app/signals.py — NeuralClaw Qt Signal Bridge (Phase H)

Defines all Qt signals used for cross-thread communication between the
asyncio voice pipeline and the Qt GUI thread. This is the ONLY safe way
to update Qt widgets from asyncio coroutines running in a different thread.

Architecture
------------
    asyncio thread (voice pipeline)
        └── calls emit_*(…) helpers  → puts event on Qt signal queue
    Qt main thread
        └── signal handler updates overlay / tray state

Rules (enforced by Qt):
    - NEVER call QWidget methods directly from asyncio thread
    - NEVER call await from a Qt slot
    - ALL cross-thread updates go through these signals

Signal inventory
----------------
    wake_detected       — hotword fired; carry WakeEvent data
    turn_started        — orchestrator beginning a turn (text = transcription)
    turn_completed      — orchestrator returned (text = response, status = ok/error/blocked)
    tts_started         — TTS playback beginning (text = utterance)
    tts_stopped         — TTS playback finished
    task_started        — scheduler task launched (name, task_id)
    task_completed      — scheduler task done (name, task_id, status)
    agent_status_changed— general status string changed
    error_occurred      — error text for brief display
"""

from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal


class NeuralClawSignals(QObject):
    """
    Singleton QObject that owns all application-wide Qt signals.

    Usage::

        # In asyncio thread:
        signals = get_signals()
        signals.wake_detected.emit("hey_mycroft", 0.91)

        # In Qt slot (main thread):
        signals.wake_detected.connect(overlay.on_wake)

    Never instantiate directly — use get_signals().
    """

    # Hotword / wake
    wake_detected = pyqtSignal(str, float)          # model_name, confidence

    # Voice turn lifecycle
    turn_started = pyqtSignal(str)                  # transcribed_text
    turn_completed = pyqtSignal(str, str)           # response_text, status ("ok"|"error"|"blocked")

    # TTS
    tts_started = pyqtSignal(str)                   # utterance_text
    tts_stopped = pyqtSignal()

    # Scheduler tasks
    task_started = pyqtSignal(str, str)             # task_name, task_id
    task_completed = pyqtSignal(str, str, str)      # task_name, task_id, status

    # General
    agent_status_changed = pyqtSignal(str)          # status_label
    error_occurred = pyqtSignal(str)                # error_text


# ── Module-level singleton ────────────────────────────────────────────────────

_signals: NeuralClawSignals | None = None


def get_signals() -> NeuralClawSignals:
    """Return the application-wide signal singleton. Created on first call."""
    global _signals
    if _signals is None:
        _signals = NeuralClawSignals()
    return _signals
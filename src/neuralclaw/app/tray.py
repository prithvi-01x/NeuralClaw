"""
app/tray.py — NeuralClaw System Tray Icon (Phase H)

Provides a QSystemTrayIcon with a right-click context menu:
    • Status indicator (current agent state)
    • Open Dashboard  (placeholder in Phase H, fully implemented in Phase I)
    • Settings        (placeholder)
    • ─────────────
    • Quit

The tray icon uses programmatically-drawn icons so no PNG assets are
required. The icon colour reflects the current agent state:
    grey  → idle
    cyan  → listening
    blue  → thinking
    green → speaking / task running
    red   → error

Architecture
-----------
    NeuralClawTray receives Qt signals from the asyncio pipeline and updates
    its icon and tooltip accordingly. All Qt operations happen in the Qt
    main thread — the signals are cross-thread safe.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, QSize, pyqtSlot
from PyQt6.QtGui import (
    QIcon, QPixmap, QPainter, QColor, QBrush, QPen,
    QRadialGradient, QFont, QAction
)
from PyQt6.QtWidgets import QSystemTrayIcon, QMenu, QApplication, QMessageBox

from neuralclaw.app.state import OverlayState, get_state
from neuralclaw.app.signals import get_signals

# ── Icon parameters ───────────────────────────────────────────────────────────

_ICON_SIZE = 22    # px — standard tray icon size on most Linux DEs

_STATE_COLOURS: dict[str, QColor] = {
    OverlayState.IDLE:         QColor(80, 80, 100),
    OverlayState.LISTENING:    QColor(0, 220, 210),
    OverlayState.THINKING:     QColor(60, 120, 255),
    OverlayState.SPEAKING:     QColor(120, 220, 120),
    OverlayState.ERROR:        QColor(255, 60, 60),
    OverlayState.TASK_RUNNING: QColor(60, 200, 100),
}

_TOOLTIPS: dict[str, str] = {
    OverlayState.IDLE:         "NeuralClaw — idle",
    OverlayState.LISTENING:    "NeuralClaw — listening…",
    OverlayState.THINKING:     "NeuralClaw — thinking…",
    OverlayState.SPEAKING:     "NeuralClaw — speaking",
    OverlayState.ERROR:        "NeuralClaw — error",
    OverlayState.TASK_RUNNING: "NeuralClaw — task running",
}


def _make_icon(state: str) -> QIcon:
    """Render a coloured dot icon for the given state."""
    colour = _STATE_COLOURS.get(state, _STATE_COLOURS[OverlayState.IDLE])
    px = QPixmap(_ICON_SIZE, _ICON_SIZE)
    px.fill(Qt.GlobalColor.transparent)

    p = QPainter(px)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    cx = cy = _ICON_SIZE / 2
    r = _ICON_SIZE / 2 - 2

    # Radial gradient for a glowing dot effect
    grad = QRadialGradient(cx, cy, r)
    bright = QColor(colour)
    bright.setAlpha(255)
    dim = QColor(colour)
    dim.setAlpha(80)
    outer = QColor(colour)
    outer.setAlpha(0)
    grad.setColorAt(0.0, bright)
    grad.setColorAt(0.5, dim)
    grad.setColorAt(1.0, outer)

    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(grad))
    p.drawEllipse(2, 2, _ICON_SIZE - 4, _ICON_SIZE - 4)

    # Crisp centre dot
    p.setBrush(QBrush(bright))
    dot_r = int(r * 0.4)
    p.drawEllipse(
        int(cx - dot_r), int(cy - dot_r),
        dot_r * 2, dot_r * 2,
    )
    p.end()

    return QIcon(px)


class NeuralClawTray(QSystemTrayIcon):
    """
    System tray icon for NeuralClaw.

    Instantiate after QApplication.exec() loop is running::

        tray = NeuralClawTray(overlay=overlay_widget)
        tray.show()

    Parameters
    ----------
    overlay  : OverlayWidget | None  — passed to the "Toggle overlay" menu item
    """

    def __init__(self, overlay=None, parent: Optional[QApplication] = None) -> None:
        super().__init__(parent)

        self._overlay = overlay
        self._current_state = OverlayState.IDLE

        # ── Initial icon + tooltip ────────────────────────────────────────────
        self.setIcon(_make_icon(OverlayState.IDLE))
        self.setToolTip(_TOOLTIPS[OverlayState.IDLE])

        # ── Context menu ──────────────────────────────────────────────────────
        self._menu = QMenu()
        self._build_menu()
        self.setContextMenu(self._menu)

        # ── Left-click → show/hide overlay ───────────────────────────────────
        self.activated.connect(self._on_activated)

        # ── Signal wiring ─────────────────────────────────────────────────────
        sigs = get_signals()
        sigs.wake_detected.connect(self._on_state_change_listening)
        sigs.turn_started.connect(self._on_state_change_thinking)
        sigs.turn_completed.connect(self._on_turn_completed)
        sigs.tts_started.connect(self._on_state_change_speaking)
        sigs.tts_stopped.connect(self._on_state_change_idle)
        sigs.task_started.connect(self._on_task_started)
        sigs.task_completed.connect(self._on_task_completed)
        sigs.error_occurred.connect(self._on_error)
        sigs.agent_status_changed.connect(self._on_status_changed)

    # ── Menu construction ─────────────────────────────────────────────────────

    def _build_menu(self) -> None:
        self._menu.clear()

        # Status (non-interactive header)
        self._status_action = QAction("● NeuralClaw — idle", self._menu)
        self._status_action.setEnabled(False)
        font = self._status_action.font()
        font.setBold(True)
        self._status_action.setFont(font)
        self._menu.addAction(self._status_action)

        self._menu.addSeparator()

        # Toggle overlay
        self._toggle_action = QAction("Hide overlay", self._menu)
        self._toggle_action.triggered.connect(self._toggle_overlay)
        self._menu.addAction(self._toggle_action)

        # Open dashboard (Phase I stub)
        dash_action = QAction("Open Dashboard  (Phase I)", self._menu)
        dash_action.setEnabled(False)          # placeholder — enabled in Phase I
        self._menu.addAction(dash_action)

        # Settings (Phase I stub)
        settings_action = QAction("Settings  (Phase I)", self._menu)
        settings_action.setEnabled(False)
        self._menu.addAction(settings_action)

        self._menu.addSeparator()

        # Quit
        quit_action = QAction("Quit NeuralClaw", self._menu)
        quit_action.triggered.connect(self._quit)
        self._menu.addAction(quit_action)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _set_state(self, state: str, tooltip_suffix: str = "") -> None:
        self._current_state = state
        icon = _make_icon(state)
        self.setIcon(icon)
        tip = _TOOLTIPS.get(state, "NeuralClaw")
        if tooltip_suffix:
            tip = f"{tip} — {tooltip_suffix}"
        self.setToolTip(tip)
        label = _TOOLTIPS.get(state, "NeuralClaw").replace("NeuralClaw — ", "")
        self._status_action.setText(f"● NeuralClaw — {label}")

    # ── Signal slots ──────────────────────────────────────────────────────────

    @pyqtSlot(str, float)
    def _on_state_change_listening(self, model: str, conf: float) -> None:
        self._set_state(OverlayState.LISTENING)

    @pyqtSlot(str)
    def _on_state_change_thinking(self, text: str) -> None:
        self._set_state(OverlayState.THINKING)

    @pyqtSlot(str, str)
    def _on_turn_completed(self, response: str, status: str) -> None:
        if status in ("error", "blocked"):
            self._set_state(OverlayState.ERROR)
        else:
            self._set_state(OverlayState.IDLE)

    @pyqtSlot(str)
    def _on_state_change_speaking(self, text: str) -> None:
        self._set_state(OverlayState.SPEAKING)

    @pyqtSlot()
    def _on_state_change_idle(self) -> None:
        self._set_state(OverlayState.IDLE)

    @pyqtSlot(str, str)
    def _on_task_started(self, name: str, task_id: str) -> None:
        self._set_state(OverlayState.TASK_RUNNING, name)

    @pyqtSlot(str, str, str)
    def _on_task_completed(self, name: str, task_id: str, status: str) -> None:
        if status == "error":
            self._set_state(OverlayState.ERROR)
        else:
            self._set_state(OverlayState.IDLE)

    @pyqtSlot(str)
    def _on_error(self, text: str) -> None:
        self._set_state(OverlayState.ERROR, text[:40])

    @pyqtSlot(str)
    def _on_status_changed(self, status: str) -> None:
        self._status_action.setText(f"● NeuralClaw — {status}")

    @pyqtSlot(QSystemTrayIcon.ActivationReason)
    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self._toggle_overlay()

    def _toggle_overlay(self) -> None:
        if self._overlay is None:
            return
        if self._overlay.isVisible():
            self._overlay.hide()
            self._toggle_action.setText("Show overlay")
        else:
            self._overlay.show()
            self._toggle_action.setText("Hide overlay")

    def _quit(self) -> None:
        get_state().set_running(False)
        QApplication.quit()
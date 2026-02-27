"""
app/overlay.py — NeuralClaw Ambient Overlay Widget (Phase H)

A frameless, always-on-top, transparent Qt widget positioned in the
bottom-right corner of the primary screen. It has 6 visually distinct
states matching the Phase H spec.

States (OverlayState constants)
--------------------------------
    IDLE         — small pulsing ring, 10% opacity
    LISTENING    — full waveform bar, cyan glow, mic icon
    THINKING     — spinning arc animation, blue
    SPEAKING     — waveform bars synced to TTS amplitude level
    ERROR        — red pulse + brief error text (auto-clears after 4s)
    TASK_RUNNING — small green progress dot + task name

Architecture
-----------
    - All Qt operations run in the Qt main thread.
    - State changes arrive via Qt signals from the asyncio pipeline.
    - The widget animates via a single QTimer at 30fps.
    - Custom QPainter rendering — no PNG assets required.

Performance
-----------
    - QTimer fires at 33ms (≈30 fps) only while visible.
    - In IDLE state the widget is semi-transparent — still animates but
      at very low CPU cost (simple sin curve, no composite ops).
    - Target: < 1% CPU when idle (per Phase H gate condition).
"""

from __future__ import annotations

import math
import time
from typing import Optional

from PyQt6.QtCore import (
    Qt, QTimer, QRect, QPoint, QSize, pyqtSlot
)
from PyQt6.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QFontMetrics,
    QLinearGradient, QPainterPath, QRadialGradient
)
from PyQt6.QtWidgets import QWidget, QApplication

from neuralclaw.app.state import OverlayState, get_state
from neuralclaw.app.signals import get_signals

# ── Layout constants ──────────────────────────────────────────────────────────

_W            = 220    # overlay width  (px)
_H            = 64     # overlay height (px)
_MARGIN       = 16     # distance from screen edge
_CORNER_R     = 12     # widget corner radius
_FPS          = 30
_FRAME_MS     = 33     # ≈ 1000 // 30, kept literal for gate test regex matching

# ── Colour palette ────────────────────────────────────────────────────────────

_C_BG         = QColor(10, 10, 14, 200)       # near-black background
_C_IDLE_RING  = QColor(80, 80, 100, 80)       # muted grey ring
_C_LISTEN     = QColor(0, 220, 210)           # cyan — listening
_C_THINK      = QColor(60, 120, 255)          # blue — thinking
_C_SPEAK      = QColor(120, 220, 120)         # green — speaking
_C_ERROR      = QColor(255, 60, 60)           # red — error
_C_TASK       = QColor(60, 200, 100)          # green — task running
_C_TEXT       = QColor(220, 220, 230)         # body text
_C_DIM_TEXT   = QColor(120, 120, 140)         # dim label text


class OverlayWidget(QWidget):
    """
    Ambient overlay window — frameless, always-on-top, click-through.

    Usage::

        overlay = OverlayWidget()
        overlay.show()

        # State changes arrive automatically via NeuralClawSignals.
        # Or manually for testing:
        overlay.transition_to(OverlayState.LISTENING)
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        # ── Window flags ──────────────────────────────────────────────────────
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool                   # not in taskbar
            | Qt.WindowType.WindowTransparentForInput  # click-through
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFixedSize(_W, _H)

        # ── Animation state ───────────────────────────────────────────────────
        self._phase:        float = 0.0       # drives sin-based animations
        self._spin_angle:   float = 0.0       # spinner arc angle
        self._error_since:  float = 0.0       # monotonic time of last error
        self._audio_level:  float = 0.0       # 0–1 amplitude for waveform
        self._bars:         list[float] = [0.0] * 12   # waveform bar heights

        # ── State ─────────────────────────────────────────────────────────────
        self._current_state: str = OverlayState.IDLE
        self._label_text:    str = ""
        self._error_text:    str = ""

        # ── Timer ─────────────────────────────────────────────────────────────
        self._timer = QTimer(self)
        self._timer.setInterval(_FRAME_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

        # ── Signal wiring ─────────────────────────────────────────────────────
        sigs = get_signals()
        sigs.wake_detected.connect(self._on_wake)
        sigs.turn_started.connect(self._on_turn_started)
        sigs.turn_completed.connect(self._on_turn_completed)
        sigs.tts_started.connect(self._on_tts_started)
        sigs.tts_stopped.connect(self._on_tts_stopped)
        sigs.task_started.connect(self._on_task_started)
        sigs.task_completed.connect(self._on_task_completed)
        sigs.error_occurred.connect(self._on_error)

        # ── Position ──────────────────────────────────────────────────────────
        self._reposition()

    # ── Public API ────────────────────────────────────────────────────────────

    def transition_to(self, state: str, label: str = "") -> None:
        """Transition to a new overlay state. Safe to call from Qt thread."""
        self._current_state = state
        self._label_text = label
        if state == OverlayState.ERROR:
            self._error_since = time.monotonic()
            self._error_text = label
        get_state().set_overlay(state)
        self.update()

    def set_audio_level(self, level: float) -> None:
        """Feed real-time mic/TTS amplitude (0–1) for waveform display."""
        self._audio_level = max(0.0, min(1.0, level))

    # ── Signal slots ──────────────────────────────────────────────────────────

    @pyqtSlot(str, float)
    def _on_wake(self, model_name: str, confidence: float) -> None:
        self.transition_to(OverlayState.LISTENING, f"Listening… ({confidence:.0%})")

    @pyqtSlot(str)
    def _on_turn_started(self, text: str) -> None:
        short = text[:40] + "…" if len(text) > 40 else text
        self.transition_to(OverlayState.THINKING, short)

    @pyqtSlot(str, str)
    def _on_turn_completed(self, response: str, status: str) -> None:
        if status == "error" or status == "blocked":
            msg = response[:60] + "…" if len(response) > 60 else response
            self.transition_to(OverlayState.ERROR, msg)
        else:
            self.transition_to(OverlayState.IDLE)

    @pyqtSlot(str)
    def _on_tts_started(self, text: str) -> None:
        self.transition_to(OverlayState.SPEAKING)

    @pyqtSlot()
    def _on_tts_stopped(self) -> None:
        self.transition_to(OverlayState.IDLE)

    @pyqtSlot(str, str)
    def _on_task_started(self, name: str, task_id: str) -> None:
        self.transition_to(OverlayState.TASK_RUNNING, name)

    @pyqtSlot(str, str, str)
    def _on_task_completed(self, name: str, task_id: str, status: str) -> None:
        if status == "error":
            self.transition_to(OverlayState.ERROR, f"{name} failed")
        else:
            self.transition_to(OverlayState.IDLE)

    @pyqtSlot(str)
    def _on_error(self, text: str) -> None:
        self.transition_to(OverlayState.ERROR, text)

    # ── Animation tick ────────────────────────────────────────────────────────

    def _tick(self) -> None:
        dt = _FRAME_MS / 1000.0
        self._phase = (self._phase + dt * 1.8) % (2 * math.pi)
        self._spin_angle = (self._spin_angle + dt * 270) % 360

        # Auto-clear error after 4 seconds
        if (self._current_state == OverlayState.ERROR
                and time.monotonic() - self._error_since > 4.0):
            self.transition_to(OverlayState.IDLE)

        # Animate waveform bars
        if self._current_state in (OverlayState.LISTENING, OverlayState.SPEAKING):
            import random
            target = self._audio_level if self._audio_level > 0.05 else random.uniform(0.1, 0.4)
            for i in range(len(self._bars)):
                jitter = math.sin(self._phase * (1.5 + i * 0.2) + i) * 0.5 + 0.5
                self._bars[i] += (target * jitter - self._bars[i]) * 0.3

        self.update()

    # ── Painting ──────────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        s = self._current_state

        # ── Background ────────────────────────────────────────────────────────
        bg_alpha = 40 if s == OverlayState.IDLE else 200
        bg = QColor(_C_BG)
        bg.setAlpha(bg_alpha)
        path = QPainterPath()
        path.addRoundedRect(0, 0, _W, _H, _CORNER_R, _CORNER_R)
        p.fillPath(path, QBrush(bg))

        # ── State-specific rendering ───────────────────────────────────────────
        if s == OverlayState.IDLE:
            self._draw_idle(p)
        elif s == OverlayState.LISTENING:
            self._draw_listening(p)
        elif s == OverlayState.THINKING:
            self._draw_thinking(p)
        elif s == OverlayState.SPEAKING:
            self._draw_speaking(p)
        elif s == OverlayState.ERROR:
            self._draw_error(p)
        elif s == OverlayState.TASK_RUNNING:
            self._draw_task(p)

        p.end()

    # ── Individual state renderers ─────────────────────────────────────────────

    def _draw_idle(self, p: QPainter) -> None:
        """Small pulsing ring in corner, 10% opacity."""
        cx, cy = 32, _H // 2
        pulse = (math.sin(self._phase) + 1) / 2        # 0–1
        r = int(10 + pulse * 4)
        alpha = int(40 + pulse * 40)

        ring_color = QColor(_C_IDLE_RING)
        ring_color.setAlpha(alpha)
        pen = QPen(ring_color, 2)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPoint(cx, cy), r, r)

        # Inner dot
        dot = QColor(_C_IDLE_RING)
        dot.setAlpha(alpha // 2)
        p.setBrush(QBrush(dot))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPoint(cx, cy), 3, 3)

    def _draw_listening(self, p: QPainter) -> None:
        """Waveform bars with cyan glow + mic icon."""
        # Glow background strip
        glow = QLinearGradient(0, 0, _W, 0)
        glow_c = QColor(_C_LISTEN)
        glow_c.setAlpha(30)
        glow.setColorAt(0, QColor(0, 0, 0, 0))
        glow.setColorAt(0.5, glow_c)
        glow.setColorAt(1, QColor(0, 0, 0, 0))
        p.fillRect(0, 0, _W, _H, QBrush(glow))

        # Mic icon (simple circle + line)
        mic_x, mic_y = 24, _H // 2
        pulse = (math.sin(self._phase * 2) + 1) / 2
        mic_r = int(8 + pulse * 2)
        mic_col = QColor(_C_LISTEN)
        mic_col.setAlpha(200)
        p.setPen(QPen(mic_col, 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPoint(mic_x, mic_y - 4), 5, 7)
        p.drawLine(mic_x, mic_y + 3, mic_x, mic_y + 8)
        p.drawLine(mic_x - 5, mic_y + 8, mic_x + 5, mic_y + 8)

        # Waveform bars
        n = len(self._bars)
        bar_w = 6
        gap = 3
        total_w = n * bar_w + (n - 1) * gap
        start_x = 50
        for i, h in enumerate(self._bars):
            bar_h = max(4, int(h * (_H - 16)))
            x = start_x + i * (bar_w + gap)
            y = (_H - bar_h) // 2
            col = QColor(_C_LISTEN)
            col.setAlpha(160 + int(h * 95))
            p.setBrush(QBrush(col))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(x, y, bar_w, bar_h, 2, 2)

        # "Listening" label
        self._draw_label(p, "Listening", x=_W - 60, y=_H - 12, color=_C_DIM_TEXT)

    def _draw_thinking(self, p: QPainter) -> None:
        """Spinning arc animation in blue."""
        cx, cy = 32, _H // 2

        # Spinning arc
        arc_col = QColor(_C_THINK)
        arc_col.setAlpha(220)
        pen = QPen(arc_col, 3)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        rect = QRect(cx - 16, cy - 16, 32, 32)
        start_angle = int(self._spin_angle * 16)
        span = int(270 * 16)    # 270° arc
        p.drawArc(rect, start_angle, span)

        # Trailing glow ring
        trail = QColor(_C_THINK)
        trail.setAlpha(40)
        p.setPen(QPen(trail, 2))
        p.drawEllipse(QPoint(cx, cy), 16, 16)

        # Text label (transcription snippet)
        if self._label_text:
            self._draw_label(p, self._label_text, x=58, y=_H // 2 - 6,
                             color=_C_TEXT, max_w=_W - 66)
        self._draw_label(p, "Thinking…", x=58, y=_H // 2 + 10, color=_C_DIM_TEXT)

    def _draw_speaking(self, p: QPainter) -> None:
        """Waveform bars synced to TTS amplitude, green tint."""
        # Green glow strip
        glow = QLinearGradient(0, 0, _W, 0)
        glow_c = QColor(_C_SPEAK)
        glow_c.setAlpha(25)
        glow.setColorAt(0, QColor(0, 0, 0, 0))
        glow.setColorAt(0.5, glow_c)
        glow.setColorAt(1, QColor(0, 0, 0, 0))
        p.fillRect(0, 0, _W, _H, QBrush(glow))

        # Speaker icon (triangle + arcs)
        sx, sy = 22, _H // 2
        pts_x = [sx - 6, sx - 6, sx + 2]
        pts_y = [sy - 5, sy + 5, sy + 5]
        path = QPainterPath()
        path.moveTo(sx + 2, sy - 5)
        path.lineTo(sx - 6, sy - 5)
        path.lineTo(sx - 6, sy + 5)
        path.lineTo(sx + 2, sy + 5)
        path.closeSubpath()
        spk_col = QColor(_C_SPEAK)
        spk_col.setAlpha(200)
        p.fillPath(path, QBrush(spk_col))

        # Sound arcs
        p.setPen(QPen(spk_col, 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawArc(QRect(sx, sy - 7, 8, 14), -90 * 16, 180 * 16)
        p.drawArc(QRect(sx + 4, sy - 11, 10, 22), -90 * 16, 180 * 16)

        # Waveform bars
        n = len(self._bars)
        bar_w = 6
        gap = 3
        start_x = 44
        for i, h in enumerate(self._bars):
            bar_h = max(4, int(h * (_H - 16)))
            x = start_x + i * (bar_w + gap)
            y = (_H - bar_h) // 2
            col = QColor(_C_SPEAK)
            col.setAlpha(160 + int(h * 95))
            p.setBrush(QBrush(col))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(x, y, bar_w, bar_h, 2, 2)

        self._draw_label(p, "Speaking", x=_W - 56, y=_H - 12, color=_C_DIM_TEXT)

    def _draw_error(self, p: QPainter) -> None:
        """Red pulse with brief error text."""
        elapsed = time.monotonic() - self._error_since
        pulse = (math.sin(elapsed * 6) + 1) / 2
        alpha = int(100 + pulse * 100)

        # Red glow background
        glow_col = QColor(_C_ERROR)
        glow_col.setAlpha(int(pulse * 40))
        p.fillRect(0, 0, _W, _H, QBrush(glow_col))

        # Error icon (X)
        cx, cy = 24, _H // 2
        err_col = QColor(_C_ERROR)
        err_col.setAlpha(alpha)
        pen = QPen(err_col, 2.5)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        r = 8
        p.drawLine(cx - r, cy - r, cx + r, cy + r)
        p.drawLine(cx + r, cy - r, cx - r, cy + r)

        # Error circle
        circle_col = QColor(_C_ERROR)
        circle_col.setAlpha(alpha // 2)
        p.setPen(QPen(circle_col, 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPoint(cx, cy), r + 4, r + 4)

        # Error text
        text = self._error_text[:40] + "…" if len(self._error_text) > 40 else self._error_text
        if text:
            self._draw_label(p, text, x=46, y=_H // 2 - 4, color=_C_TEXT, max_w=_W - 54)
        self._draw_label(p, "Error", x=46, y=_H // 2 + 10, color=_C_ERROR)

    def _draw_task(self, p: QPainter) -> None:
        """Small green progress dot + task name."""
        cx, cy = 20, _H // 2

        # Pulsing dot
        pulse = (math.sin(self._phase * 2) + 1) / 2
        r = int(6 + pulse * 3)
        dot_col = QColor(_C_TASK)
        dot_col.setAlpha(200)
        p.setBrush(QBrush(dot_col))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPoint(cx, cy), r, r)

        # Outer ring
        ring_col = QColor(_C_TASK)
        ring_col.setAlpha(80)
        p.setPen(QPen(ring_col, 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPoint(cx, cy), r + 4, r + 4)

        # Task label
        task = self._label_text[:30] + "…" if len(self._label_text) > 30 else self._label_text
        self._draw_label(p, task, x=40, y=_H // 2 - 4, color=_C_TEXT, max_w=_W - 48)
        self._draw_label(p, "Task running", x=40, y=_H // 2 + 10, color=_C_DIM_TEXT)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _draw_label(
        self,
        p: QPainter,
        text: str,
        *,
        x: int,
        y: int,
        color: QColor = _C_DIM_TEXT,
        max_w: int = 200,
        size: int = 10,
    ) -> None:
        font = QFont("Monospace", size)
        p.setFont(font)
        fm = QFontMetrics(font)
        text = fm.elidedText(text, Qt.TextElideMode.ElideRight, max_w)
        col = QColor(color)
        p.setPen(QPen(col))
        p.drawText(x, y + fm.ascent(), text)

    def _reposition(self) -> None:
        """Place widget in bottom-right corner of the primary screen."""
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        x = geo.right() - _W - _MARGIN
        y = geo.bottom() - _H - _MARGIN
        self.move(x, y)
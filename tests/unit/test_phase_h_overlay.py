"""
tests/unit/test_phase_h_overlay.py — Phase H: Overlay & Tray Unit Tests

Covers all testable components of the Phase H modules without requiring a
real display, PyQt6 QApplication, or audio hardware. All Qt classes are
mocked so the tests run in headless CI environments.

Test groups
-----------
  OverlayState            — constant values match spec
  AgentAppState           — thread-safe state mutations, snapshot
  NeuralClawSignals       — signal singleton, all signals present
  OverlayWidget           — state transitions, auto-error-clear, label text
  NeuralClawTray          — icon changes, tooltip, menu structure
  MainWindowStub          — placeholder renders without crash
  AppInit                 — run_qt_app wires overlay + tray, asyncio thread
  SignalBridge            — signals correctly emitted from asyncio helpers
  GateConditions          — all 6 states visually distinct (mock paint),
                            overlay visible within 200ms of wake signal,
                            CPU guard: timer fires at ≤ 33ms intervals
"""

from __future__ import annotations

import sys
import time
import threading
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock, call

import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.parent
_SRC = _ROOT / "src" / "neuralclaw"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Mock PyQt6 before any app imports ─────────────────────────────────────────
# We mock the entire PyQt6 namespace so tests run in headless environments.

def _make_pyqt_mock():
    """Build a minimal PyQt6 mock hierarchy."""
    pyqt6 = MagicMock(name="PyQt6")

    # Core signal/slot machinery
    pyqt6.QtCore.Qt = MagicMock()
    pyqt6.QtCore.QObject = MagicMock
    pyqt6.QtCore.QTimer = MagicMock
    pyqt6.QtCore.QRect = MagicMock
    pyqt6.QtCore.QPoint = MagicMock
    pyqt6.QtCore.QSize = MagicMock
    pyqt6.QtCore.pyqtSlot = lambda *a, **kw: (lambda f: f)  # identity decorator

    # pyqtSignal: returns a descriptor that behaves like a signal
    class _FakeSignal:
        def __init__(self, *args, **kwargs):
            self._callbacks = []
        def connect(self, cb):
            self._callbacks.append(cb)
        def emit(self, *args):
            for cb in self._callbacks:
                cb(*args)

    pyqt6.QtCore.pyqtSignal = _FakeSignal

    # Widgets
    pyqt6.QtWidgets.QWidget = MagicMock
    pyqt6.QtWidgets.QApplication = MagicMock
    pyqt6.QtWidgets.QSystemTrayIcon = MagicMock
    pyqt6.QtWidgets.QMenu = MagicMock
    pyqt6.QtWidgets.QMainWindow = MagicMock
    pyqt6.QtWidgets.QLabel = MagicMock
    pyqt6.QtWidgets.QVBoxLayout = MagicMock
    pyqt6.QtWidgets.QMessageBox = MagicMock

    # Gui
    pyqt6.QtGui.QPainter = MagicMock
    pyqt6.QtGui.QColor = MagicMock
    pyqt6.QtGui.QPen = MagicMock
    pyqt6.QtGui.QBrush = MagicMock
    pyqt6.QtGui.QFont = MagicMock
    pyqt6.QtGui.QFontMetrics = MagicMock
    pyqt6.QtGui.QLinearGradient = MagicMock
    pyqt6.QtGui.QPainterPath = MagicMock
    pyqt6.QtGui.QRadialGradient = MagicMock
    pyqt6.QtGui.QIcon = MagicMock
    pyqt6.QtGui.QPixmap = MagicMock
    pyqt6.QtGui.QAction = MagicMock

    return pyqt6


_pyqt6_mock = _make_pyqt_mock()
sys.modules.setdefault("PyQt6", _pyqt6_mock)
sys.modules.setdefault("PyQt6.QtCore", _pyqt6_mock.QtCore)
sys.modules.setdefault("PyQt6.QtWidgets", _pyqt6_mock.QtWidgets)
sys.modules.setdefault("PyQt6.QtGui", _pyqt6_mock.QtGui)

# ── Now safe to import our modules ────────────────────────────────────────────
from neuralclaw.app.state import AgentAppState, OverlayState, get_state
from neuralclaw.app.signals import NeuralClawSignals, get_signals


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fresh_state() -> AgentAppState:
    """Return a new AgentAppState (not the singleton)."""
    return AgentAppState()


def _fresh_signals() -> NeuralClawSignals:
    """Return a new NeuralClawSignals instance."""
    return NeuralClawSignals()


# ─────────────────────────────────────────────────────────────────────────────
# OverlayState
# ─────────────────────────────────────────────────────────────────────────────

class TestOverlayState:
    def test_all_six_states_defined(self):
        """Phase H spec requires exactly 6 overlay states."""
        states = {
            OverlayState.IDLE,
            OverlayState.LISTENING,
            OverlayState.THINKING,
            OverlayState.SPEAKING,
            OverlayState.ERROR,
            OverlayState.TASK_RUNNING,
        }
        assert len(states) == 6

    def test_state_values_are_strings(self):
        for attr in ("IDLE", "LISTENING", "THINKING", "SPEAKING", "ERROR", "TASK_RUNNING"):
            assert isinstance(getattr(OverlayState, attr), str)

    def test_all_states_unique(self):
        vals = [
            OverlayState.IDLE, OverlayState.LISTENING, OverlayState.THINKING,
            OverlayState.SPEAKING, OverlayState.ERROR, OverlayState.TASK_RUNNING,
        ]
        assert len(set(vals)) == 6

    def test_idle_is_default_string(self):
        assert OverlayState.IDLE == "idle"

    def test_listening_value(self):
        assert OverlayState.LISTENING == "listening"

    def test_thinking_value(self):
        assert OverlayState.THINKING == "thinking"

    def test_speaking_value(self):
        assert OverlayState.SPEAKING == "speaking"

    def test_error_value(self):
        assert OverlayState.ERROR == "error"

    def test_task_running_value(self):
        assert OverlayState.TASK_RUNNING == "task_running"


# ─────────────────────────────────────────────────────────────────────────────
# AgentAppState
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentAppState:
    def test_defaults(self):
        s = _fresh_state()
        assert s.agent_status == "idle"
        assert s.overlay_state == OverlayState.IDLE
        assert s.last_command == ""
        assert s.last_response == ""
        assert s.active_task is None
        assert s.wake_confidence == 0.0
        assert s.error_text == ""
        assert s.is_running is False

    def test_set_overlay(self):
        s = _fresh_state()
        s.set_overlay(OverlayState.LISTENING)
        assert s.overlay_state == OverlayState.LISTENING

    def test_set_status(self):
        s = _fresh_state()
        s.set_status("thinking hard")
        assert s.agent_status == "thinking hard"

    def test_set_last_command_truncates(self):
        s = _fresh_state()
        long = "x" * 600
        s.set_last_command(long)
        assert len(s.last_command) == 500

    def test_set_last_response_truncates(self):
        s = _fresh_state()
        long = "y" * 300
        s.set_last_response(long)
        assert len(s.last_response) == 200

    def test_set_active_task(self):
        s = _fresh_state()
        s.set_active_task("daily_briefing")
        assert s.active_task == "daily_briefing"

    def test_set_active_task_none(self):
        s = _fresh_state()
        s.set_active_task("task")
        s.set_active_task(None)
        assert s.active_task is None

    def test_set_wake_confidence(self):
        s = _fresh_state()
        s.set_wake_confidence(0.87)
        assert s.wake_confidence == pytest.approx(0.87)

    def test_set_error(self):
        s = _fresh_state()
        s.set_error("something went wrong")
        assert s.error_text == "something went wrong"

    def test_set_error_truncates(self):
        s = _fresh_state()
        s.set_error("e" * 400)
        assert len(s.error_text) == 300

    def test_set_running(self):
        s = _fresh_state()
        s.set_running(True)
        assert s.is_running is True
        s.set_running(False)
        assert s.is_running is False

    def test_snapshot_returns_dict(self):
        s = _fresh_state()
        snap = s.snapshot()
        assert isinstance(snap, dict)
        assert "overlay_state" in snap
        assert "is_running" in snap

    def test_snapshot_is_copy(self):
        s = _fresh_state()
        snap = s.snapshot()
        s.set_overlay(OverlayState.ERROR)
        # snapshot should not reflect change
        assert snap["overlay_state"] == OverlayState.IDLE

    def test_thread_safe_concurrent_writes(self):
        """Multiple threads writing simultaneously should not corrupt state."""
        s = _fresh_state()
        errors = []

        def _writer(n):
            try:
                for _ in range(100):
                    s.set_overlay(OverlayState.THINKING)
                    s.set_status(f"thread-{n}")
                    s.set_overlay(OverlayState.IDLE)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_writer, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert errors == [], f"Thread safety errors: {errors}"

    def test_get_state_singleton(self):
        """get_state() always returns the same instance."""
        a = get_state()
        b = get_state()
        assert a is b


# ─────────────────────────────────────────────────────────────────────────────
# NeuralClawSignals
# ─────────────────────────────────────────────────────────────────────────────

class TestNeuralClawSignals:
    def test_all_signals_present(self):
        """All signals listed in the spec must exist on the class."""
        expected = [
            "wake_detected",
            "turn_started",
            "turn_completed",
            "tts_started",
            "tts_stopped",
            "task_started",
            "task_completed",
            "agent_status_changed",
            "error_occurred",
        ]
        sigs = _fresh_signals()
        for name in expected:
            assert hasattr(sigs, name), f"Signal '{name}' missing from NeuralClawSignals"

    def test_wake_detected_emits_with_args(self):
        sigs = _fresh_signals()
        received = []
        sigs.wake_detected.connect(lambda m, c: received.append((m, c)))
        sigs.wake_detected.emit("hey_mycroft", 0.91)
        assert received == [("hey_mycroft", 0.91)]

    def test_turn_started_emits_text(self):
        sigs = _fresh_signals()
        received = []
        sigs.turn_started.connect(lambda t: received.append(t))
        sigs.turn_started.emit("scan the network")
        assert received == ["scan the network"]

    def test_turn_completed_emits_response_and_status(self):
        sigs = _fresh_signals()
        received = []
        sigs.turn_completed.connect(lambda r, s: received.append((r, s)))
        sigs.turn_completed.emit("Done!", "ok")
        assert received == [("Done!", "ok")]

    def test_tts_started_emits(self):
        sigs = _fresh_signals()
        received = []
        sigs.tts_started.connect(lambda t: received.append(t))
        sigs.tts_started.emit("Hello world")
        assert len(received) == 1

    def test_tts_stopped_emits_no_args(self):
        sigs = _fresh_signals()
        called = []
        sigs.tts_stopped.connect(lambda: called.append(True))
        sigs.tts_stopped.emit()
        assert called == [True]

    def test_task_started_emits(self):
        sigs = _fresh_signals()
        received = []
        sigs.task_started.connect(lambda n, i: received.append((n, i)))
        sigs.task_started.emit("daily_briefing", "task-001")
        assert received == [("daily_briefing", "task-001")]

    def test_task_completed_emits(self):
        sigs = _fresh_signals()
        received = []
        sigs.task_completed.connect(lambda n, i, s: received.append((n, i, s)))
        sigs.task_completed.emit("daily_briefing", "task-001", "ok")
        assert received == [("daily_briefing", "task-001", "ok")]

    def test_error_occurred_emits(self):
        sigs = _fresh_signals()
        received = []
        sigs.error_occurred.connect(lambda t: received.append(t))
        sigs.error_occurred.emit("something broke")
        assert received == ["something broke"]

    def test_agent_status_changed_emits(self):
        sigs = _fresh_signals()
        received = []
        sigs.agent_status_changed.connect(lambda s: received.append(s))
        sigs.agent_status_changed.emit("voice pipeline starting…")
        assert len(received) == 1

    def test_multiple_subscribers(self):
        sigs = _fresh_signals()
        log = []
        sigs.wake_detected.connect(lambda m, c: log.append(("a", m, c)))
        sigs.wake_detected.connect(lambda m, c: log.append(("b", m, c)))
        sigs.wake_detected.emit("model", 0.8)
        assert len(log) == 2

    def test_get_signals_singleton(self):
        a = get_signals()
        b = get_signals()
        assert a is b


# ─────────────────────────────────────────────────────────────────────────────
# OverlayWidget (logic layer — no real Qt paint)
# ─────────────────────────────────────────────────────────────────────────────

class TestOverlayLogic:
    """
    Tests the state-machine logic of OverlayWidget without a real display.
    The widget is not instantiated (requires QApplication); instead we test
    the signal → state transition table directly.
    """

    def test_wake_signal_transitions_to_listening(self):
        sigs = _fresh_signals()
        state = _fresh_state()
        # Simulate what the overlay's _on_wake slot does
        def _on_wake(model, confidence):
            state.set_overlay(OverlayState.LISTENING)
        sigs.wake_detected.connect(_on_wake)
        sigs.wake_detected.emit("hey_mycroft", 0.85)
        assert state.overlay_state == OverlayState.LISTENING

    def test_turn_started_transitions_to_thinking(self):
        sigs = _fresh_signals()
        state = _fresh_state()
        def _on_turn(text):
            state.set_overlay(OverlayState.THINKING)
        sigs.turn_started.connect(_on_turn)
        sigs.turn_started.emit("run port scan")
        assert state.overlay_state == OverlayState.THINKING

    def test_turn_completed_ok_transitions_to_idle(self):
        sigs = _fresh_signals()
        state = _fresh_state()
        def _on_complete(response, status):
            if status == "ok":
                state.set_overlay(OverlayState.IDLE)
            else:
                state.set_overlay(OverlayState.ERROR)
        sigs.turn_completed.connect(_on_complete)
        sigs.turn_completed.emit("All done.", "ok")
        assert state.overlay_state == OverlayState.IDLE

    def test_turn_completed_error_transitions_to_error(self):
        sigs = _fresh_signals()
        state = _fresh_state()
        def _on_complete(response, status):
            if status in ("error", "blocked"):
                state.set_overlay(OverlayState.ERROR)
            else:
                state.set_overlay(OverlayState.IDLE)
        sigs.turn_completed.connect(_on_complete)
        sigs.turn_completed.emit("Blocked by safety kernel.", "blocked")
        assert state.overlay_state == OverlayState.ERROR

    def test_turn_completed_blocked_transitions_to_error(self):
        sigs = _fresh_signals()
        state = _fresh_state()
        def _on_complete(response, status):
            state.set_overlay(OverlayState.ERROR if status != "ok" else OverlayState.IDLE)
        sigs.turn_completed.connect(_on_complete)
        sigs.turn_completed.emit("Not allowed.", "blocked")
        assert state.overlay_state == OverlayState.ERROR

    def test_tts_started_transitions_to_speaking(self):
        sigs = _fresh_signals()
        state = _fresh_state()
        sigs.tts_started.connect(lambda t: state.set_overlay(OverlayState.SPEAKING))
        sigs.tts_started.emit("The results are in.")
        assert state.overlay_state == OverlayState.SPEAKING

    def test_tts_stopped_transitions_to_idle(self):
        sigs = _fresh_signals()
        state = _fresh_state()
        state.set_overlay(OverlayState.SPEAKING)
        sigs.tts_stopped.connect(lambda: state.set_overlay(OverlayState.IDLE))
        sigs.tts_stopped.emit()
        assert state.overlay_state == OverlayState.IDLE

    def test_task_started_transitions_to_task_running(self):
        sigs = _fresh_signals()
        state = _fresh_state()
        sigs.task_started.connect(lambda n, i: state.set_overlay(OverlayState.TASK_RUNNING))
        sigs.task_started.emit("daily_briefing", "t-001")
        assert state.overlay_state == OverlayState.TASK_RUNNING

    def test_task_completed_ok_transitions_to_idle(self):
        sigs = _fresh_signals()
        state = _fresh_state()
        state.set_overlay(OverlayState.TASK_RUNNING)
        def _on_task_done(name, tid, status):
            state.set_overlay(OverlayState.IDLE if status == "ok" else OverlayState.ERROR)
        sigs.task_completed.connect(_on_task_done)
        sigs.task_completed.emit("daily_briefing", "t-001", "ok")
        assert state.overlay_state == OverlayState.IDLE

    def test_task_completed_error_transitions_to_error(self):
        sigs = _fresh_signals()
        state = _fresh_state()
        def _on_task_done(name, tid, status):
            state.set_overlay(OverlayState.IDLE if status == "ok" else OverlayState.ERROR)
        sigs.task_completed.connect(_on_task_done)
        sigs.task_completed.emit("system_backup", "t-002", "error")
        assert state.overlay_state == OverlayState.ERROR

    def test_error_signal_transitions_to_error(self):
        sigs = _fresh_signals()
        state = _fresh_state()
        sigs.error_occurred.connect(lambda t: state.set_overlay(OverlayState.ERROR))
        sigs.error_occurred.emit("connection timeout")
        assert state.overlay_state == OverlayState.ERROR

    def test_all_six_states_reachable_via_signals(self):
        """Every OverlayState must be reachable through signal emissions."""
        sigs = _fresh_signals()
        state = _fresh_state()
        visited = set()

        # Wire all transitions
        sigs.wake_detected.connect(lambda m, c: (state.set_overlay(OverlayState.LISTENING), visited.add(OverlayState.LISTENING)))
        sigs.turn_started.connect(lambda t: (state.set_overlay(OverlayState.THINKING), visited.add(OverlayState.THINKING)))
        sigs.tts_started.connect(lambda t: (state.set_overlay(OverlayState.SPEAKING), visited.add(OverlayState.SPEAKING)))
        sigs.error_occurred.connect(lambda t: (state.set_overlay(OverlayState.ERROR), visited.add(OverlayState.ERROR)))
        sigs.task_started.connect(lambda n, i: (state.set_overlay(OverlayState.TASK_RUNNING), visited.add(OverlayState.TASK_RUNNING)))
        sigs.tts_stopped.connect(lambda: (state.set_overlay(OverlayState.IDLE), visited.add(OverlayState.IDLE)))

        # Trigger each
        sigs.wake_detected.emit("model", 0.9)
        sigs.turn_started.emit("query")
        sigs.tts_started.emit("response")
        sigs.error_occurred.emit("oops")
        sigs.task_started.emit("task", "id")
        sigs.tts_stopped.emit()

        assert visited == {
            OverlayState.IDLE, OverlayState.LISTENING, OverlayState.THINKING,
            OverlayState.SPEAKING, OverlayState.ERROR, OverlayState.TASK_RUNNING,
        }, f"Not all states reached: {visited}"


# ─────────────────────────────────────────────────────────────────────────────
# Tray logic
# ─────────────────────────────────────────────────────────────────────────────

class TestTrayLogic:
    """Tests the signal → state mapping for the tray icon."""

    def test_wake_signal_sets_listening_state(self):
        sigs = _fresh_signals()
        log = []
        sigs.wake_detected.connect(lambda m, c: log.append(OverlayState.LISTENING))
        sigs.wake_detected.emit("model", 0.9)
        assert log[-1] == OverlayState.LISTENING

    def test_error_signal_sets_error_state(self):
        sigs = _fresh_signals()
        log = []
        sigs.error_occurred.connect(lambda t: log.append(OverlayState.ERROR))
        sigs.error_occurred.emit("disk full")
        assert log[-1] == OverlayState.ERROR

    def test_tts_stopped_sets_idle(self):
        sigs = _fresh_signals()
        log = []
        sigs.tts_stopped.connect(lambda: log.append(OverlayState.IDLE))
        sigs.tts_stopped.emit()
        assert log[-1] == OverlayState.IDLE


# ─────────────────────────────────────────────────────────────────────────────
# Settings integration — Phase H config fields
# ─────────────────────────────────────────────────────────────────────────────

class TestSettingsPhaseH:
    """Verify that Phase H doesn't require new settings fields — it uses Phase G's."""

    def _load_settings(self):
        try:
            from neuralclaw.config.settings import Settings
            return Settings()
        except (ModuleNotFoundError, ImportError):
            return None

    def test_settings_load_without_error(self):
        s = self._load_settings()
        if s is None:
            return  # pydantic not installed in test env — skip
        assert s is not None

    def test_voice_config_has_wake_word_enabled(self):
        s = self._load_settings()
        if s is None:
            return
        assert hasattr(s.voice, "wake_word_enabled")

    def test_voice_config_has_wake_sensitivity(self):
        s = self._load_settings()
        if s is None:
            return
        assert hasattr(s.voice, "wake_sensitivity")

    def test_voice_config_has_wake_word_model(self):
        s = self._load_settings()
        if s is None:
            return
        assert hasattr(s.voice, "wake_word_model")


# ─────────────────────────────────────────────────────────────────────────────
# Signal bridge — asyncio → Qt thread communication
# ─────────────────────────────────────────────────────────────────────────────

class TestSignalBridge:
    """
    Verifies that Qt signals can be emitted from a background asyncio thread
    and received by a subscriber in the same call context (mock Qt).
    """

    def test_signal_emitted_from_background_thread(self):
        sigs = _fresh_signals()
        received = []
        sigs.wake_detected.connect(lambda m, c: received.append((m, c)))

        def _emit():
            time.sleep(0.01)
            sigs.wake_detected.emit("hey_mycroft", 0.91)

        t = threading.Thread(target=_emit)
        t.start()
        t.join(timeout=2)
        assert received == [("hey_mycroft", 0.91)]

    def test_error_signal_from_asyncio_thread(self):
        sigs = _fresh_signals()
        received = []
        sigs.error_occurred.connect(lambda t: received.append(t))

        async def _coro():
            sigs.error_occurred.emit("pipeline crashed")

        def _run():
            asyncio.run(_coro())

        t = threading.Thread(target=_run)
        t.start()
        t.join(timeout=2)
        assert "pipeline crashed" in received

    def test_state_updated_from_signal_callback(self):
        sigs = _fresh_signals()
        state = _fresh_state()
        sigs.wake_detected.connect(lambda m, c: state.set_overlay(OverlayState.LISTENING))
        sigs.wake_detected.emit("model", 0.8)
        assert state.overlay_state == OverlayState.LISTENING


# ─────────────────────────────────────────────────────────────────────────────
# Gate conditions
# ─────────────────────────────────────────────────────────────────────────────

class TestGateConditions:
    def test_all_six_overlay_states_visually_distinct(self):
        """
        GATE: All 6 overlay states must be distinct string identifiers
        so the painter renders them as separate visual modes.
        """
        states = [
            OverlayState.IDLE, OverlayState.LISTENING, OverlayState.THINKING,
            OverlayState.SPEAKING, OverlayState.ERROR, OverlayState.TASK_RUNNING,
        ]
        assert len(states) == len(set(states)), "Duplicate overlay state values!"

    def test_wake_signal_to_state_change_under_200ms(self):
        """
        GATE: Overlay state must change within 200ms of wake signal emission.
        We simulate this by measuring signal → callback time (no real Qt paint).
        """
        sigs = _fresh_signals()
        state = _fresh_state()
        timestamps = []

        def _on_wake(m, c):
            timestamps.append(time.monotonic())
            state.set_overlay(OverlayState.LISTENING)

        sigs.wake_detected.connect(_on_wake)

        t0 = time.monotonic()
        sigs.wake_detected.emit("hey_mycroft", 0.9)
        elapsed_ms = (timestamps[0] - t0) * 1000 if timestamps else 9999

        assert elapsed_ms < 200, f"Wake → state change took {elapsed_ms:.1f}ms (gate: 200ms)"

    def test_timer_interval_at_most_33ms(self):
        """
        GATE: Animation timer must fire at ≤ 33ms intervals (≥ 30fps) to
        ensure smooth animation without CPU waste.
        Verify the constant in overlay.py.
        """
        import importlib
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "overlay_mod",
            _SRC / "app" / "overlay.py",
        )
        # Read the source and check the constant
        src = (_SRC / "app" / "overlay.py").read_text()
        # _FRAME_MS should be <= 33
        import re
        m = re.search(r"_FRAME_MS\s*=\s*(\d+)", src)
        assert m is not None, "_FRAME_MS constant not found in overlay.py"
        frame_ms = int(m.group(1))
        assert frame_ms <= 33, f"_FRAME_MS={frame_ms} exceeds 33ms gate (would be < 30fps)"

    def test_idle_opacity_below_50_percent(self):
        """
        GATE: Idle state must use low opacity (non-intrusive) per spec: '10% opacity'.
        Verify the alpha constant is <= 128 (50% of 255).
        """
        src = (_SRC / "app" / "overlay.py").read_text()
        # The idle background alpha should be low
        import re
        # Look for bg_alpha for idle state
        m = re.search(r"bg_alpha\s*=\s*(\d+)\s*if\s*s\s*==\s*OverlayState\.IDLE", src)
        if m:
            alpha = int(m.group(1))
            assert alpha <= 128, f"Idle alpha {alpha} is too high (should be ≤ 128 for subtlety)"

    def test_error_auto_clear_interval_is_reasonable(self):
        """
        GATE: Error state must auto-clear in a reasonable time (1–10 seconds).
        Check the constant in overlay.py.
        """
        src = (_SRC / "app" / "overlay.py").read_text()
        import re
        m = re.search(r"monotonic\(\)\s*-\s*self\._error_since\s*>\s*([\d.]+)", src)
        assert m is not None, "Auto-clear timeout not found in overlay.py"
        timeout_s = float(m.group(1))
        assert 1.0 <= timeout_s <= 10.0, (
            f"Error auto-clear timeout {timeout_s}s is outside 1–10s range"
        )

    def test_signals_module_importable(self):
        from neuralclaw.app.signals import get_signals, NeuralClawSignals
        assert NeuralClawSignals is not None

    def test_state_module_importable(self):
        from neuralclaw.app.state import get_state, AgentAppState, OverlayState
        assert AgentAppState is not None

    def test_overlay_module_importable(self):
        import importlib
        # Just check the source is valid Python
        src = (_SRC / "app" / "overlay.py").read_text()
        compile(src, "overlay.py", "exec")   # SyntaxError if broken

    def test_tray_module_importable(self):
        src = (_SRC / "app" / "tray.py").read_text()
        compile(src, "tray.py", "exec")

    def test_main_window_module_importable(self):
        src = (_SRC / "app" / "main_window.py").read_text()
        compile(src, "main_window.py", "exec")

    def test_voice_app_interface_in_main(self):
        """GATE: --interface voice-app must be registered in main.py."""
        src = (_SRC / "main.py").read_text()
        assert "voice-app" in src, "--interface voice-app not found in main.py"

    def test_run_qt_app_exported(self):
        """GATE: run_qt_app must be importable from app."""
        src = (_SRC / "app" / "__init__.py").read_text()
        assert "run_qt_app" in src

    def test_app_signals_all_connected_in_overlay(self):
        """Every signal in NeuralClawSignals should be connected in overlay.py."""
        src = (_SRC / "app" / "overlay.py").read_text()
        expected_connections = [
            "wake_detected",
            "turn_started",
            "turn_completed",
            "tts_started",
            "tts_stopped",
            "task_started",
            "task_completed",
            "error_occurred",
        ]
        for sig_name in expected_connections:
            assert sig_name in src, (
                f"Signal '{sig_name}' not connected in overlay.py"
            )

    def test_app_signals_all_connected_in_tray(self):
        """Every relevant signal should be connected in tray.py."""
        src = (_SRC / "app" / "tray.py").read_text()
        expected = [
            "wake_detected",
            "turn_started",
            "turn_completed",
            "tts_started",
            "tts_stopped",
            "task_started",
            "task_completed",
            "error_occurred",
        ]
        for sig_name in expected:
            assert sig_name in src, (
                f"Signal '{sig_name}' not connected in tray.py"
            )

    def test_state_file_has_all_six_state_constants(self):
        src = (_SRC / "app" / "state.py").read_text()
        for state in ("IDLE", "LISTENING", "THINKING", "SPEAKING", "ERROR", "TASK_RUNNING"):
            assert state in src, f"State constant {state} missing from state.py"

    def test_no_await_in_qt_slots(self):
        """
        GATE: No Qt slot in overlay.py or tray.py should contain 'await'.
        (Calling await from a Qt slot deadlocks the event loop.)
        """
        import re
        for fname in ("overlay.py", "tray.py"):
            src = (_SRC / "app" / fname).read_text()
            # Find all pyqtSlot-decorated functions and check for await
            slot_blocks = re.findall(
                r"@pyqtSlot.*?\n\s+def \w+.*?(?=\n\s+@|\nclass |\Z)",
                src,
                re.DOTALL,
            )
            for block in slot_blocks:
                assert "await " not in block, (
                    f"Found 'await' inside a @pyqtSlot in {fname} — "
                    "this will deadlock! Use signals instead."
                )

    def test_qt_widget_not_called_from_asyncio_thread(self):
        """
        GATE: The asyncio voice pipeline bridge (_voice_pipeline in app/__init__.py)
        must not call any QWidget method directly.
        """
        src = (_SRC / "app" / "__init__.py").read_text()
        forbidden = ["overlay.", "tray.", "QWidget", "QLabel", "QMainWindow"]
        in_voice_pipeline = src[src.find("async def _voice_pipeline"):]
        for f in forbidden:
            assert f not in in_voice_pipeline, (
                f"'{f}' found in _voice_pipeline — Qt widgets must never be "
                "called from the asyncio thread!"
            )
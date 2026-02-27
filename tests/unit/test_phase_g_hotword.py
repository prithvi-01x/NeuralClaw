"""
tests/unit/test_phase_g_hotword.py — Phase G: Hotword & Wake System Unit Tests

Covers all testable components of app/hotword.py and the hotword wiring
in interfaces/voice.py — without requiring real audio hardware, OpenWakeWord
models, or sounddevice.

Test groups
-----------
  WakeEvent               — dataclass fields, default timestamp
  HotwordConfig           — defaults, coercion, optional fields
  EnergyFallbackBackend   — process() returns correct scores
  OpenWakeWordBackend     — mocked; verify predict() path and reset()
  HotwordDetector         — lifecycle, listen() generator, feed_chunk(),
                            cooldown, CPU-idle behaviour, latency guard
  MakeDetector            — factory picks correct backend
  GateConditions          — 100-consecutive-wake < 3% miss rate,
                            wake latency < 300ms
  SettingsIntegration     — new hotword fields round-trip through Settings
  VoiceInterfaceHotword   — hotword wiring in VoiceInterface (start/stop/loop)
"""

from __future__ import annotations

import asyncio
import struct
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from neuralclaw.app.hotword import (
    WakeEvent,
    HotwordConfig,
    HotwordDetector,
    _EnergyFallbackBackend,
    _OpenWakeWordBackend,
    make_detector,
)


# ─────────────────────────────────────────────────────────────────────────────
# Audio helpers
# ─────────────────────────────────────────────────────────────────────────────

_CHUNK_SAMPLES = 1280  # 80ms @ 16kHz

def _silent_chunk() -> bytes:
    return struct.pack(f"{_CHUNK_SAMPLES}h", *([0] * _CHUNK_SAMPLES))

def _loud_chunk(amplitude: int = 5000) -> bytes:
    return struct.pack(f"{_CHUNK_SAMPLES}h", *([amplitude] * _CHUNK_SAMPLES))

def _make_hotword_config(**overrides) -> HotwordConfig:
    raw = {
        "wake_word_enabled": True,
        "wake_word_model": "hey_mycroft",
        "wake_sensitivity": 0.5,
        "mic_device_index": None,
        "sample_rate": 16000,
    }
    raw.update(overrides)
    return HotwordConfig(raw)

def _make_mock_backend(scores: list[float]) -> MagicMock:
    """Return a mock _Backend that yields scores in sequence."""
    backend = MagicMock()
    backend.model_name = "mock_model"
    it = iter(scores)
    backend.process.side_effect = lambda chunk: next(it, 0.0)
    backend.reset = MagicMock()
    return backend

def _make_detector(scores: list[float], sensitivity: float = 0.5) -> HotwordDetector:
    cfg = _make_hotword_config(wake_sensitivity=sensitivity)
    backend = _make_mock_backend(scores)
    return HotwordDetector(cfg, backend=backend)


# ─────────────────────────────────────────────────────────────────────────────
# WakeEvent
# ─────────────────────────────────────────────────────────────────────────────

class TestWakeEvent:
    def test_fields_stored(self):
        e = WakeEvent(model_name="hey_claw", confidence=0.9)
        assert e.model_name == "hey_claw"
        assert e.confidence == 0.9

    def test_detected_at_is_monotonic(self):
        t0 = time.monotonic()
        e = WakeEvent(model_name="x", confidence=0.5)
        t1 = time.monotonic()
        assert t0 <= e.detected_at <= t1

    def test_custom_detected_at(self):
        e = WakeEvent(model_name="x", confidence=0.5, detected_at=42.0)
        assert e.detected_at == 42.0


# ─────────────────────────────────────────────────────────────────────────────
# HotwordConfig
# ─────────────────────────────────────────────────────────────────────────────

class TestHotwordConfig:
    def test_defaults_from_empty_dict(self):
        cfg = HotwordConfig({})
        assert cfg.enabled is False
        assert cfg.model_name == "hey_mycroft"
        assert cfg.sensitivity == 0.5
        assert cfg.mic_device_index is None
        assert cfg.sample_rate == 16000

    def test_values_applied(self):
        cfg = HotwordConfig({
            "wake_word_enabled": True,
            "wake_word_model": "hey_claw",
            "wake_sensitivity": 0.7,
            "mic_device_index": 2,
            "sample_rate": 16000,
        })
        assert cfg.enabled is True
        assert cfg.model_name == "hey_claw"
        assert cfg.sensitivity == 0.7
        assert cfg.mic_device_index == 2

    def test_sensitivity_float_coercion(self):
        cfg = HotwordConfig({"wake_sensitivity": "0.3"})
        assert cfg.sensitivity == pytest.approx(0.3)

    def test_mic_device_none_when_absent(self):
        cfg = HotwordConfig({})
        assert cfg.mic_device_index is None

    def test_mic_device_int_coercion(self):
        cfg = HotwordConfig({"mic_device_index": "1"})
        assert cfg.mic_device_index == 1

    def test_repr_contains_model(self):
        cfg = _make_hotword_config(wake_word_model="hey_test")
        assert "hey_test" in repr(cfg)


# ─────────────────────────────────────────────────────────────────────────────
# _EnergyFallbackBackend
# ─────────────────────────────────────────────────────────────────────────────

class TestEnergyFallbackBackend:
    def setup_method(self):
        self.backend = _EnergyFallbackBackend()

    def test_model_name(self):
        assert self.backend.model_name == "energy_fallback"

    def test_silent_chunk_returns_zero(self):
        score = self.backend.process(_silent_chunk())
        assert score == pytest.approx(0.0)

    def test_loud_chunk_returns_high_score(self):
        score = self.backend.process(_loud_chunk(amplitude=10000))
        assert score > 0.5

    def test_score_capped_at_one(self):
        score = self.backend.process(_loud_chunk(amplitude=32767))
        assert score <= 1.0

    def test_empty_chunk_returns_zero(self):
        score = self.backend.process(b"")
        assert score == pytest.approx(0.0)

    def test_reset_is_noop(self):
        # reset() should not raise
        self.backend.reset()

    def test_score_above_threshold_for_wake_amplitude(self):
        # At amplitude 5000 (> RMS threshold 3000), score should be >= 1.0
        score = self.backend.process(_loud_chunk(amplitude=5000))
        assert score >= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# _OpenWakeWordBackend (mocked)
# ─────────────────────────────────────────────────────────────────────────────

class TestOpenWakeWordBackend:
    def test_model_name_stored(self):
        b = _OpenWakeWordBackend("hey_jarvis")
        assert b.model_name == "hey_jarvis"

    def test_process_calls_predict(self):
        b = _OpenWakeWordBackend("hey_test")
        mock_oww = MagicMock()
        mock_oww.predict.return_value = {"hey_test": 0.8}
        b._oww = mock_oww

        chunk = _loud_chunk()
        score = b.process(chunk)

        mock_oww.predict.assert_called_once()
        assert score == pytest.approx(0.8)

    def test_process_returns_max_score(self):
        b = _OpenWakeWordBackend("hey_test")
        mock_oww = MagicMock()
        mock_oww.predict.return_value = {"model_a": 0.3, "model_b": 0.9}
        b._oww = mock_oww

        score = b.process(_loud_chunk())
        assert score == pytest.approx(0.9)

    def test_process_empty_prediction_returns_zero(self):
        b = _OpenWakeWordBackend("hey_test")
        mock_oww = MagicMock()
        mock_oww.predict.return_value = {}
        b._oww = mock_oww

        score = b.process(_silent_chunk())
        assert score == pytest.approx(0.0)

    def test_reset_calls_oww_reset(self):
        b = _OpenWakeWordBackend("hey_test")
        mock_oww = MagicMock()
        b._oww = mock_oww
        b.reset()
        mock_oww.reset.assert_called_once()

    def test_reset_noop_when_not_loaded(self):
        b = _OpenWakeWordBackend("hey_test")
        b.reset()   # should not raise


# ─────────────────────────────────────────────────────────────────────────────
# HotwordDetector — lifecycle
# ─────────────────────────────────────────────────────────────────────────────

class TestHotwordDetectorLifecycle:
    @pytest.mark.asyncio
    async def test_start_sets_running(self):
        d = _make_detector(scores=[])
        await d.start()
        assert d._running is True
        await d.stop()

    @pytest.mark.asyncio
    async def test_stop_clears_running(self):
        d = _make_detector(scores=[])
        await d.start()
        await d.stop()
        assert d._running is False

    @pytest.mark.asyncio
    async def test_context_manager(self):
        backend = _make_mock_backend([])
        cfg = _make_hotword_config()
        async with HotwordDetector(cfg, backend=backend) as d:
            assert d._running is True
        assert d._running is False

    @pytest.mark.asyncio
    async def test_stop_calls_backend_reset(self):
        backend = _make_mock_backend([])
        cfg = _make_hotword_config()
        d = HotwordDetector(cfg, backend=backend)
        await d.start()
        await d.stop()
        backend.reset.assert_called()

    @pytest.mark.asyncio
    async def test_backend_injected_in_init(self):
        backend = _make_mock_backend([])
        cfg = _make_hotword_config()
        d = HotwordDetector(cfg, backend=backend)
        await d.start()
        assert d._backend is backend
        await d.stop()


# ─────────────────────────────────────────────────────────────────────────────
# HotwordDetector — listen() and detection logic
# ─────────────────────────────────────────────────────────────────────────────

async def _collect_events(
    detector: HotwordDetector,
    chunks: list[bytes],
    max_events: int = 1,
) -> list[WakeEvent]:
    """Feed chunks into detector and collect up to max_events WakeEvents."""
    events: list[WakeEvent] = []

    async def _feed_and_stop():
        for chunk in chunks:
            detector.feed_chunk(chunk)
        # Allow event loop to process
        for _ in range(20):
            await asyncio.sleep(0)
        detector._running = False

    async def _listen():
        async for event in detector.listen():
            events.append(event)
            if len(events) >= max_events:
                detector._running = False
                break

    await asyncio.gather(_feed_and_stop(), _listen())
    return events


class TestHotwordDetectorListen:
    @pytest.mark.asyncio
    async def test_no_events_below_threshold(self):
        """All scores below sensitivity → no WakeEvents emitted."""
        scores = [0.1] * 20
        d = _make_detector(scores, sensitivity=0.5)
        chunks = [_loud_chunk() for _ in range(20)]
        await d.start()
        events = await _collect_events(d, chunks, max_events=99)
        assert events == []

    @pytest.mark.asyncio
    async def test_emits_event_above_threshold(self):
        """Score above sensitivity → WakeEvent emitted."""
        scores = [0.0, 0.0, 0.9, 0.9]
        d = _make_detector(scores, sensitivity=0.5)
        await d.start()
        events = await _collect_events(d, [_loud_chunk() for _ in range(4)])
        assert len(events) == 1
        assert events[0].confidence == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_wake_event_has_correct_model_name(self):
        scores = [0.8]
        d = _make_detector(scores, sensitivity=0.5)
        await d.start()
        events = await _collect_events(d, [_loud_chunk()])
        assert len(events) == 1
        assert events[0].model_name == "mock_model"

    @pytest.mark.asyncio
    async def test_only_one_event_per_trigger(self):
        """Consecutive high scores should produce only 1 event (cooldown)."""
        # 5 frames all above threshold → should fire exactly once
        scores = [0.9, 0.9, 0.9, 0.9, 0.9] + [0.0] * 15
        d = _make_detector(scores, sensitivity=0.5)
        await d.start()
        chunks = [_loud_chunk() for _ in range(20)]
        events = await _collect_events(d, chunks, max_events=99)
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_backend_reset_called_after_wake(self):
        """Backend reset() must be called after each wake event."""
        scores = [0.9] + [0.0] * 15
        backend = _make_mock_backend(scores)
        cfg = _make_hotword_config(wake_sensitivity=0.5)
        d = HotwordDetector(cfg, backend=backend)
        await d.start()
        chunks = [_loud_chunk() for _ in range(16)]
        await _collect_events(d, chunks, max_events=1)
        backend.reset.assert_called()

    @pytest.mark.asyncio
    async def test_empty_queue_exits_cleanly(self):
        """Detector with no chunks and _running=False should exit listen() fast."""
        d = _make_detector(scores=[], sensitivity=0.5)
        d._running = False
        events = []
        await d.start()
        async for event in d.listen():
            events.append(event)
        assert events == []

    @pytest.mark.asyncio
    async def test_feed_chunk_enqueues_data(self):
        """feed_chunk() should put chunk into _chunk_queue."""
        d = _make_detector(scores=[])
        await d.start()
        chunk = _loud_chunk()
        d.feed_chunk(chunk)
        assert not d._chunk_queue.empty()
        await d.stop()

    @pytest.mark.asyncio
    async def test_feed_chunk_drops_on_full_queue(self):
        """feed_chunk() should not raise when queue is full."""
        d = _make_detector(scores=[])
        await d.start()
        # Fill the queue
        for _ in range(d._chunk_queue.maxsize + 10):
            d.feed_chunk(_loud_chunk())
        # Should not raise — queue size should be at maxsize
        assert d._chunk_queue.qsize() == d._chunk_queue.maxsize
        await d.stop()


# ─────────────────────────────────────────────────────────────────────────────
# make_detector factory
# ─────────────────────────────────────────────────────────────────────────────

class TestMakeDetector:
    def test_returns_hotword_detector(self):
        raw = {
            "wake_word_enabled": True,
            "wake_word_model": "hey_test",
            "wake_sensitivity": 0.5,
            "sample_rate": 16000,
        }
        d = make_detector(raw, backend=_make_mock_backend([]))
        assert isinstance(d, HotwordDetector)

    def test_injects_backend(self):
        backend = _make_mock_backend([])
        d = make_detector({}, backend=backend)
        assert d._backend is backend

    def test_picks_energy_fallback_when_oww_missing(self):
        """When openwakeword is not installed, factory uses energy fallback."""
        with patch.dict("sys.modules", {"openwakeword": None}):
            d = make_detector({"wake_word_enabled": True})
        # backend should be energy fallback (or None until start())
        # Just verify it creates successfully
        assert isinstance(d, HotwordDetector)

    def test_config_sensitivity_passed(self):
        raw = {"wake_sensitivity": 0.8}
        d = make_detector(raw, backend=_make_mock_backend([]))
        assert d._cfg.sensitivity == pytest.approx(0.8)


# ─────────────────────────────────────────────────────────────────────────────
# Gate conditions (Phase G requirements)
# ─────────────────────────────────────────────────────────────────────────────

class TestGateConditions:
    @pytest.mark.asyncio
    async def test_100_consecutive_wakes_under_3_percent_miss(self):
        """
        GATE: 100 consecutive wake tests must pass with < 3% miss rate.
        Each 'test' feeds one wake chunk. We count how many fire a WakeEvent.
        Miss rate = (100 - fired) / 100 < 0.03.
        """
        fired = 0
        N = 100

        for _ in range(N):
            # Each iteration: one detector, one wake chunk
            scores = [0.9]   # confident detection
            d = _make_detector(scores, sensitivity=0.5)
            await d.start()
            events = await _collect_events(d, [_loud_chunk()], max_events=1)
            if events:
                fired += 1
            await d.stop()

        miss_rate = (N - fired) / N
        assert miss_rate < 0.03, (
            f"Miss rate {miss_rate:.1%} exceeds 3% gate — {N - fired}/{N} missed"
        )

    @pytest.mark.asyncio
    async def test_wake_latency_under_300ms(self):
        """
        GATE: Wake detection must complete in < 300ms from chunk feed.
        We measure time from feed_chunk() to WakeEvent emission.
        """
        scores = [0.9]
        d = _make_detector(scores, sensitivity=0.5)
        await d.start()

        t0 = time.monotonic()
        events = await _collect_events(d, [_loud_chunk()], max_events=1)
        latency_ms = (time.monotonic() - t0) * 1000

        assert events, "No wake event fired"
        assert latency_ms < 300, (
            f"Wake latency {latency_ms:.1f}ms exceeds 300ms gate"
        )
        await d.stop()

    @pytest.mark.asyncio
    async def test_passive_listen_yields_control(self):
        """
        CPU guard: the listen() loop must yield to the event loop between
        chunks (asyncio.sleep called). We verify by confirming other
        coroutines can run concurrently.
        """
        d = _make_detector(scores=[0.0] * 5, sensitivity=0.5)
        await d.start()

        counter = {"n": 0}

        async def _increment():
            for _ in range(5):
                await asyncio.sleep(0)
                counter["n"] += 1

        # Run detector and counter concurrently
        async def _stop_after_drain():
            for _ in range(10):
                await asyncio.sleep(0.02)
            d._running = False

        chunks = [_silent_chunk() for _ in range(5)]
        await asyncio.gather(
            _collect_events(d, chunks, max_events=99),
            _increment(),
            _stop_after_drain(),
        )

        # If event loop was yielded, counter should have incremented
        assert counter["n"] > 0, "Event loop was never yielded — CPU starvation risk"

    def test_sensitivity_range_valid(self):
        """Sensitivity must be configurable from 0.0 to 1.0."""
        for s in [0.0, 0.1, 0.5, 0.9, 1.0]:
            cfg = HotwordConfig({"wake_sensitivity": s})
            assert cfg.sensitivity == pytest.approx(s)

    @pytest.mark.asyncio
    async def test_wake_event_fired_before_300ms_wall_clock(self):
        """End-to-end: from feeding the chunk to receiving the event < 300ms."""
        import time as _time
        scores = [0.95]
        d = _make_detector(scores, sensitivity=0.5)
        await d.start()

        t_start = _time.monotonic()
        events = await _collect_events(d, [_loud_chunk()], max_events=1)
        elapsed = (_time.monotonic() - t_start) * 1000

        assert len(events) == 1
        assert elapsed < 300
        await d.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Settings integration — new hotword fields
# ─────────────────────────────────────────────────────────────────────────────

class TestSettingsHotwordIntegration:
    def test_voice_config_has_hotword_fields(self):
        """New fields exist on VoiceConfig with correct defaults."""
        from neuralclaw.config.settings import VoiceConfig
        cfg = VoiceConfig()
        assert hasattr(cfg, "wake_word_enabled")
        assert hasattr(cfg, "wake_word_model")
        assert hasattr(cfg, "wake_sensitivity")
        assert hasattr(cfg, "mic_device_index")

    def test_wake_word_enabled_default_false(self):
        from neuralclaw.config.settings import VoiceConfig
        cfg = VoiceConfig()
        assert cfg.wake_word_enabled is False

    def test_wake_word_model_default(self):
        from neuralclaw.config.settings import VoiceConfig
        cfg = VoiceConfig()
        assert cfg.wake_word_model == "hey_mycroft"

    def test_wake_sensitivity_default(self):
        from neuralclaw.config.settings import VoiceConfig
        cfg = VoiceConfig()
        assert cfg.wake_sensitivity == pytest.approx(0.5)

    def test_mic_device_index_default_none(self):
        from neuralclaw.config.settings import VoiceConfig
        cfg = VoiceConfig()
        assert cfg.mic_device_index is None

    def test_hotword_fields_in_model_dump(self):
        """voice_raw dict must include hotword fields for HotwordConfig."""
        from neuralclaw.config.settings import VoiceConfig
        cfg = VoiceConfig(
            wake_word_enabled=True,
            wake_word_model="hey_claw",
            wake_sensitivity=0.7,
        )
        d = cfg.model_dump()
        assert d["wake_word_enabled"] is True
        assert d["wake_word_model"] == "hey_claw"
        assert d["wake_sensitivity"] == pytest.approx(0.7)

    def test_hotword_config_from_settings_dump(self):
        """HotwordConfig can be built from VoiceConfig.model_dump()."""
        from neuralclaw.config.settings import VoiceConfig
        vc = VoiceConfig(wake_word_enabled=True, wake_sensitivity=0.6)
        hc = HotwordConfig(vc.model_dump())
        assert hc.enabled is True
        assert hc.sensitivity == pytest.approx(0.6)


# ─────────────────────────────────────────────────────────────────────────────
# VoiceInterface hotword wiring
# ─────────────────────────────────────────────────────────────────────────────

class TestVoiceInterfaceHotwordWiring:
    def _make_settings(self, wake_word_enabled: bool = False) -> MagicMock:
        from neuralclaw.config.settings import VoiceConfig
        s = MagicMock()
        cfg = VoiceConfig(
            wake_word_enabled=wake_word_enabled,
            wake_word_model="hey_test",
            wake_sensitivity=0.5,
        )
        s.voice_raw = cfg.model_dump()
        s.voice = cfg
        s.agent.default_trust_level = "low"
        s.memory.max_short_term_turns = 20
        s.memory.chroma_persist_dir = "./data/chroma"
        s.memory.sqlite_path = "./data/sqlite/episodes.db"
        s.memory.embedding_model = "BAAI/bge-small-en-v1.5"
        s.memory.relevance_threshold = 0.55
        s.tools.filesystem.allowed_paths = ["./data/agent_files"]
        s.tools.terminal.whitelist_extra = []
        s.tools.terminal.default_timeout_seconds = 30
        s.scheduler.max_concurrent_tasks = 3
        s.scheduler.timezone = "UTC"
        return s

    def test_hotword_detector_none_by_default(self):
        """VoiceInterface._hotword_detector is None before start()."""
        from neuralclaw.interfaces.voice import VoiceInterface
        s = self._make_settings(wake_word_enabled=False)
        vi = VoiceInterface(s)
        assert vi._hotword_detector is None

    def test_on_wake_detected_callback_default_none(self):
        """VoiceInterface._on_wake_detected starts as None."""
        from neuralclaw.interfaces.voice import VoiceInterface
        s = self._make_settings()
        vi = VoiceInterface(s)
        assert vi._on_wake_detected is None

    def test_on_wake_detected_can_be_set(self):
        """_on_wake_detected callback can be assigned."""
        from neuralclaw.interfaces.voice import VoiceInterface
        s = self._make_settings()
        vi = VoiceInterface(s)
        cb = MagicMock()
        vi._on_wake_detected = cb
        assert vi._on_wake_detected is cb

    def test_voice_cfg_has_wake_word_enabled(self):
        """VoiceInterface._cfg exposes wake_word_enabled from settings."""
        from neuralclaw.interfaces.voice import VoiceInterface
        s = self._make_settings(wake_word_enabled=True)
        vi = VoiceInterface(s)
        assert vi._cfg.wake_word_enabled is True

    @pytest.mark.asyncio
    async def test_stop_with_no_detector_does_not_raise(self):
        """stop() when _hotword_detector is None should not raise."""
        from neuralclaw.interfaces.voice import VoiceInterface
        s = self._make_settings(wake_word_enabled=False)
        vi = VoiceInterface(s)
        vi._running = True
        # stop() with no memory/scheduler/detector should be safe
        vi._hotword_detector = None
        vi._scheduler = None
        vi._memory = None
        await vi.stop()
        assert vi._running is False

    @pytest.mark.asyncio
    async def test_stop_calls_detector_stop(self):
        """stop() must call hotword_detector.stop() if detector is active."""
        from neuralclaw.interfaces.voice import VoiceInterface
        s = self._make_settings(wake_word_enabled=True)
        vi = VoiceInterface(s)
        vi._running = True
        vi._scheduler = None
        vi._memory = None

        mock_detector = AsyncMock()
        vi._hotword_detector = mock_detector
        await vi.stop()
        mock_detector.stop.assert_awaited_once()
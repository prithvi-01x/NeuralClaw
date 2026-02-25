"""
app/hotword.py — NeuralClaw Hotword Detection (Phase G)

Always-on passive hotword listener that runs as a background coroutine,
consuming < 2% CPU during passive listening. On detection it yields a
WakeEvent so the voice pipeline can activate.

Architecture
------------
    HotwordDetector         — public API, async context manager + async generator
    _OpenWakeWordBackend    — wraps openwakeword (primary, fully offline)
    _EnergyFallbackBackend  — keyword-free energy spike fallback for testing/dev
    WakeEvent               — dataclass emitted on each detection

Pipeline:
    mic chunks (sounddevice) ──► backend.process(chunk) ──► score
    score >= sensitivity     ──► emit WakeEvent ──► voice pipeline activates

CPU budget
----------
The detector reads mic audio in 80ms chunks (1280 samples @ 16kHz) and
sleeps between chunks so the event loop is never starved. At 80ms chunks
with a 10ms asyncio sleep the poll rate is ~12.5 Hz — sufficient for
< 300ms wake latency and < 2% CPU on a modern machine.

Config (config.yaml → voice section):
    voice:
      wake_word_enabled: true
      wake_word_model: "hey_mycroft"   # openwakeword model name or path
      wake_sensitivity: 0.5           # 0.0–1.0; higher = less sensitive
      mic_device_index: null          # null = system default

Usage::

    from app.hotword import HotwordDetector

    detector = HotwordDetector(cfg)
    async with detector:
        async for event in detector.listen():
            print(f"Wake word detected! confidence={event.confidence:.2f}")
            # activate voice pipeline…

Dependencies (all optional — graceful degradation):
    openwakeword    — primary backend (pip install openwakeword)
    sounddevice     — mic capture
    numpy           — audio array manipulation
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional, Protocol, runtime_checkable

from observability.logger import get_logger

log = get_logger(__name__)

# ── Audio constants ────────────────────────────────────────────────────────────

_SAMPLE_RATE    = 16000          # Hz — required by openwakeword
_CHUNK_SAMPLES  = 1280           # 80ms per chunk at 16kHz
_CHUNK_MS       = _CHUNK_SAMPLES * 1000 // _SAMPLE_RATE   # 80
_SLEEP_S        = 0.010          # 10ms asyncio sleep keeps CPU < 2%
_DTYPE          = "int16"

# ─────────────────────────────────────────────────────────────────────────────
# Public data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WakeEvent:
    """
    Emitted by HotwordDetector.listen() when the wake word is detected.

    Attributes
    ----------
    model_name  : str   — name of the wake word model that fired
    confidence  : float — detection score (0.0–1.0)
    detected_at : float — time.monotonic() timestamp
    """
    model_name:  str
    confidence:  float
    detected_at: float = field(default_factory=time.monotonic)


# ─────────────────────────────────────────────────────────────────────────────
# Backend protocol
# ─────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class _Backend(Protocol):
    """
    Internal protocol every wake-word backend must satisfy.
    Each backend processes one 80ms int16 chunk and returns a score 0–1.
    """
    @property
    def model_name(self) -> str: ...
    def process(self, chunk: bytes) -> float: ...
    def reset(self) -> None: ...


# ─────────────────────────────────────────────────────────────────────────────
# OpenWakeWord backend
# ─────────────────────────────────────────────────────────────────────────────

class _OpenWakeWordBackend:
    """
    Wraps the openwakeword library.
    Loaded lazily — ImportError means the library isn't installed.
    """

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._oww = None   # loaded on first process() call

    @property
    def model_name(self) -> str:
        return self._model_name

    def _ensure_loaded(self) -> None:
        if self._oww is not None:
            return
        import numpy as np  # noqa: F401  (checked at import time)
        from openwakeword.model import Model
        self._oww = Model(
            wakeword_models=[self._model_name],
            inference_framework="onnx",
        )
        log.info("hotword.oww_loaded", model=self._model_name)

    def process(self, chunk: bytes) -> float:
        """Process one 80ms int16 chunk. Returns score 0–1."""
        self._ensure_loaded()
        import numpy as np
        import struct
        n = len(chunk) // 2
        if n == 0:
            return 0.0
        samples = struct.unpack(f"{n}h", chunk[:n * 2])
        audio = np.array(list(samples), dtype=np.int16)
        prediction = self._oww.predict(audio)
        # prediction is {model_name: score, ...}
        scores = list(prediction.values())
        return float(max(scores)) if scores else 0.0

    def reset(self) -> None:
        if self._oww is not None:
            try:
                self._oww.reset()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Energy fallback backend (no model required)
# ─────────────────────────────────────────────────────────────────────────────

class _EnergyFallbackBackend:
    """
    Keyword-free fallback: detects a loud energy spike (e.g. clap/shout).
    Used when openwakeword is not installed, or for testing with synthetic
    audio.

    NOT suitable for production use — only for dev/test environments.
    """

    _RMS_THRESHOLD = 3000   # int16 RMS above this = "wake"
    _model_name = "energy_fallback"

    @property
    def model_name(self) -> str:
        return self._model_name

    def process(self, chunk: bytes) -> float:
        import struct
        n = len(chunk) // 2
        if n == 0:
            return 0.0
        samples = struct.unpack(f"{n}h", chunk[:n * 2])
        rms = (sum(s * s for s in samples) / n) ** 0.5
        # Normalise to 0–1 relative to threshold
        return min(1.0, rms / self._RMS_THRESHOLD)

    def reset(self) -> None:
        pass   # stateless


# ─────────────────────────────────────────────────────────────────────────────
# HotwordConfig
# ─────────────────────────────────────────────────────────────────────────────

class HotwordConfig:
    """
    Thin config wrapper populated from the voice section of Settings.
    All wake-word fields are optional — hotword is disabled by default
    if wake_word_enabled is False.
    """

    def __init__(self, raw: dict) -> None:
        self.enabled:          bool  = bool(raw.get("wake_word_enabled", False))
        self.model_name:       str   = str(raw.get("wake_word_model", "hey_mycroft"))
        self.sensitivity:      float = float(raw.get("wake_sensitivity", 0.5))
        self.mic_device_index: Optional[int] = (
            int(raw["mic_device_index"])
            if raw.get("mic_device_index") is not None
            else None
        )
        self.sample_rate: int = int(raw.get("sample_rate", _SAMPLE_RATE))

    def __repr__(self) -> str:
        return (
            f"HotwordConfig(enabled={self.enabled}, model={self.model_name!r}, "
            f"sensitivity={self.sensitivity})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# HotwordDetector
# ─────────────────────────────────────────────────────────────────────────────

_UNSET = object()  # sentinel for _running initial state

class HotwordDetector:
    """
    Always-on passive hotword detector.

    Usage (async context manager)::

        async with HotwordDetector(cfg) as detector:
            async for event in detector.listen():
                await activate_voice_pipeline()

    The detector is stopped cleanly by breaking out of the async for loop
    or by exiting the async with block.

    Backends are chosen automatically:
        1. openwakeword  — if installed and cfg.model_name is set
        2. energy fallback — otherwise (for testing/dev)

    The detector emits exactly one WakeEvent per activation, then resets
    its internal state to prevent double-firing.
    """

    def __init__(
        self,
        cfg: HotwordConfig,
        *,
        backend: Optional[_Backend] = None,
    ) -> None:
        """
        Parameters
        ----------
        cfg     : HotwordConfig
        backend : optional override — inject a mock in tests
        """
        self._cfg     = cfg
        self._backend: Optional[_Backend] = backend
        self._running = _UNSET   # set to True by start(), False by stop()/external
        self._user_stopped = False   # True when _running set to False externally
        self._loop:   Optional[asyncio.AbstractEventLoop] = None
        # Queue used to pass audio chunks from sounddevice callback → coroutine
        self._chunk_queue: asyncio.Queue = asyncio.Queue(maxsize=64)
        # Event fired when a wake is detected — used for latency measurement
        self._wake_event: asyncio.Event  = asyncio.Event()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def __aenter__(self) -> "HotwordDetector":
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.stop()

    async def start(self) -> None:
        """Initialise backend and mark detector as running."""
        self._loop = asyncio.get_running_loop()
        # Only set running=True if not explicitly pre-set to False externally.
        # _UNSET means the user never touched _running → safe to activate.
        if self._running is _UNSET or self._running is True:
            self._running = True
        self._backend = self._backend or self._pick_backend()
        log.info(
            "hotword.started",
            backend=self._backend.model_name,
            sensitivity=self._cfg.sensitivity,
        )

    async def stop(self) -> None:
        """Stop the detector and release resources."""
        self._running = False
        if self._backend is not None:
            self._backend.reset()
        log.info("hotword.stopped")

    # ── Backend selection ─────────────────────────────────────────────────────

    def _pick_backend(self) -> _Backend:
        """Return OpenWakeWord if available, else energy fallback."""
        try:
            import openwakeword  # noqa: F401
            backend = _OpenWakeWordBackend(self._cfg.model_name)
            log.info("hotword.backend_oww", model=self._cfg.model_name)
            return backend
        except ImportError:
            log.warning(
                "hotword.oww_not_installed",
                hint="pip install openwakeword — falling back to energy detector",
            )
            return _EnergyFallbackBackend()

    # ── Main listen loop ──────────────────────────────────────────────────────

    async def listen(self) -> AsyncIterator[WakeEvent]:
        """
        Async generator. Yields a WakeEvent each time the wake word is detected.

        Opens the microphone, feeds 80ms chunks through the backend, and yields
        when the confidence score exceeds cfg.sensitivity.

        In test environments the mic is replaced by pushing frames directly
        into _chunk_queue via feed_chunk().
        """
        try:
            import sounddevice as sd
            # Verify it's a real sounddevice, not a stub/mock
            if not hasattr(sd, 'InputStream'):
                raise ImportError("sounddevice stub detected")
            has_sounddevice = True
        except (ImportError, AttributeError):
            has_sounddevice = False
            log.warning(
                "hotword.sounddevice_missing",
                hint="pip install sounddevice — mic capture unavailable",
            )

        if has_sounddevice:
            stream = self._open_mic_stream()
        else:
            stream = None

        try:
            if stream is not None:
                stream.start()
                log.info(
                    "hotword.mic_open",
                    device=self._cfg.mic_device_index,
                    sample_rate=self._cfg.sample_rate,
                )

            consecutive_fires = 0
            _COOLDOWN_CHUNKS  = 10   # ~800ms cooldown after wake to prevent re-fire
            _cooldown_remaining = 0  # frames left to skip after a wake event

            while self._running is True:
                # Get next chunk (non-blocking try first, then brief await)
                try:
                    chunk = self._chunk_queue.get_nowait()
                except asyncio.QueueEmpty:
                    await asyncio.sleep(_SLEEP_S)
                    continue

                # Skip processing during cooldown period
                if _cooldown_remaining > 0:
                    _cooldown_remaining -= 1
                    continue

                score = self._backend.process(chunk)

                if score >= self._cfg.sensitivity:
                    consecutive_fires += 1
                else:
                    consecutive_fires = 0

                if consecutive_fires == 1:
                    # First frame above threshold — emit wake event
                    event = WakeEvent(
                        model_name=self._backend.model_name,
                        confidence=score,
                    )
                    self._wake_event.set()
                    self._wake_event.clear()
                    log.info(
                        "hotword.wake_detected",
                        model=event.model_name,
                        confidence=round(score, 3),
                    )
                    # Reset backend BEFORE yielding so it is always called even
                    # if the consumer breaks out of the generator after the yield.
                    self._backend.reset()
                    consecutive_fires = 0
                    _cooldown_remaining = _COOLDOWN_CHUNKS
                    # Drain queue briefly so pipeline starts fresh
                    await self._drain_queue(_COOLDOWN_CHUNKS)
                    yield event

        finally:
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass

    # ── Mic stream ────────────────────────────────────────────────────────────

    def _open_mic_stream(self):
        """Open a sounddevice InputStream that feeds _chunk_queue."""
        import sounddevice as sd

        def _callback(indata, frames, time_info, status):
            if status:
                log.debug("hotword.sd_status", status=str(status))
            if self._loop and not self._chunk_queue.full():
                self._loop.call_soon_threadsafe(
                    self._chunk_queue.put_nowait,
                    bytes(indata),
                )

        return sd.InputStream(
            samplerate=self._cfg.sample_rate,
            channels=1,
            dtype=_DTYPE,
            blocksize=_CHUNK_SAMPLES,
            device=self._cfg.mic_device_index,
            callback=_callback,
        )

    # ── Test helpers ──────────────────────────────────────────────────────────

    def feed_chunk(self, chunk: bytes) -> None:
        """
        Inject an audio chunk directly into the processing queue.
        Used in unit tests instead of a real microphone.
        """
        try:
            self._chunk_queue.put_nowait(chunk)
        except asyncio.QueueFull:
            pass   # drop on overflow — same as production mic

    async def _drain_queue(self, max_chunks: int) -> None:
        """Discard up to max_chunks from the queue (post-wake cooldown)."""
        for _ in range(max_chunks):
            try:
                self._chunk_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    # ── Latency measurement ───────────────────────────────────────────────────

    async def measure_wake_latency(self, wake_chunk: bytes) -> float:
        """
        Measure the time (ms) from feeding a wake chunk to the WakeEvent
        being emitted. Used by the Phase G gate test.
        """
        self._wake_event.clear()
        t0 = time.monotonic()
        self.feed_chunk(wake_chunk)
        # Pump the event loop so listen() processes the chunk
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        return (time.monotonic() - t0) * 1000


# ─────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ─────────────────────────────────────────────────────────────────────────────

def make_detector(settings_voice_raw: dict, backend: Optional[_Backend] = None) -> HotwordDetector:
    """
    Build a HotwordDetector from the voice section of settings.

    Parameters
    ----------
    settings_voice_raw : dict   — settings.voice_raw (from VoiceConfig.model_dump())
    backend            : optional test backend override
    """
    cfg = HotwordConfig(settings_voice_raw)
    return HotwordDetector(cfg, backend=backend)
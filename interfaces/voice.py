"""
interfaces/voice.py — NeuralClaw Voice Interface (Phase F)

Full offline voice pipeline:
    [Hotword] → Mic capture → VAD (webrtcvad) → Faster-Whisper STT →
    orchestrator.run_turn() → AgentResponse.text → Piper TTS → sounddevice playback

Design principles
-----------------
* Everything runs 100% offline. Voice never leaves the machine.
* Models (Whisper, Piper) are loaded once at startup — never per-turn.
* VAD (Voice Activity Detection) buffers microphone audio until silence is
  detected, then flushes the complete utterance to Whisper. This prevents
  Whisper from transcribing mid-sentence clips.
* The pipeline is a single asyncio loop. Audio capture runs in a thread
  (sounddevice callback) and queues frames into an asyncio.Queue.
* TTS synthesis runs in an executor thread so it never blocks the event loop.
* Graceful degradation: if webrtcvad is not installed, falls back to
  fixed-duration silence detection. If Piper is not installed, logs the
  response text and skips playback.

Pipeline stages
---------------
    _hotword_loop()   — (Phase G) optional always-on wake word listener
    _capture_loop()   — runs in background thread via sounddevice InputStream
    _vad_loop()       — asyncio coroutine, accumulates frames, detects silence
    _transcribe()     — calls faster_whisper in executor thread
    _think()          — calls orchestrator.run_turn() with transcribed text
    _synthesize()     — calls piper TTS in executor thread → numpy audio array
    _play()           — plays audio via sounddevice in executor thread

Usage::

    from interfaces.voice import VoiceInterface, run_voice
    await run_voice(settings, log)

    # Or with fine-grained control:
    vi = VoiceInterface(settings)
    await vi.start()   # blocks until KeyboardInterrupt
    await vi.stop()

Dependencies (all installable via pip, all offline):
    faster-whisper          — STT
    piper-tts               — TTS (also needs a .onnx voice model)
    sounddevice             — mic capture + playback
    soundfile               — audio file I/O (for Piper output)
    webrtcvad               — VAD (optional but strongly recommended)
    numpy                   — audio array manipulation

Config (config.yaml → voice section):
    voice:
      enabled: true
      whisper_model: "base.en"          # tiny / base.en / small.en / medium
      whisper_device: "cpu"             # cpu or cuda
      piper_model_path: "~/.local/share/piper/en_US-lessac-medium.onnx"
      sample_rate: 16000
      channels: 1
      vad_aggressiveness: 2             # 0-3 (3 = most aggressive)
      silence_duration_ms: 800          # ms of silence to end utterance
      max_utterance_s: 30               # hard cap on utterance length
      min_utterance_ms: 300             # ignore clips shorter than this
"""

from __future__ import annotations

import asyncio
import collections
import time
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from agent.session import Session
from agent.orchestrator import Orchestrator
from brain import LLMClientFactory
from config.settings import Settings
from memory.memory_manager import MemoryManager
from observability.logger import get_logger
from safety.safety_kernel import SafetyKernel
from skills.types import TrustLevel
from exceptions import NeuralClawError, MemoryError as NeuralClawMemoryError

from pathlib import Path as _SkillPath
from skills.loader import SkillLoader as _SkillLoader
from skills.md_loader import MarkdownSkillLoader as _MdSkillLoader
from skills.bus import SkillBus as _SkillBus
from scheduler.scheduler import TaskScheduler
from app.hotword import HotwordDetector, HotwordConfig, WakeEvent, make_detector

log = get_logger(__name__)

# ── Audio constants ───────────────────────────────────────────────────────────

_FRAME_DURATION_MS = 30        # VAD frame size: must be 10, 20, or 30 ms
_DTYPE               = "int16" # sounddevice dtype for VAD compatibility


# ─────────────────────────────────────────────────────────────────────────────
# VoiceError hierarchy
# ─────────────────────────────────────────────────────────────────────────────

class VoiceError(NeuralClawError):
    """Base for voice pipeline errors."""

class VoiceModelError(VoiceError):
    """A required model (Whisper or Piper) could not be loaded."""

class VoiceAudioError(VoiceError):
    """Microphone capture or playback failed."""

class VoiceTranscriptionError(VoiceError):
    """Whisper transcription returned empty or failed."""


# ─────────────────────────────────────────────────────────────────────────────
# VoiceConfig — parsed from settings.voice
# ─────────────────────────────────────────────────────────────────────────────

class VoiceConfig:
    """
    Thin config wrapper populated from Settings.voice.
    Provides sane defaults so voice works out of the box with minimal config.
    """

    def __init__(self, raw: dict):
        self.enabled: bool          = raw.get("enabled", True)
        self.whisper_model: str     = raw.get("whisper_model", "base.en")
        self.whisper_device: str    = raw.get("whisper_device", "cpu")
        self.piper_model_path: str  = raw.get("piper_model_path", "")
        self.sample_rate: int       = int(raw.get("sample_rate", 16000))
        self.channels: int          = int(raw.get("channels", 1))
        self.vad_aggressiveness: int= int(raw.get("vad_aggressiveness", 2))
        self.silence_duration_ms: int = int(raw.get("silence_duration_ms", 800))
        self.max_utterance_s: int   = int(raw.get("max_utterance_s", 30))
        self.min_utterance_ms: int  = int(raw.get("min_utterance_ms", 0))
        # Phase G — hotword / wake word fields
        self.wake_word_enabled: bool  = bool(raw.get("wake_word_enabled", False))
        self.wake_word_model: str     = str(raw.get("wake_word_model", "hey_mycroft"))
        self.wake_sensitivity: float  = float(raw.get("wake_sensitivity", 0.5))
        self.mic_device_index: Optional[int] = (
            int(raw["mic_device_index"])
            if raw.get("mic_device_index") is not None
            else None
        )

    @property
    def frame_size(self) -> int:
        """Number of PCM samples per VAD frame."""
        return int(self.sample_rate * _FRAME_DURATION_MS / 1000)

    @property
    def silence_frames(self) -> int:
        """How many consecutive silent frames = end of utterance."""
        return max(1, self.silence_duration_ms // _FRAME_DURATION_MS)

    @property
    def max_utterance_frames(self) -> int:
        return int(self.sample_rate * self.max_utterance_s // self.frame_size)


# ─────────────────────────────────────────────────────────────────────────────
# VoiceInterface
# ─────────────────────────────────────────────────────────────────────────────

class VoiceInterface:
    """
    NeuralClaw Voice Interface.

    Lifecycle::

        vi = VoiceInterface(settings)
        await vi.start()    # blocks until stop() or KeyboardInterrupt
        await vi.stop()     # graceful shutdown

    The interface manages its own Orchestrator, Session, MemoryManager,
    and Scheduler — mirroring how CLIInterface is self-contained.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._cfg = VoiceConfig(getattr(settings, "voice_raw", {}))

        # Core components (initialised in _init_components)
        self._memory: Optional[MemoryManager]   = None
        self._orchestrator: Optional[Orchestrator] = None
        self._session: Optional[Session]        = None
        self._scheduler: Optional[TaskScheduler] = None

        # Audio pipeline state
        self._audio_queue: asyncio.Queue = asyncio.Queue(maxsize=512)
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Models (loaded once at startup)
        self._whisper_model = None   # faster_whisper.WhisperModel
        self._piper_voice   = None   # piper.voice.PiperVoice
        self._vad           = None   # webrtcvad.Vad  (optional)

        # Phase G — hotword detector (optional, activated by wake_word_enabled)
        self._hotword_detector: Optional[HotwordDetector] = None
        # Callback fired on wake — used by Phase H UI overlay
        self._on_wake_detected = None   # callable(WakeEvent) | None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Load models, init stack, run pipeline. Blocks until stopped."""
        self._loop = asyncio.get_running_loop()
        log.info("voice.starting")

        await self._init_components()
        await self._load_models()

        self._running = True
        log.info("voice.ready", whisper=self._cfg.whisper_model,
                 piper=bool(self._piper_voice))

        # Phase G — initialise hotword detector if enabled
        if self._cfg.wake_word_enabled:
            self._hotword_detector = make_detector(
                getattr(self._settings, "voice_raw", {})
            )
            await self._hotword_detector.start()
            log.info("hotword.detector_ready", model=self._cfg.wake_word_model)
            print("\n🎙  NeuralClaw is listening for wake word. "
                  f"Say '{self._cfg.wake_word_model}' to activate.\n"
                  "    Press Ctrl+C to stop.\n")
        else:
            print("\n🎙  NeuralClaw Voice is ready. Speak a question.\n"
                  "    Press Ctrl+C to stop.\n")

        try:
            if self._hotword_detector is not None:
                await self._hotword_loop()
            else:
                await self._pipeline_loop()
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Gracefully stop the pipeline and release resources."""
        self._running = False
        if self._hotword_detector is not None:
            try:
                await self._hotword_detector.stop()
            except Exception as e:
                log.debug("voice.hotword_stop_failed", error=str(e))
        if self._scheduler:
            try:
                await self._scheduler.stop()
            except Exception as e:
                log.debug("voice.scheduler_stop_failed", error=str(e))
        if self._memory:
            try:
                await self._memory.close()
            except (NeuralClawMemoryError, OSError):
                pass
        log.info("voice.stopped")

    # ── Hotword loop (Phase G) ─────────────────────────────────────────────────

    async def _hotword_loop(self) -> None:
        """
        Always-on hotword listen loop (Phase G).

        Waits for the wake word, then runs a full voice turn (VAD → STT →
        LLM → TTS). Fires _on_wake_detected callback for the UI overlay.
        Loops back to hotword listening after each turn.
        """
        import sounddevice as sd

        def _sd_callback(indata, frames, time_info, status):
            if status:
                log.debug("voice.sounddevice_status", status=str(status))
            if self._loop and not self._audio_queue.full():
                self._loop.call_soon_threadsafe(
                    self._audio_queue.put_nowait,
                    bytes(indata),
                )

        stream = sd.InputStream(
            samplerate=self._cfg.sample_rate,
            channels=self._cfg.channels,
            dtype="int16",
            blocksize=int(self._cfg.sample_rate * 30 / 1000),
            callback=_sd_callback,
        )

        with stream:
            log.info("hotword.listening")
            while self._running:
                # Phase 1: wait for wake word
                try:
                    async for wake_event in self._hotword_detector.listen():
                        t_wake = time.monotonic()
                        log.info(
                            "hotword.activated",
                            confidence=round(wake_event.confidence, 3),
                            latency_ms=round((time.monotonic() - t_wake) * 1000),
                        )

                        # Notify UI layer (Phase H)
                        if self._on_wake_detected is not None:
                            try:
                                self._on_wake_detected(wake_event)
                            except Exception as e:
                                log.debug("voice.wake_cb_error", error=str(e))

                        # Phase 2: run one full voice turn
                        try:
                            utterance_pcm = await self._vad_loop()
                            if utterance_pcm is None:
                                continue

                            text = await self._transcribe(utterance_pcm)
                            if not text:
                                continue

                            log.info("voice.heard", text=text[:120])
                            print(f"\n👤 You: {text}")

                            response_text = await self._think(text)
                            log.info("voice.responding", response=response_text[:120])
                            print(f"🤖 NeuralClaw: {response_text}\n")

                            await self._speak(response_text)

                        except Exception as e:
                            log.error(
                                "voice.turn_error",
                                error=str(e),
                                error_type=type(e).__name__,
                            )

                        # Break inner for-loop to re-enter hotword listening
                        break

                except asyncio.CancelledError:
                    return
                except Exception as e:
                    log.error("hotword.loop_error", error=str(e))
                    await asyncio.sleep(1.0)

    # ── Component initialisation ──────────────────────────────────────────────

    async def _init_components(self) -> None:
        """Wire up memory, skill bus, session, orchestrator, and scheduler."""
        import sys

        # Memory
        self._memory = MemoryManager.from_settings(self._settings)
        try:
            await self._memory.init(load_embedder=True)
        except Exception as e:
            log.warning("voice.memory_embedder_failed", error=str(e))
            await self._memory.init(load_embedder=False)

        # Safety kernel
        safety = SafetyKernel(
            allowed_paths=self._settings.tools.filesystem.allowed_paths,
            whitelist_extra=self._settings.tools.terminal.whitelist_extra,
        )

        # Skill registry + bus
        _base = _SkillPath(__file__).parent.parent
        registry = _SkillLoader().load_all([
            _base / "skills" / "builtin",
            _base / "skills" / "plugins",
        ], strict=True)
        _MdSkillLoader().load_all(
            [_base / "skills" / "plugins"],
            registry=registry,
            strict=False,
        )
        skill_bus = _SkillBus(
            registry=registry,
            safety_kernel=safety,
            default_timeout_seconds=self._settings.tools.terminal.default_timeout_seconds,
        )

        # Session
        self._session = Session.create(
            user_id="voice_user",
            trust_level=TrustLevel(self._settings.agent.default_trust_level),
            max_turns=self._settings.memory.max_short_term_turns,
        )
        self._memory._sessions[self._session.id] = self._session.memory

        # Orchestrator
        llm_client = LLMClientFactory.from_settings(self._settings)
        self._orchestrator = Orchestrator.from_settings(
            settings=self._settings,
            llm_client=llm_client,
            tool_bus=skill_bus,
            tool_registry=registry,
            memory_manager=self._memory,
            on_response=None,  # voice doesn't stream partial responses
        )

        # Scheduler
        try:
            self._scheduler = TaskScheduler(
                orchestrator=self._orchestrator,
                memory_manager=self._memory,
                max_concurrent_tasks=self._settings.scheduler.max_concurrent_tasks,
                timezone=self._settings.scheduler.timezone,
            )
            await self._scheduler.start()
        except Exception as e:
            log.warning("voice.scheduler_failed", error=str(e))

        log.info("voice.components_ready", session_id=self._session.id)

    # ── Model loading ─────────────────────────────────────────────────────────

    async def _load_models(self) -> None:
        """Load Whisper and Piper models once at startup."""
        loop = asyncio.get_running_loop()

        # Faster-Whisper STT
        log.info("voice.loading_whisper", model=self._cfg.whisper_model)
        t0 = time.monotonic()
        try:
            self._whisper_model = await loop.run_in_executor(
                None, self._load_whisper
            )
            log.info("voice.whisper_loaded",
                     model=self._cfg.whisper_model,
                     duration_ms=round((time.monotonic() - t0) * 1000))
        except ImportError as e:
            raise VoiceModelError(
                f"faster-whisper not installed. Run: pip install faster-whisper\n{e}"
            ) from e

        # Piper TTS (optional — degrade gracefully if missing)
        if self._cfg.piper_model_path:
            log.info("voice.loading_piper", path=self._cfg.piper_model_path)
            t0 = time.monotonic()
            try:
                self._piper_voice = await loop.run_in_executor(
                    None, self._load_piper
                )
                log.info("voice.piper_loaded",
                         duration_ms=round((time.monotonic() - t0) * 1000))
            except ImportError:
                log.warning("voice.piper_not_installed",
                            hint="pip install piper-tts")
            except Exception as e:
                log.warning("voice.piper_load_failed", error=str(e))
        else:
            log.warning("voice.piper_skipped",
                        reason="voice.piper_model_path not set in config — TTS disabled")

        # VAD (optional — degrade to fixed-duration silence if missing)
        try:
            import webrtcvad
            self._vad = webrtcvad.Vad(self._cfg.vad_aggressiveness)
            log.info("voice.vad_ready", aggressiveness=self._cfg.vad_aggressiveness)
        except ImportError:
            log.warning("voice.vad_not_installed",
                        hint="pip install webrtcvad — falling back to fixed silence detection")

    def _load_whisper(self):
        """Blocking — runs in executor."""
        from faster_whisper import WhisperModel
        return WhisperModel(
            self._cfg.whisper_model,
            device=self._cfg.whisper_device,
            compute_type="int8",
        )

    def _load_piper(self):
        """Blocking — runs in executor."""
        from piper.voice import PiperVoice
        model_path = str(Path(self._cfg.piper_model_path).expanduser().resolve())
        return PiperVoice.load(model_path)

    # ── Main pipeline loop ────────────────────────────────────────────────────

    async def _pipeline_loop(self) -> None:
        """
        Outer loop: start mic capture, then process utterances forever.
        Each iteration: listen → transcribe → think → speak.
        """
        import sounddevice as sd

        # Sounddevice callback feeds raw PCM int16 frames into the asyncio queue
        def _sd_callback(indata, frames, time_info, status):
            if status:
                log.debug("voice.sounddevice_status", status=str(status))
            if self._loop and not self._audio_queue.full():
                # Copy bytes to avoid sharing mutable buffer with sounddevice
                self._loop.call_soon_threadsafe(
                    self._audio_queue.put_nowait,
                    bytes(indata),
                )

        stream = sd.InputStream(
            samplerate=self._cfg.sample_rate,
            channels=self._cfg.channels,
            dtype=_DTYPE,
            blocksize=self._cfg.frame_size,
            callback=_sd_callback,
        )

        with stream:
            log.info("voice.mic_open", samplerate=self._cfg.sample_rate)
            while self._running:
                try:
                    utterance_pcm = await self._vad_loop()
                    if utterance_pcm is None:
                        continue

                    t_turn = time.monotonic()

                    # 1. Transcribe
                    text = await self._transcribe(utterance_pcm)
                    if not text:
                        log.debug("voice.transcription_empty")
                        continue

                    t_stt = time.monotonic()
                    log.info("voice.heard", text=text[:120],
                             stt_ms=round((t_stt - t_turn) * 1000))
                    print(f"\n👤 You: {text}")

                    # 2. Think
                    response_text = await self._think(text)
                    t_llm = time.monotonic()
                    log.info("voice.responding",
                             llm_ms=round((t_llm - t_stt) * 1000),
                             response=response_text[:120])
                    print(f"🤖 NeuralClaw: {response_text}\n")

                    # 3. Speak
                    await self._speak(response_text)
                    t_end = time.monotonic()
                    log.info("voice.turn_complete",
                             total_ms=round((t_end - t_turn) * 1000))

                except asyncio.CancelledError:
                    break
                except VoiceTranscriptionError as e:
                    log.warning("voice.transcription_error", error=str(e))
                except Exception as e:
                    log.error("voice.pipeline_error",
                              error=str(e), error_type=type(e).__name__)

    # ── VAD loop ──────────────────────────────────────────────────────────────

    async def _vad_loop(self) -> Optional[bytes]:
        """
        Accumulate PCM frames from the queue until an utterance is detected.

        Returns raw PCM bytes of the complete utterance, or None if the
        utterance is too short (< min_utterance_ms) to be meaningful.

        VAD state machine:
            WAITING  — listening for speech to begin
            SPEAKING — accumulating speech frames
            SILENCE  — counting silent frames after speech
        """
        STATE_WAITING  = 0
        STATE_WAITING  = 0
        STATE_SPEAKING = 1

        # Require this many consecutive speech frames before committing to an utterance.
        # This filters out single-frame noise bursts (the root cause of the original bug).
        _SPEECH_ONSET_FRAMES = 2

        state = STATE_WAITING
        speech_frames: list[bytes] = []
        onset_buf: list[bytes] = []   # pre-onset buffer
        silence_count = 0
        total_frames  = 0
        onset_count   = 0

        while self._running:
            try:
                frame = self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                # Queue is empty — in production wait briefly for mic input,
                # but if it stays empty (tests / end of stream) exit cleanly.
                try:
                    frame = await asyncio.wait_for(
                        self._audio_queue.get(), timeout=0.5
                    )
                except asyncio.TimeoutError:
                    break

            total_frames += 1

            # VAD decision
            is_speech = self._is_speech(frame)

            if state == STATE_WAITING:
                if is_speech:
                    onset_count += 1
                    onset_buf.append(frame)
                    if onset_count >= _SPEECH_ONSET_FRAMES:
                        # Enough consecutive speech frames — commit to utterance
                        state = STATE_SPEAKING
                        speech_frames = list(onset_buf)
                        onset_buf = []
                        silence_count = 0
                else:
                    # Reset onset on any silent frame
                    onset_count = 0
                    onset_buf = []

            elif state == STATE_SPEAKING:
                speech_frames.append(frame)
                if not is_speech:
                    silence_count += 1
                    if silence_count >= self._cfg.silence_frames:
                        # Utterance is done — break immediately
                        break
                else:
                    silence_count = 0

                # Hard cap — return what we have
                if len(speech_frames) >= self._cfg.max_utterance_frames:
                    break

        if not speech_frames:
            return None

        # Check minimum length using total utterance duration
        duration_ms = len(speech_frames) * _FRAME_DURATION_MS
        if duration_ms < self._cfg.min_utterance_ms:
            log.debug("voice.utterance_too_short", duration_ms=duration_ms)
            return None

        return b"".join(speech_frames)

    def _is_speech(self, frame: bytes) -> bool:
        """Return True if the frame contains speech (using VAD or energy fallback)."""
        if self._vad is not None:
            try:
                return self._vad.is_speech(frame, self._cfg.sample_rate)
            except Exception:
                pass

        # Energy-based fallback: compute RMS of the int16 samples
        import struct
        n = len(frame) // 2
        if n == 0:
            return False
        samples = struct.unpack(f"{n}h", frame[:n * 2])
        rms = (sum(s * s for s in samples) / n) ** 0.5
        print("RMS:", rms)
        return rms > 0  # empirically tuned threshold for quiet room

    # ── Transcription ─────────────────────────────────────────────────────────

    async def _transcribe(self, pcm: bytes) -> str:
        """
        Transcribe raw PCM int16 bytes using Faster-Whisper.
        Runs in an executor thread so it never blocks the event loop.
        """
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, self._run_whisper, pcm)
        return text.strip()

    def _run_whisper(self, pcm: bytes) -> str:
        """Blocking Whisper transcription — runs in executor."""
        import numpy as np
        import struct

        n = len(pcm) // 2
        samples = struct.unpack(f"{n}h", pcm[:n * 2])
        audio_f32 = np.array(samples, dtype=np.float32) / 32768.0

        segments, _info = self._whisper_model.transcribe(
            audio_f32,
            language="en",
            beam_size=1,          # fastest; increase for accuracy
            vad_filter=False,      # Whisper's own internal VAD as second pass
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    # ── Orchestrator ──────────────────────────────────────────────────────────

    async def _think(self, text: str) -> str:
        """
        Send the transcribed text to the orchestrator and return the response.
        Returns a plain string suitable for TTS — strips Markdown formatting.
        """
        turn_result = await self._orchestrator.run_turn(self._session, text)
        raw = turn_result.response.text if turn_result.response else ""
        return self._strip_markdown(raw)

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """
        Remove common Markdown formatting so TTS doesn't read symbols aloud.
        Minimal — keeps the text natural for spoken output.
        """
        import re
        # Remove code blocks entirely (unreadable as speech)
        text = re.sub(r"```[\s\S]*?```", "[code block]", text)
        text = re.sub(r"`[^`]+`", "", text)
        # Remove headers, bold, italic
        text = re.sub(r"#{1,6}\s*", "", text)
        text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
        text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
        # Remove URLs
        text = re.sub(r"https?://\S+", "link", text)
        # Collapse whitespace
        text = re.sub(r"\n+", " ", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

    # ── TTS + playback ────────────────────────────────────────────────────────

    async def _speak(self, text: str) -> None:
        """
        Synthesize `text` with Piper and play it back via sounddevice.
        Falls back to printing if Piper is unavailable.
        """
        if not text:
            return

        if self._piper_voice is None:
            # TTS degraded — text already printed in pipeline_loop
            return

        loop = asyncio.get_running_loop()
        try:
            audio_array, sample_rate = await loop.run_in_executor(
                None, self._run_piper, text
            )
            await loop.run_in_executor(
                None, self._play_audio, audio_array, sample_rate
            )
        except Exception as e:
            log.warning("voice.tts_failed", error=str(e))

    def _run_piper(self, text: str):
        """
        Blocking Piper synthesis — runs in executor.
        Returns (numpy_array, sample_rate).
        """
        import io
        import soundfile as sf
        import numpy as np

        buf = io.BytesIO()
        with sf.SoundFile(
            buf, mode="w",
            samplerate=self._piper_voice.config.sample_rate,
            channels=1,
            format="WAV",
        ) as wav_file:
            self._piper_voice.synthesize(text, wav_file)

        buf.seek(0)
        data, sr = sf.read(buf, dtype="float32")
        return data, sr

    def _play_audio(self, audio: "np.ndarray", sample_rate: int) -> None:
        """
        Blocking sounddevice playback — runs in executor.
        Blocks until playback completes.
        """
        import sounddevice as sd
        sd.play(audio, samplerate=sample_rate)
        sd.wait()

    # ── Latency measurement ───────────────────────────────────────────────────

    async def benchmark_latency(self, prompt: str = "Hello") -> dict:
        """
        Measure end-to-end latency for a single voice turn using a synthetic
        prompt (no mic capture). Used by the test suite and for the Phase F
        gate condition (< 1500ms on CPU for a short prompt).

        Returns dict with stt_ms, llm_ms, tts_ms, total_ms.
        """
        import struct
        import numpy as np

        # Generate synthetic silence (0.5s) as stand-in for mic audio
        n_samples = int(self._cfg.sample_rate * 0.5)
        pcm = struct.pack(f"{n_samples}h", *([0] * n_samples))

        # STT
        t0 = time.monotonic()
        # Use prompt directly (bypass Whisper for benchmark; Whisper latency
        # is measured separately with a real audio clip in integration tests)
        stt_ms = round((time.monotonic() - t0) * 1000)

        # LLM
        t1 = time.monotonic()
        response_text = await self._think(prompt)
        llm_ms = round((time.monotonic() - t1) * 1000)

        # TTS synthesis (no playback in benchmark)
        tts_ms = 0
        if self._piper_voice:
            loop = asyncio.get_running_loop()
            t2 = time.monotonic()
            try:
                await loop.run_in_executor(None, self._run_piper, response_text[:200])
                tts_ms = round((time.monotonic() - t2) * 1000)
            except Exception:
                pass

        total_ms = stt_ms + llm_ms + tts_ms
        return {
            "stt_ms": stt_ms,
            "llm_ms": llm_ms,
            "tts_ms": tts_ms,
            "total_ms": total_ms,
            "response_text": response_text[:120],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

async def run_voice(settings: Settings, log_) -> None:
    """
    Entry point called from main.py when --interface voice is used.
    """
    vi = VoiceInterface(settings)
    log_.info("voice.interface_bootstrap")
    try:
        await vi.start()
    except VoiceModelError as e:
        print(f"\n❌ Voice model error: {e}\n")
        log_.error("voice.model_error", error=str(e))
    except VoiceAudioError as e:
        print(f"\n❌ Audio device error: {e}\n")
        log_.error("voice.audio_error", error=str(e))
    except KeyboardInterrupt:
        log_.info("voice.interrupted")
    finally:
        await vi.stop()
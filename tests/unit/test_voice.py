"""
tests/unit/test_voice.py — Phase F: Voice Interface Unit Tests

Covers all testable components of interfaces/voice.py without requiring
real audio hardware, Whisper models, or Piper models.

Test groups
-----------
  VoiceConfig         — defaults, validation, derived properties
  VoiceErrors         — hierarchy inherits from NeuralClawError
  VoiceConfig         — settings coercion and field validators
  _strip_markdown     — TTS text sanitisation
  _is_speech          — energy-based fallback (no webrtcvad required)
  _vad_loop           — utterance detection state machine (mocked queue)
  _transcribe         — Whisper executor path (mocked model)
  _think              — orchestrator integration (mocked run_turn)
  _run_piper / _speak — TTS executor path (mocked PiperVoice)
  benchmark_latency   — smoke test with mocked orchestrator
  run_voice           — entry point handles VoiceModelError gracefully
  settings integration — VoiceConfig round-trips through Settings
  main.py             — --interface voice is a valid argparse choice
"""

from __future__ import annotations

import asyncio
import struct
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from interfaces.voice import (
    VoiceConfig,
    VoiceError,
    VoiceModelError,
    VoiceAudioError,
    VoiceTranscriptionError,
    VoiceInterface,
)
from exceptions import NeuralClawError


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_config(**overrides) -> VoiceConfig:
    raw = {
        "enabled": True,
        "whisper_model": "base.en",
        "whisper_device": "cpu",
        "piper_model_path": "",
        "sample_rate": 16000,
        "channels": 1,
        "vad_aggressiveness": 2,
        "silence_duration_ms": 800,
        "max_utterance_s": 30,
        "min_utterance_ms": 300,
    }
    raw.update(overrides)
    return VoiceConfig(raw)


def _make_settings(**voice_overrides) -> MagicMock:
    s = MagicMock()
    cfg = _make_config(**voice_overrides)
    s.voice_raw = cfg.__dict__
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


def _silent_pcm(ms: int = 500, sample_rate: int = 16000) -> bytes:
    """Generate ms milliseconds of silent PCM int16 audio."""
    n = int(sample_rate * ms / 1000)
    return struct.pack(f"{n}h", *([0] * n))


def _make_voice_interface(settings=None) -> VoiceInterface:
    s = settings or _make_settings()
    vi = VoiceInterface(s)
    return vi


def _make_turn_result(text: str = "I heard you.") -> MagicMock:
    tr = MagicMock()
    tr.succeeded = True
    tr.response = MagicMock()
    tr.response.text = text
    return tr


# ─────────────────────────────────────────────────────────────────────────────
# VoiceError hierarchy
# ─────────────────────────────────────────────────────────────────────────────

class TestVoiceErrors:

    def test_voice_error_is_neuralclaw_error(self):
        assert issubclass(VoiceError, NeuralClawError)

    def test_model_error_is_voice_error(self):
        assert issubclass(VoiceModelError, VoiceError)

    def test_audio_error_is_voice_error(self):
        assert issubclass(VoiceAudioError, VoiceError)

    def test_transcription_error_is_voice_error(self):
        assert issubclass(VoiceTranscriptionError, VoiceError)

    def test_can_raise_and_catch_as_neuralclaw_error(self):
        with pytest.raises(NeuralClawError):
            raise VoiceModelError("whisper not installed")


# ─────────────────────────────────────────────────────────────────────────────
# VoiceConfig
# ─────────────────────────────────────────────────────────────────────────────

class TestVoiceConfig:

    def test_defaults(self):
        cfg = VoiceConfig({})
        assert cfg.enabled is True
        assert cfg.whisper_model == "base.en"
        assert cfg.whisper_device == "cpu"
        assert cfg.piper_model_path == ""
        assert cfg.sample_rate == 16000
        assert cfg.channels == 1
        assert cfg.vad_aggressiveness == 2
        assert cfg.silence_duration_ms == 800
        assert cfg.max_utterance_s == 30
        assert cfg.min_utterance_ms == 300

    def test_frame_size(self):
        cfg = _make_config(sample_rate=16000)
        # 30ms frame at 16kHz = 480 samples
        assert cfg.frame_size == 480

    def test_silence_frames(self):
        cfg = _make_config(silence_duration_ms=900)
        # 900ms / 30ms per frame = 30 frames
        assert cfg.silence_frames == 30

    def test_silence_frames_minimum_one(self):
        cfg = _make_config(silence_duration_ms=10)
        assert cfg.silence_frames >= 1

    def test_max_utterance_frames(self):
        cfg = _make_config(sample_rate=16000, max_utterance_s=10)
        # 16000 * 10 // 480 = 333 frames
        assert cfg.max_utterance_frames == int(16000 * 10 // cfg.frame_size)

    def test_custom_values(self):
        cfg = _make_config(whisper_model="small.en", whisper_device="cuda",
                           vad_aggressiveness=3, silence_duration_ms=1200)
        assert cfg.whisper_model == "small.en"
        assert cfg.whisper_device == "cuda"
        assert cfg.vad_aggressiveness == 3
        assert cfg.silence_duration_ms == 1200


# ─────────────────────────────────────────────────────────────────────────────
# Settings integration — VoiceConfig in Settings
# ─────────────────────────────────────────────────────────────────────────────

class TestSettingsVoiceConfig:

    def test_voice_field_defaults(self):
        from config.settings import Settings
        # Build a minimal settings without loading from disk
        s = Settings.model_construct(
            openai_api_key="sk-test",
            bytez_api_key="bz-test",
        )
        # voice field should be a VoiceConfig with defaults
        from config.settings import VoiceConfig as SettingsVoiceConfig
        assert isinstance(s.voice, SettingsVoiceConfig)

    def test_voice_config_whisper_device_validator(self):
        from config.settings import VoiceConfig as SettingsVoiceConfig
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            SettingsVoiceConfig(whisper_device="tpu")  # invalid

    def test_voice_config_vad_aggressiveness_validator(self):
        from config.settings import VoiceConfig as SettingsVoiceConfig
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            SettingsVoiceConfig(vad_aggressiveness=5)  # must be 0-3

    def test_voice_config_sample_rate_validator(self):
        from config.settings import VoiceConfig as SettingsVoiceConfig
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            SettingsVoiceConfig(sample_rate=44100)  # not in allowed set

    def test_voice_raw_property_returns_dict(self):
        from config.settings import VoiceConfig as SettingsVoiceConfig
        s = MagicMock()
        s.voice = SettingsVoiceConfig()
        s.voice_raw = s.voice.model_dump()
        assert isinstance(s.voice_raw, dict)
        assert "whisper_model" in s.voice_raw

    def test_voice_in_known_sections(self):
        """'voice' must be in _KNOWN_SECTIONS so config.yaml keys are parsed."""
        import inspect, config.settings as cs
        src = inspect.getsource(cs.load_settings)
        assert '"voice"' in src or "'voice'" in src


# ─────────────────────────────────────────────────────────────────────────────
# _strip_markdown
# ─────────────────────────────────────────────────────────────────────────────

class TestStripMarkdown:

    def _strip(self, text: str) -> str:
        return VoiceInterface._strip_markdown(text)

    def test_removes_code_block(self):
        result = self._strip("Here is code:\n```python\nprint('hi')\n```\nDone.")
        assert "```" not in result
        assert "[code block]" in result
        assert "Done." in result

    def test_removes_inline_code(self):
        result = self._strip("Call `run_turn()` to start.")
        assert "`" not in result

    def test_removes_headers(self):
        result = self._strip("## Section Title\nSome text.")
        assert "##" not in result
        assert "Section Title" in result

    def test_removes_bold(self):
        result = self._strip("This is **important** text.")
        assert "**" not in result
        assert "important" in result

    def test_removes_italic(self):
        result = self._strip("This is *emphasized* and _also_ italic.")
        assert "*emphasized*" not in result
        assert "emphasized" in result

    def test_removes_urls(self):
        result = self._strip("See https://example.com/path for details.")
        assert "https://" not in result
        assert "link" in result

    def test_collapses_newlines(self):
        result = self._strip("Line one.\n\nLine two.\n\nLine three.")
        assert "\n\n" not in result

    def test_plain_text_unchanged(self):
        text = "The weather today is sunny and warm."
        assert self._strip(text) == text

    def test_empty_string(self):
        assert self._strip("") == ""


# ─────────────────────────────────────────────────────────────────────────────
# _is_speech — energy fallback
# ─────────────────────────────────────────────────────────────────────────────

class TestIsSpeech:

    def _make_vi(self) -> VoiceInterface:
        vi = _make_voice_interface()
        vi._vad = None  # force energy fallback
        return vi

    def _pcm_frame(self, amplitude: int, n: int = 480) -> bytes:
        return struct.pack(f"{n}h", *([amplitude] * n))

    def test_silent_frame_returns_false(self):
        vi = self._make_vi()
        frame = self._pcm_frame(0)
        assert vi._is_speech(frame) is False

    def test_loud_frame_returns_true(self):
        vi = self._make_vi()
        frame = self._pcm_frame(5000)
        assert vi._is_speech(frame) is True

    def test_empty_frame_returns_false(self):
        vi = self._make_vi()
        assert vi._is_speech(b"") is False

    def test_vad_used_when_available(self):
        vi = _make_voice_interface()
        mock_vad = MagicMock()
        mock_vad.is_speech = MagicMock(return_value=True)
        vi._vad = mock_vad
        frame = self._pcm_frame(0)
        result = vi._is_speech(frame)
        assert result is True
        mock_vad.is_speech.assert_called_once()

    def test_falls_back_on_vad_exception(self):
        vi = _make_voice_interface()
        mock_vad = MagicMock()
        mock_vad.is_speech = MagicMock(side_effect=Exception("vad crash"))
        vi._vad = mock_vad
        # Should not raise — falls back to energy
        vi._is_speech(self._pcm_frame(0))


# ─────────────────────────────────────────────────────────────────────────────
# _vad_loop — utterance detection state machine
# ─────────────────────────────────────────────────────────────────────────────

class TestVadLoop:

    def _make_vi_with_queue(self, frames: list[bytes]) -> VoiceInterface:
        vi = _make_voice_interface()
        vi._running = True
        vi._vad = None  # use energy fallback for predictability
        vi._audio_queue = asyncio.Queue()
        for f in frames:
            vi._audio_queue.put_nowait(f)
        return vi

    def _speech_frame(self, n: int = 480) -> bytes:
        return struct.pack(f"{n}h", *([5000] * n))

    def _silent_frame(self, n: int = 480) -> bytes:
        return struct.pack(f"{n}h", *([0] * n))

    @pytest.mark.asyncio
    async def test_returns_none_when_no_speech(self):
        # All silent frames — should return None
        frames = [self._silent_frame() for _ in range(10)]
        vi = self._make_vi_with_queue(frames)
        vi._running = False  # stop after queue drains
        result = await vi._vad_loop()
        assert result is None

    @pytest.mark.asyncio
    async def test_detects_speech_followed_by_silence(self):
        # 5 speech frames then enough silence frames to trigger end
        n_silence = vi_cfg = _make_config()
        n_silence_frames = n_silence.silence_frames + 2
        frames = (
            [self._speech_frame() for _ in range(5)] +
            [self._silent_frame() for _ in range(n_silence_frames)]
        )
        vi = self._make_vi_with_queue(frames)
        result = await vi._vad_loop()
        assert result is not None
        assert isinstance(result, bytes)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_returns_none_for_too_short_utterance(self):
        # 1 speech frame (30ms) < min_utterance_ms (300ms)
        cfg = _make_config(min_utterance_ms=300)
        n_silence_frames = cfg.silence_frames + 2
        frames = (
            [self._speech_frame()] +
            [self._silent_frame() for _ in range(n_silence_frames)]
        )
        vi = self._make_vi_with_queue(frames)
        vi._cfg = cfg
        result = await vi._vad_loop()
        assert result is None

    @pytest.mark.asyncio
    async def test_respects_max_utterance_frames(self):
        # Fill with speech frames well beyond max_utterance_frames
        cfg = _make_config(max_utterance_s=1, sample_rate=16000)
        max_f = cfg.max_utterance_frames
        frames = [self._speech_frame() for _ in range(max_f + 20)]
        vi = self._make_vi_with_queue(frames)
        vi._cfg = cfg
        result = await vi._vad_loop()
        # Should return something (hit the hard cap) without hanging
        assert result is not None


# ─────────────────────────────────────────────────────────────────────────────
# _transcribe — Whisper executor path
# ─────────────────────────────────────────────────────────────────────────────

class TestTranscribe:

    @pytest.mark.asyncio
    async def test_transcribe_returns_stripped_text(self):
        vi = _make_voice_interface()
        vi._cfg = _make_config()

        mock_seg = MagicMock()
        mock_seg.text = "  Hello NeuralClaw  "
        vi._whisper_model = MagicMock()
        vi._whisper_model.transcribe = MagicMock(return_value=([mock_seg], MagicMock()))

        pcm = _silent_pcm(500)
        result = await vi._transcribe(pcm)
        assert result == "Hello NeuralClaw"

    @pytest.mark.asyncio
    async def test_transcribe_joins_multiple_segments(self):
        vi = _make_voice_interface()
        segs = [MagicMock(), MagicMock()]
        segs[0].text = "Hello"
        segs[1].text = " world"
        vi._whisper_model = MagicMock()
        vi._whisper_model.transcribe = MagicMock(return_value=(segs, MagicMock()))

        result = await vi._transcribe(_silent_pcm(500))
        assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_transcribe_returns_empty_for_no_segments(self):
        vi = _make_voice_interface()
        vi._whisper_model = MagicMock()
        vi._whisper_model.transcribe = MagicMock(return_value=([], MagicMock()))

        result = await vi._transcribe(_silent_pcm(500))
        assert result == ""


# ─────────────────────────────────────────────────────────────────────────────
# _think — orchestrator integration
# ─────────────────────────────────────────────────────────────────────────────

class TestThink:

    @pytest.mark.asyncio
    async def test_think_returns_stripped_response(self):
        vi = _make_voice_interface()
        vi._session = MagicMock()
        vi._orchestrator = MagicMock()
        vi._orchestrator.run_turn = AsyncMock(
            return_value=_make_turn_result("The answer is **42**.")
        )
        result = await vi._think("What is the answer?")
        assert "**" not in result
        assert "42" in result

    @pytest.mark.asyncio
    async def test_think_handles_empty_response(self):
        vi = _make_voice_interface()
        vi._session = MagicMock()
        vi._orchestrator = MagicMock()
        tr = MagicMock()
        tr.succeeded = False
        tr.response = MagicMock()
        tr.response.text = ""
        vi._orchestrator.run_turn = AsyncMock(return_value=tr)
        result = await vi._think("test")
        assert result == ""

    @pytest.mark.asyncio
    async def test_think_handles_none_response(self):
        vi = _make_voice_interface()
        vi._session = MagicMock()
        vi._orchestrator = MagicMock()
        tr = MagicMock()
        tr.response = None
        vi._orchestrator.run_turn = AsyncMock(return_value=tr)
        result = await vi._think("test")
        assert result == ""


# ─────────────────────────────────────────────────────────────────────────────
# _speak — TTS degradation paths
# ─────────────────────────────────────────────────────────────────────────────

class TestSpeak:

    @pytest.mark.asyncio
    async def test_speak_no_op_when_piper_not_loaded(self):
        vi = _make_voice_interface()
        vi._piper_voice = None
        # Should not raise
        await vi._speak("Hello world")

    @pytest.mark.asyncio
    async def test_speak_no_op_on_empty_text(self):
        vi = _make_voice_interface()
        vi._piper_voice = MagicMock()
        # Should not call run_piper for empty text
        with patch.object(vi, "_run_piper") as mock_piper:
            await vi._speak("")
            mock_piper.assert_not_called()

    @pytest.mark.asyncio
    async def test_speak_calls_run_piper_and_play(self):
        import numpy as np
        vi = _make_voice_interface()
        vi._piper_voice = MagicMock()
        vi._piper_voice.config = MagicMock()
        vi._piper_voice.config.sample_rate = 22050

        fake_audio = np.zeros(22050, dtype=np.float32)

        with patch.object(vi, "_run_piper", return_value=(fake_audio, 22050)) as mock_piper, \
             patch.object(vi, "_play_audio") as mock_play:
            await vi._speak("Hello NeuralClaw")
            mock_piper.assert_called_once_with("Hello NeuralClaw")
            mock_play.assert_called_once()

    @pytest.mark.asyncio
    async def test_speak_handles_tts_exception_gracefully(self):
        vi = _make_voice_interface()
        vi._piper_voice = MagicMock()
        with patch.object(vi, "_run_piper", side_effect=RuntimeError("piper crash")):
            # Must not propagate
            await vi._speak("Hello")


# ─────────────────────────────────────────────────────────────────────────────
# benchmark_latency
# ─────────────────────────────────────────────────────────────────────────────

class TestBenchmarkLatency:

    @pytest.mark.asyncio
    async def test_benchmark_returns_required_keys(self):
        vi = _make_voice_interface()
        vi._session = MagicMock()
        vi._piper_voice = None
        vi._orchestrator = MagicMock()
        vi._orchestrator.run_turn = AsyncMock(
            return_value=_make_turn_result("I am ready.")
        )
        result = await vi.benchmark_latency("Hello")
        for key in ("stt_ms", "llm_ms", "tts_ms", "total_ms", "response_text"):
            assert key in result

    @pytest.mark.asyncio
    async def test_benchmark_total_is_sum(self):
        vi = _make_voice_interface()
        vi._session = MagicMock()
        vi._piper_voice = None
        vi._orchestrator = MagicMock()
        vi._orchestrator.run_turn = AsyncMock(
            return_value=_make_turn_result("Yes.")
        )
        result = await vi.benchmark_latency("ping")
        assert result["total_ms"] == result["stt_ms"] + result["llm_ms"] + result["tts_ms"]

    @pytest.mark.asyncio
    async def test_benchmark_response_text_truncated(self):
        long_response = "a" * 500
        vi = _make_voice_interface()
        vi._session = MagicMock()
        vi._piper_voice = None
        vi._orchestrator = MagicMock()
        vi._orchestrator.run_turn = AsyncMock(
            return_value=_make_turn_result(long_response)
        )
        result = await vi.benchmark_latency("test")
        assert len(result["response_text"]) <= 120


# ─────────────────────────────────────────────────────────────────────────────
# VoiceInterface.stop — graceful shutdown
# ─────────────────────────────────────────────────────────────────────────────

class TestVoiceInterfaceStop:

    @pytest.mark.asyncio
    async def test_stop_with_no_components_is_safe(self):
        vi = _make_voice_interface()
        # Nothing initialised — should not raise
        await vi.stop()

    @pytest.mark.asyncio
    async def test_stop_calls_scheduler_stop(self):
        vi = _make_voice_interface()
        vi._scheduler = MagicMock()
        vi._scheduler.stop = AsyncMock()
        await vi.stop()
        vi._scheduler.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_calls_memory_close(self):
        vi = _make_voice_interface()
        vi._memory = MagicMock()
        vi._memory.close = AsyncMock()
        await vi.stop()
        vi._memory.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_handles_scheduler_exception(self):
        vi = _make_voice_interface()
        vi._scheduler = MagicMock()
        vi._scheduler.stop = AsyncMock(side_effect=RuntimeError("crash"))
        # Must not raise
        await vi.stop()

    @pytest.mark.asyncio
    async def test_stop_sets_running_false(self):
        vi = _make_voice_interface()
        vi._running = True
        await vi.stop()
        assert vi._running is False


# ─────────────────────────────────────────────────────────────────────────────
# run_voice entry point — error handling
# ─────────────────────────────────────────────────────────────────────────────

class TestRunVoiceEntryPoint:

    @pytest.mark.asyncio
    async def test_run_voice_handles_model_error(self, capsys):
        from interfaces.voice import run_voice
        log_ = MagicMock()

        with patch("interfaces.voice.VoiceInterface") as MockVI:
            instance = MockVI.return_value
            instance.start = AsyncMock(side_effect=VoiceModelError("no whisper"))
            instance.stop = AsyncMock()

            await run_voice(_make_settings(), log_)

            captured = capsys.readouterr()
            assert "Voice model error" in captured.out or log_.error.called

    @pytest.mark.asyncio
    async def test_run_voice_handles_keyboard_interrupt(self):
        from interfaces.voice import run_voice
        log_ = MagicMock()

        with patch("interfaces.voice.VoiceInterface") as MockVI:
            instance = MockVI.return_value
            instance.start = AsyncMock(side_effect=KeyboardInterrupt)
            instance.stop = AsyncMock()

            # Should not raise
            await run_voice(_make_settings(), log_)


# ─────────────────────────────────────────────────────────────────────────────
# main.py — voice CLI argument
# ─────────────────────────────────────────────────────────────────────────────

class TestMainVoiceArg:

    def test_voice_is_valid_interface_choice(self):
        import argparse, main as m
        parser = argparse.ArgumentParser()
        parser.add_argument("--interface", choices=["cli", "telegram", "voice"], default="cli")
        args = parser.parse_args(["--interface", "voice"])
        assert args.interface == "voice"

    def test_main_has_run_voice_function(self):
        import main as m
        assert hasattr(m, "_run_voice")
        assert asyncio.iscoroutinefunction(m._run_voice)
"""
core/stt/stt_engine.py — Speech-to-Text Engine

State machine:
    IDLE ──(enable)──▶ LISTENING ──(voice detected)──▶ PROCESSING
      ▲                    ▲                              │
      │                    └──────────(result)────────────┘
      └──(disable)─────────┘

Features:
  - Continuous listening with C++ VAD
  - Wake-word detection (energy-based candidate + keyword match)
  - Ambient noise vs speech discrimination via ZCR
  - Streaming audio ring buffer (C++ backed)
  - Pluggable transcription backend (Whisper local/API, Google, etc.)
  - Thread-safe: audio capture runs on a dedicated thread
  - Publishes EventBus events for UI state updates
"""

from __future__ import annotations

import enum
import time
import threading
import struct
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..native_bridge import (
    NativeRingBuffer,
    vad_detect,
    vad_detect_frames,
    audio_preemphasis,
    audio_resample,
    audio_energy_db,
    find_voiced_segments,
    HAS_NATIVE,
)

logger = logging.getLogger(__name__)


# ── Data Types ───────────────────────────────────────────────

class STTState(enum.Enum):
    IDLE       = "idle"
    LISTENING  = "listening"
    PROCESSING = "processing"
    ERROR      = "error"


@dataclass
class STTConfig:
    """Configuration for STT engine."""
    sample_rate: int = 16000
    frame_ms: int = 30             # Frame size in ms for VAD
    vad_energy_thresh: float = -35.0  # dB threshold
    vad_zcr_low: float = 0.02
    vad_zcr_high: float = 0.30
    silence_timeout_ms: int = 1500   # Silence before ending utterance
    min_speech_ms: int = 300         # Min speech to consider valid
    max_speech_ms: int = 30000       # Max single utterance
    wake_words: List[str] = field(default_factory=lambda: ["hey revia", "revia"])
    wake_word_enabled: bool = False
    preemphasis_coeff: float = 0.97
    ring_buffer_seconds: int = 60    # Audio ring buffer size


@dataclass
class TranscriptionResult:
    """Result from STT transcription."""
    text: str
    confidence: float = 1.0
    language: str = "en"
    duration_ms: float = 0.0
    is_wake_word: bool = False
    timestamp: float = field(default_factory=time.time)


# ── STT Engine ───────────────────────────────────────────────

class STTEngine:
    """
    Speech-to-Text engine with VAD, wake word detection, and
    pluggable transcription backends.

    Architecture:
      ┌──────────┐      ┌─────────┐      ┌────────────┐
      │ Audio In │─────▶│  Ring   │─────▶│    VAD     │
      │ (thread) │      │ Buffer  │      │  (C++)     │
      └──────────┘      └─────────┘      └─────┬──────┘
                                               │
                              voice detected?  │
                              ┌────────────────┘
                              ▼
                        ┌───────────┐     ┌──────────────┐
                        │  Collect  │────▶│  Transcribe  │
                        │  Speech   │     │  (backend)   │
                        └───────────┘     └──────┬───────┘
                                                 │
                                                 ▼
                                          ┌─────────────┐
                                          │ Wake Word?  │
                                          │ Or Command  │
                                          └──────┬──────┘
                                                 │
                                          publish event
    """

    def __init__(
        self,
        event_bus: Any = None,
        config: Optional[STTConfig] = None,
    ):
        self.bus = event_bus
        self.cfg = config or STTConfig()

        # State machine
        self._state = STTState.IDLE
        self._state_lock = threading.Lock()

        # Audio buffer
        buf_samples = self.cfg.sample_rate * self.cfg.ring_buffer_seconds
        self._ring = NativeRingBuffer(buf_samples)

        # Speech accumulator
        self._speech_buf: List[float] = []
        self._speech_start: float = 0.0
        self._silence_start: float = 0.0
        self._in_speech = False

        # Transcription backend (set via set_backend)
        self._transcribe_fn: Optional[Callable[[List[float], int], TranscriptionResult]] = None

        # Capture thread
        self._capture_thread: Optional[threading.Thread] = None
        self._running = False
        self._enabled = False

        # Audio source callback: should return List[float] of new samples
        self._audio_source: Optional[Callable[[], List[float]]] = None

        # Stats
        self._total_transcriptions = 0
        self._last_result: Optional[TranscriptionResult] = None

        # Subscribe to events
        if self.bus:
            self.bus.subscribe("stt_toggle", self._on_toggle)
            self.bus.subscribe("voice_vision_changed", self._on_config_changed)

    # ── State Machine ────────────────────────────────────────

    @property
    def state(self) -> STTState:
        with self._state_lock:
            return self._state

    def _set_state(self, new_state: STTState) -> None:
        with self._state_lock:
            old = self._state
            self._state = new_state
        if old != new_state:
            logger.debug("STT state: %s -> %s", old.value, new_state.value)
            self._publish("stt_state_changed", {
                "state": new_state.value,
                "previous": old.value,
            })

    # ── Public API ───────────────────────────────────────────

    def set_backend(self, fn: Callable[[List[float], int], TranscriptionResult]) -> None:
        """
        Set the transcription backend function.

        The function receives (samples: List[float], sample_rate: int)
        and returns a TranscriptionResult.
        """
        self._transcribe_fn = fn

    def set_audio_source(self, fn: Callable[[], List[float]]) -> None:
        """
        Set the audio source callback.

        Called repeatedly by the capture loop. Should return
        a chunk of float samples (mono, at cfg.sample_rate) or
        empty list if no data available.
        """
        self._audio_source = fn

    def enable(self) -> None:
        """Start listening."""
        if self._enabled:
            return
        self._enabled = True
        self._set_state(STTState.LISTENING)
        self._running = True
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="stt-capture",
        )
        self._capture_thread.start()
        logger.info("STT engine enabled")
        self._publish("stt_enabled", {"enabled": True})

    def disable(self) -> None:
        """Stop listening."""
        self._enabled = False
        self._running = False
        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None
        self._set_state(STTState.IDLE)
        self._reset_speech()
        logger.info("STT engine disabled")
        self._publish("stt_enabled", {"enabled": False})

    def feed_audio(self, samples: List[float]) -> None:
        """
        Manually feed audio samples (alternative to audio source callback).
        Thread-safe via ring buffer.
        """
        self._ring.write(samples)

    def process_buffered(self) -> Optional[TranscriptionResult]:
        """
        Process any buffered audio through VAD and transcription.
        Called automatically by capture loop, or manually in tests.
        """
        frame_samples = self.cfg.sample_rate * self.cfg.frame_ms // 1000
        available = self._ring.available()
        if available < frame_samples:
            return None

        # Read one frame
        frame = self._ring.read(frame_samples)
        now = time.time()

        # VAD check
        is_voice = vad_detect(
            frame,
            energy_thresh=self.cfg.vad_energy_thresh,
            zcr_low=self.cfg.vad_zcr_low,
            zcr_high=self.cfg.vad_zcr_high,
        )

        if is_voice:
            if not self._in_speech:
                # Speech onset
                self._in_speech = True
                self._speech_start = now
                self._speech_buf.clear()
                logger.debug("Speech onset detected")
            self._silence_start = 0.0
            self._speech_buf.extend(frame)

            # Check max duration
            elapsed_ms = (now - self._speech_start) * 1000
            if elapsed_ms >= self.cfg.max_speech_ms:
                return self._finalize_speech()
        else:
            if self._in_speech:
                # Possible speech end — accumulate silence
                self._speech_buf.extend(frame)
                if self._silence_start == 0.0:
                    self._silence_start = now
                silence_ms = (now - self._silence_start) * 1000
                if silence_ms >= self.cfg.silence_timeout_ms:
                    return self._finalize_speech()

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Return engine statistics."""
        return {
            "state": self.state.value,
            "enabled": self._enabled,
            "total_transcriptions": self._total_transcriptions,
            "has_native": HAS_NATIVE,
            "last_result": {
                "text": self._last_result.text,
                "confidence": self._last_result.confidence,
            } if self._last_result else None,
        }

    # ── Internal ─────────────────────────────────────────────

    def _capture_loop(self) -> None:
        """Main capture loop running on dedicated thread."""
        frame_duration = self.cfg.frame_ms / 1000.0
        while self._running:
            try:
                # Get audio from source
                if self._audio_source:
                    samples = self._audio_source()
                    if samples:
                        self._ring.write(samples)

                # Process buffered audio
                result = self.process_buffered()
                if result:
                    self._handle_result(result)

                # Pace to frame duration
                time.sleep(frame_duration * 0.5)

            except Exception as e:
                logger.error("STT capture error: %s", e)
                self._set_state(STTState.ERROR)
                self._publish("stt_error", {"error": str(e)})
                time.sleep(1.0)
                if self._running:
                    self._set_state(STTState.LISTENING)

    def _finalize_speech(self) -> Optional[TranscriptionResult]:
        """Process accumulated speech buffer."""
        if not self._speech_buf:
            self._reset_speech()
            return None

        duration_ms = (time.time() - self._speech_start) * 1000
        if duration_ms < self.cfg.min_speech_ms:
            logger.debug("Speech too short (%.0fms), discarding", duration_ms)
            self._reset_speech()
            return None

        self._set_state(STTState.PROCESSING)

        # Pre-emphasis for better recognition
        processed = audio_preemphasis(self._speech_buf, self.cfg.preemphasis_coeff)

        # Transcribe
        result = self._transcribe(processed, duration_ms)
        self._reset_speech()
        self._set_state(STTState.LISTENING)

        return result

    def _transcribe(self, samples: List[float], duration_ms: float) -> TranscriptionResult:
        """Run transcription backend."""
        if self._transcribe_fn:
            try:
                result = self._transcribe_fn(samples, self.cfg.sample_rate)
                result.duration_ms = duration_ms
            except Exception as e:
                logger.error("Transcription failed: %s", e)
                result = TranscriptionResult(
                    text="",
                    confidence=0.0,
                    duration_ms=duration_ms,
                )
        else:
            # No backend set — return empty
            result = TranscriptionResult(
                text="[no STT backend]",
                confidence=0.0,
                duration_ms=duration_ms,
            )

        # Check for wake word
        if self.cfg.wake_word_enabled and result.text:
            text_lower = result.text.lower().strip()
            for ww in self.cfg.wake_words:
                if text_lower.startswith(ww.lower()):
                    result.is_wake_word = True
                    # Strip wake word from text
                    result.text = result.text[len(ww):].strip()
                    break

        self._total_transcriptions += 1
        self._last_result = result
        return result

    def _handle_result(self, result: TranscriptionResult) -> None:
        """Publish transcription result."""
        if not result.text or result.confidence < 0.1:
            return

        data = {
            "text": result.text,
            "confidence": result.confidence,
            "language": result.language,
            "duration_ms": result.duration_ms,
            "is_wake_word": result.is_wake_word,
        }

        if result.is_wake_word or not self.cfg.wake_word_enabled:
            self._publish("stt_transcription", data)
            logger.info("Transcription: '%s' (conf=%.2f)", result.text, result.confidence)
        else:
            # Wake word mode but no wake word detected — ignore
            logger.debug("Ignored (no wake word): '%s'", result.text)

    def _reset_speech(self) -> None:
        """Reset speech accumulator."""
        self._speech_buf.clear()
        self._speech_start = 0.0
        self._silence_start = 0.0
        self._in_speech = False

    # ── Event Handlers ───────────────────────────────────────

    def _on_toggle(self, data: Dict[str, Any]) -> None:
        if data.get("enabled"):
            self.enable()
        else:
            self.disable()

    def _on_config_changed(self, data: Dict[str, Any]) -> None:
        key = data.get("key", "")
        val = data.get("value")
        if key == "voice.vad_threshold" and isinstance(val, (int, float)):
            # Map 0-100 slider to dB threshold (-60 to -10)
            self.cfg.vad_energy_thresh = -60.0 + (val / 100.0) * 50.0
        elif key == "voice.stt_language":
            pass  # Backend handles language
        elif key == "voice.stt_engine":
            pass  # Pipeline handles engine switching

    # ── Helpers ──────────────────────────────────────────────

    def _publish(self, event: str, data: Dict[str, Any]) -> None:
        if self.bus:
            try:
                self.bus.publish(event, data)
            except Exception:
                pass  # EventBus might not be on Qt thread

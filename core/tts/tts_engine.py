"""
core/tts/tts_engine.py — Text-to-Speech Engine

State machine:
    IDLE ──(speak)──▶ SPEAKING ──(done)──▶ IDLE
      │                  │
      │              (mute)──▶ MUTED
      │                  │
      └──(error)──▶ ERROR ──(recover)──▶ IDLE

Features:
  - Emotion-driven prosody: pitch, rate, volume, emphasis from EmotionEngine
  - Viseme/phoneme timing for avatar animation
  - Safety filter integration: output filtered before speaking
  - Async speech synthesis on worker thread
  - Speech queue with priority (interrupt for urgent messages)
  - Pluggable synthesis backend (Piper, ElevenLabs, etc.)
"""

from __future__ import annotations

import enum
import time
import queue
import threading
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..native_bridge import estimate_phoneme_timing

logger = logging.getLogger(__name__)


# ── Viseme Mapping ───────────────────────────────────────────

# Standard MPEG-4 viseme IDs mapped from phoneme categories
PHONEME_TO_VISEME = {
    # Silence
    "sil": 0,  "sp": 0,
    # Bilabial (p, b, m)
    "p": 1, "b": 1, "m": 1,
    # Labiodental (f, v)
    "f": 2, "v": 2,
    # Interdental (th, dh)
    "th": 3, "dh": 3,
    # Alveolar (t, d, s, z, n, l)
    "t": 4, "d": 4, "s": 4, "z": 4, "n": 4, "l": 4,
    # Postalveolar (sh, zh, ch, jh)
    "sh": 5, "zh": 5, "ch": 5, "jh": 5,
    # Velar/glottal (k, g, ng, hh)
    "k": 6, "g": 6, "ng": 6, "hh": 6,
    # Vowels - open (aa, ae, ah)
    "aa": 7, "ae": 7, "ah": 7,
    # Vowels - mid (eh, er, ax)
    "eh": 8, "er": 8, "ax": 8,
    # Vowels - close (iy, ih)
    "iy": 9, "ih": 9,
    # Vowels - round (ow, uw, uh)
    "ow": 10, "uw": 10, "uh": 10,
    # Diphthongs (ay, ey, oy, aw)
    "ay": 11, "ey": 11, "oy": 11, "aw": 11,
    # R-colored (r, w, y)
    "r": 12, "w": 12, "y": 12,
}

# Viseme names for UI/animation
VISEME_NAMES = [
    "silence",       # 0
    "bilabial",      # 1  (p, b, m)
    "labiodental",   # 2  (f, v)
    "interdental",   # 3  (th)
    "alveolar",      # 4  (t, d, s)
    "postalveolar",  # 5  (sh, ch)
    "velar",         # 6  (k, g)
    "open_vowel",    # 7  (aa, ah)
    "mid_vowel",     # 8  (eh, er)
    "close_vowel",   # 9  (iy, ih)
    "round_vowel",   # 10 (ow, uw)
    "diphthong",     # 11 (ay, ey)
    "approximant",   # 12 (r, w)
]


# ── Data Types ───────────────────────────────────────────────

class TTSState(enum.Enum):
    IDLE     = "idle"
    SPEAKING = "speaking"
    MUTED    = "muted"
    ERROR    = "error"


@dataclass
class ProsodyParams:
    """
    Prosody parameters derived from emotional state.

    These map to SSML-like controls that TTS backends understand:
      pitch:    -50% to +50%  (baseline shift)
      rate:     0.5 to 2.0   (speed multiplier)
      volume:   0.0 to 1.0   (output volume)
      emphasis: 0.0 to 1.0   (how much to stress words)
      warmth:   0.0 to 1.0   (voice softness/warmth)
      energy:   0.0 to 1.0   (vocal energy/breathiness)
    """
    pitch: float = 0.0
    rate: float = 1.0
    volume: float = 0.8
    emphasis: float = 0.5
    warmth: float = 0.5
    energy: float = 0.5
    emotion_tag: str = "neutral"


@dataclass
class VisemeData:
    """Timing data for avatar mouth animation."""
    viseme_id: int = 0
    viseme_name: str = "silence"
    start_ms: float = 0.0
    end_ms: float = 0.0
    weight: float = 1.0  # 0-1, how open/pronounced


@dataclass
class SpeechResult:
    """Result from TTS synthesis."""
    audio_data: bytes = b""         # Raw PCM or encoded audio
    sample_rate: int = 22050
    duration_ms: float = 0.0
    visemes: List[VisemeData] = field(default_factory=list)
    prosody: ProsodyParams = field(default_factory=ProsodyParams)
    text: str = ""
    success: bool = True
    error: str = ""


@dataclass
class TTSConfig:
    """Configuration for TTS engine."""
    default_rate: float = 1.0
    default_volume: float = 0.8
    max_queue_size: int = 20
    enable_visemes: bool = True
    emotion_influence: float = 0.7    # How much emotion affects prosody (0-1)
    sample_rate: int = 22050


# ── Emotion → Prosody Mapping ────────────────────────────────

# Each emotion category maps to prosody adjustments
_EMOTION_PROSODY = {
    # category:       (pitch%, rate, volume, emphasis, warmth, energy)
    "joy":           (+15.0, 1.15, 0.85, 0.7, 0.8, 0.8),
    "peace":         (-5.0,  0.90, 0.70, 0.3, 0.9, 0.3),
    "love":          (+5.0,  0.95, 0.75, 0.5, 1.0, 0.5),
    "drive":         (+10.0, 1.10, 0.90, 0.8, 0.5, 0.9),
    "engagement":    (+8.0,  1.05, 0.80, 0.6, 0.6, 0.7),
    "fear":          (+20.0, 1.20, 0.70, 0.4, 0.2, 0.9),
    "sadness":       (-15.0, 0.85, 0.60, 0.3, 0.6, 0.2),
    "anger":         (+5.0,  1.10, 0.95, 0.9, 0.1, 1.0),
    "shame":         (-10.0, 0.88, 0.55, 0.2, 0.4, 0.3),
    "fatigue":       (-20.0, 0.80, 0.50, 0.2, 0.5, 0.1),
    "detachment":    (-10.0, 0.92, 0.60, 0.1, 0.3, 0.2),
    "depression":    (-25.0, 0.75, 0.45, 0.1, 0.3, 0.1),
}

# Specific emotion overrides for fine-tuned expression
_EMOTION_SPECIFIC = {
    "playful":     (+20.0, 1.20, 0.85, 0.8, 0.7, 0.9),
    "sarcastic":   (+5.0,  0.95, 0.80, 0.9, 0.2, 0.6),
    "mischievous": (+15.0, 1.10, 0.80, 0.7, 0.4, 0.8),
    "teasing":     (+12.0, 1.08, 0.82, 0.7, 0.6, 0.7),
    "excited":     (+25.0, 1.25, 0.90, 0.9, 0.6, 1.0),
    "terrified":   (+30.0, 1.30, 0.75, 0.5, 0.1, 1.0),
    "enraged":     (+10.0, 1.15, 1.00, 1.0, 0.0, 1.0),
    "heartbroken": (-20.0, 0.80, 0.50, 0.4, 0.7, 0.2),
    "confident":   (+5.0,  1.05, 0.90, 0.8, 0.6, 0.8),
    "shy":         (-15.0, 0.85, 0.50, 0.2, 0.7, 0.2),
    "bored":       (-15.0, 0.82, 0.55, 0.1, 0.3, 0.1),
}


# ── TTS Engine ───────────────────────────────────────────────

class TTSEngine:
    """
    Text-to-Speech engine with emotion-driven prosody and
    viseme timing for avatar animation.

    Pipeline:
      Text ──▶ Safety Filter ──▶ Prosody Calc ──▶ Synthesis ──▶ Viseme Gen ──▶ Playback
                                    ▲
                                    │
                              Emotion State
    """

    def __init__(
        self,
        event_bus: Any = None,
        config: Optional[TTSConfig] = None,
    ):
        self.bus = event_bus
        self.cfg = config or TTSConfig()

        # State
        self._state = TTSState.IDLE
        self._state_lock = threading.Lock()
        self._muted = False

        # Speech queue
        self._queue: queue.Queue = queue.Queue(maxsize=self.cfg.max_queue_size)
        self._worker: Optional[threading.Thread] = None
        self._running = False

        # Synthesis backend
        self._synthesize_fn: Optional[Callable[[str, ProsodyParams], SpeechResult]] = None

        # Playback callback
        self._playback_fn: Optional[Callable[[SpeechResult], None]] = None

        # Current emotion state (updated by subscription)
        self._current_emotion: str = "neutral"
        self._current_intensity: float = 0.0
        self._current_category: str = "peace"

        # Safety filter reference (set externally)
        self._safety_filter = None

        # Stats
        self._total_speeches = 0
        self._total_duration_ms = 0.0

        # Subscribe to events
        if self.bus:
            self.bus.subscribe("emotion_state_changed", self._on_emotion_changed)
            self.bus.subscribe("tts_toggle", self._on_toggle)
            self.bus.subscribe("tts_mute", self._on_mute)

        # Start worker
        self._start_worker()

    # ── State Machine ────────────────────────────────────────

    @property
    def state(self) -> TTSState:
        with self._state_lock:
            return self._state

    def _set_state(self, new_state: TTSState) -> None:
        with self._state_lock:
            old = self._state
            self._state = new_state
        if old != new_state:
            logger.debug("TTS state: %s -> %s", old.value, new_state.value)
            self._publish("tts_state_changed", {
                "state": new_state.value,
                "previous": old.value,
            })

    # ── Public API ───────────────────────────────────────────

    def set_backend(self, fn: Callable[[str, ProsodyParams], SpeechResult]) -> None:
        """
        Set synthesis backend.

        fn(text, prosody) -> SpeechResult with audio_data filled.
        """
        self._synthesize_fn = fn

    def set_playback(self, fn: Callable[[SpeechResult], None]) -> None:
        """Set audio playback callback."""
        self._playback_fn = fn

    def set_safety_filter(self, safety_filter: Any) -> None:
        """Set safety filter for output filtering before speech."""
        self._safety_filter = safety_filter

    def speak(self, text: str, priority: bool = False) -> None:
        """
        Queue text for speech synthesis.

        If priority=True, clears queue and speaks immediately.
        """
        if self._muted:
            logger.debug("TTS muted, ignoring: %s", text[:50])
            return

        if priority:
            # Clear existing queue
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

        try:
            self._queue.put_nowait(text)
        except queue.Full:
            logger.warning("TTS queue full, dropping: %s", text[:50])

    def mute(self) -> None:
        """Mute TTS output."""
        self._muted = True
        self._set_state(TTSState.MUTED)
        self._publish("tts_muted", {"muted": True})

    def unmute(self) -> None:
        """Unmute TTS output."""
        self._muted = False
        self._set_state(TTSState.IDLE)
        self._publish("tts_muted", {"muted": False})

    def stop(self) -> None:
        """Stop current speech and clear queue."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._set_state(TTSState.IDLE)

    def shutdown(self) -> None:
        """Shutdown the worker thread."""
        self._running = False
        self._queue.put(None)  # Sentinel
        if self._worker:
            self._worker.join(timeout=3.0)

    def compute_prosody(self, emotion: Optional[str] = None,
                        category: Optional[str] = None,
                        intensity: Optional[float] = None) -> ProsodyParams:
        """
        Compute prosody parameters from emotional state.

        Blends default prosody with emotion-specific adjustments
        based on the emotion_influence config parameter.
        """
        emotion = emotion or self._current_emotion
        category = category or self._current_category
        intensity = intensity if intensity is not None else self._current_intensity

        # Start with defaults
        pitch = 0.0
        rate = self.cfg.default_rate
        volume = self.cfg.default_volume
        emphasis = 0.5
        warmth = 0.5
        energy = 0.5

        # Get emotion-specific prosody
        if emotion in _EMOTION_SPECIFIC:
            ep = _EMOTION_SPECIFIC[emotion]
        elif category in _EMOTION_PROSODY:
            ep = _EMOTION_PROSODY[category]
        else:
            ep = None

        if ep:
            inf = self.cfg.emotion_influence * min(1.0, intensity * 2.0)
            pitch    = ep[0] * inf
            rate     = rate  + (ep[1] - 1.0) * inf
            volume   = volume + (ep[2] - volume) * inf
            emphasis = emphasis + (ep[3] - emphasis) * inf
            warmth   = warmth + (ep[4] - warmth) * inf
            energy   = energy + (ep[5] - energy) * inf

        # Clamp
        rate = max(0.5, min(2.0, rate))
        volume = max(0.0, min(1.0, volume))
        emphasis = max(0.0, min(1.0, emphasis))
        warmth = max(0.0, min(1.0, warmth))
        energy = max(0.0, min(1.0, energy))
        pitch = max(-50.0, min(50.0, pitch))

        return ProsodyParams(
            pitch=round(pitch, 1),
            rate=round(rate, 2),
            volume=round(volume, 2),
            emphasis=round(emphasis, 2),
            warmth=round(warmth, 2),
            energy=round(energy, 2),
            emotion_tag=emotion,
        )

    def generate_visemes(self, text: str, duration_ms: float) -> List[VisemeData]:
        """
        Generate viseme timing data for avatar animation.

        Uses native C++ phoneme timing estimation, then maps
        characters to approximate viseme categories.
        """
        if not text or duration_ms <= 0:
            return []

        # Get character-level timing from C++
        timings = estimate_phoneme_timing(len(text), duration_ms)
        visemes: List[VisemeData] = []

        for i, (start, end) in enumerate(timings):
            ch = text[i].lower() if i < len(text) else " "
            vid, vname, weight = self._char_to_viseme(ch)
            visemes.append(VisemeData(
                viseme_id=vid,
                viseme_name=vname,
                start_ms=round(start, 1),
                end_ms=round(end, 1),
                weight=weight,
            ))

        return visemes

    def get_stats(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "muted": self._muted,
            "queue_size": self._queue.qsize(),
            "total_speeches": self._total_speeches,
            "total_duration_ms": round(self._total_duration_ms, 1),
            "current_emotion": self._current_emotion,
            "current_prosody": self.compute_prosody().__dict__,
        }

    # ── Worker Thread ────────────────────────────────────────

    def _start_worker(self) -> None:
        self._running = True
        self._worker = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="tts-worker",
        )
        self._worker.start()

    def _worker_loop(self) -> None:
        """Process speech queue."""
        while self._running:
            try:
                text = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if text is None:  # Shutdown sentinel
                break

            if self._muted:
                continue

            try:
                self._process_speech(text)
            except Exception as e:
                logger.error("TTS synthesis error: %s", e)
                self._set_state(TTSState.ERROR)
                self._publish("tts_error", {"error": str(e)})
                time.sleep(0.5)
                self._set_state(TTSState.IDLE)

    def _process_speech(self, text: str) -> None:
        """Full speech pipeline: filter → prosody → synthesize → visemes → play."""
        # 1. Safety filter
        filtered_text = text
        if self._safety_filter:
            try:
                result = self._safety_filter.filter_output(text)
                filtered_text = result.text
                if result.action == "block":
                    logger.info("TTS blocked by safety filter")
                    self._publish("tts_filtered", {
                        "action": "block",
                        "original": text,
                    })
                    return
                elif result.action == "rewrite":
                    logger.info("TTS rewritten by safety filter")
                    self._publish("tts_filtered", {
                        "action": "rewrite",
                        "original": text,
                        "rewritten": filtered_text,
                    })
            except Exception as e:
                logger.warning("Safety filter error, proceeding: %s", e)

        # 2. Compute prosody from emotion
        prosody = self.compute_prosody()

        # 3. Set speaking state
        self._set_state(TTSState.SPEAKING)
        self._publish("tts_speaking", {
            "text": filtered_text,
            "prosody": prosody.__dict__,
        })

        # 4. Synthesize
        if self._synthesize_fn:
            result = self._synthesize_fn(filtered_text, prosody)
        else:
            # Estimate duration: ~150ms per character at rate 1.0
            est_duration = len(filtered_text) * 150.0 / prosody.rate
            result = SpeechResult(
                text=filtered_text,
                duration_ms=est_duration,
                prosody=prosody,
                sample_rate=self.cfg.sample_rate,
            )

        # 5. Generate visemes
        if self.cfg.enable_visemes and result.duration_ms > 0:
            result.visemes = self.generate_visemes(filtered_text, result.duration_ms)
            self._publish("tts_visemes", {
                "visemes": [
                    {
                        "id": v.viseme_id,
                        "name": v.viseme_name,
                        "start_ms": v.start_ms,
                        "end_ms": v.end_ms,
                        "weight": v.weight,
                    }
                    for v in result.visemes
                ],
                "duration_ms": result.duration_ms,
            })

        # 6. Play audio
        if self._playback_fn and result.audio_data:
            self._playback_fn(result)

        # 7. Update stats
        self._total_speeches += 1
        self._total_duration_ms += result.duration_ms

        # 8. Publish completion
        self._publish("tts_speech_done", {
            "text": filtered_text,
            "duration_ms": result.duration_ms,
            "emotion": prosody.emotion_tag,
        })

        # 9. Simulate speech duration if no playback
        if not self._playback_fn and result.duration_ms > 0:
            time.sleep(result.duration_ms / 1000.0)

        self._set_state(TTSState.IDLE)

    # ── Character → Viseme ───────────────────────────────────

    @staticmethod
    def _char_to_viseme(ch: str) -> Tuple[int, str, float]:
        """
        Map a character to an approximate viseme.
        Returns (viseme_id, viseme_name, weight).
        """
        if ch in " \t\n\r.,!?;:-":
            return 0, "silence", 0.0

        # Consonant approximations
        _MAP = {
            "p": (1, 1.0), "b": (1, 1.0), "m": (1, 0.8),
            "f": (2, 0.9), "v": (2, 0.9),
            "t": (4, 0.7), "d": (4, 0.7), "s": (4, 0.6),
            "z": (4, 0.6), "n": (4, 0.5), "l": (4, 0.6),
            "r": (12, 0.5), "w": (12, 0.7), "y": (12, 0.4),
            "k": (6, 0.6), "g": (6, 0.6), "h": (6, 0.3),
            "j": (5, 0.7), "c": (6, 0.5), "q": (6, 0.6),
            "x": (4, 0.5),
        }

        if ch in _MAP:
            vid, weight = _MAP[ch]
            return vid, VISEME_NAMES[vid], weight

        # Vowels
        _VOWELS = {
            "a": (7, 1.0), "e": (8, 0.8), "i": (9, 0.7),
            "o": (10, 0.9), "u": (10, 0.8),
        }
        if ch in _VOWELS:
            vid, weight = _VOWELS[ch]
            return vid, VISEME_NAMES[vid], weight

        # Unknown
        return 0, "silence", 0.0

    # ── Event Handlers ───────────────────────────────────────

    def _on_emotion_changed(self, data: Dict[str, Any]) -> None:
        self._current_emotion = data.get("dominant", "neutral")
        self._current_intensity = data.get("dominant_intensity", 0.0)
        self._current_category = data.get("dominant_category", "peace")

    def _on_toggle(self, data: Dict[str, Any]) -> None:
        if data.get("enabled"):
            self.unmute()
        else:
            self.mute()

    def _on_mute(self, data: Dict[str, Any]) -> None:
        if data.get("muted"):
            self.mute()
        else:
            self.unmute()

    # ── Helpers ──────────────────────────────────────────────

    def _publish(self, event: str, data: Dict[str, Any]) -> None:
        if self.bus:
            try:
                self.bus.publish(event, data)
            except Exception:
                pass

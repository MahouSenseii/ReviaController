"""
core/vision/vision_engine.py — Vision Engine

State machine:
    IDLE ──(enable)──▶ ACTIVE ──(frame)──▶ ANALYZING ──(done)──▶ ACTIVE
      ▲                   ▲                                │
      │                   └────────────────────────────────┘
      └──(disable)────────┘

Features:
  - Screenshot capture and camera input
  - C++ accelerated image preprocessing (grayscale, threshold, resize)
  - Blur detection via Laplacian variance
  - Change detection between consecutive frames
  - OCR text extraction (pluggable backend)
  - Object/scene description (pluggable LLM vision backend)
  - Emotional reaction triggers based on visual events
  - Thread-safe async processing
"""

from __future__ import annotations

import enum
import time
import threading
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..native_bridge import (
    image_to_gray,
    image_adaptive_threshold,
    image_sharpness,
    HAS_NATIVE,
)

logger = logging.getLogger(__name__)


# ── Data Types ───────────────────────────────────────────────

class VisionState(enum.Enum):
    IDLE      = "idle"
    ACTIVE    = "active"
    ANALYZING = "analyzing"
    ERROR     = "error"


@dataclass
class VisionFrame:
    """A captured image frame."""
    rgb_data: bytes = b""       # Raw RGB bytes
    width: int = 0
    height: int = 0
    source: str = "unknown"     # "screenshot", "webcam", "clipboard"
    timestamp: float = field(default_factory=time.time)
    gray_data: bytes = b""      # Grayscale (computed)
    sharpness: float = 0.0      # Blur metric


@dataclass
class VisionAnalysis:
    """Analysis result for a vision frame."""
    frame: VisionFrame = field(default_factory=VisionFrame)
    ocr_text: str = ""
    description: str = ""
    objects: List[str] = field(default_factory=list)
    change_score: float = 0.0       # 0-1, how different from previous
    is_significant_change: bool = False
    is_blurry: bool = False
    emotional_trigger: Optional[str] = None  # Emotion hint for EmotionEngine
    trigger_valence: float = 0.0     # -1 to +1
    trigger_arousal: float = 0.0     # 0 to 1
    timestamp: float = field(default_factory=time.time)


@dataclass
class VisionConfig:
    """Configuration for Vision engine."""
    capture_interval_ms: int = 2000   # Time between captures
    source: str = "screenshot"        # screenshot, webcam, clipboard
    min_change_threshold: float = 0.15  # Min change to trigger analysis
    blur_threshold: float = 100.0     # Below = blurry (Laplacian variance)
    max_ocr_area: int = 4_000_000     # Max pixels for OCR (2000x2000)
    enable_ocr: bool = True
    enable_description: bool = True
    enable_change_detect: bool = True
    enable_emotion_triggers: bool = True
    ocr_preprocess: bool = True       # Adaptive threshold before OCR
    ocr_block_size: int = 15
    ocr_c: int = 5


# ── Vision Engine ────────────────────────────────────────────

class VisionEngine:
    """
    Vision engine with C++ accelerated preprocessing and
    pluggable analysis backends.

    Pipeline:
      Capture ──▶ Preprocess(C++) ──▶ Change Detect ──▶ Analysis ──▶ Events
                      │                                      │
                  grayscale                          OCR / Describe
                  threshold                          Object Detect
                  sharpness                          Emotion Trigger
    """

    def __init__(
        self,
        event_bus: Any = None,
        config: Optional[VisionConfig] = None,
    ):
        self.bus = event_bus
        self.cfg = config or VisionConfig()

        # State
        self._state = VisionState.IDLE
        self._state_lock = threading.Lock()

        # Capture thread
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._enabled = False

        # Previous frame for change detection
        self._prev_gray: Optional[bytes] = None
        self._prev_width: int = 0
        self._prev_height: int = 0

        # Pluggable backends
        self._capture_fn: Optional[Callable[[], Optional[VisionFrame]]] = None
        self._ocr_fn: Optional[Callable[[bytes, int, int], str]] = None
        self._describe_fn: Optional[Callable[[bytes, int, int, str], str]] = None

        # Stats
        self._total_frames = 0
        self._total_analyses = 0
        self._last_analysis: Optional[VisionAnalysis] = None

        # Subscribe to events
        if self.bus:
            self.bus.subscribe("vision_toggle", self._on_toggle)
            self.bus.subscribe("voice_vision_changed", self._on_config_changed)
            self.bus.subscribe("vision_capture_now", self._on_capture_now)

    # ── State Machine ────────────────────────────────────────

    @property
    def state(self) -> VisionState:
        with self._state_lock:
            return self._state

    def _set_state(self, new_state: VisionState) -> None:
        with self._state_lock:
            old = self._state
            self._state = new_state
        if old != new_state:
            logger.debug("Vision state: %s -> %s", old.value, new_state.value)
            self._publish("vision_state_changed", {
                "state": new_state.value,
                "previous": old.value,
            })

    # ── Public API ───────────────────────────────────────────

    def set_capture_source(self, fn: Callable[[], Optional[VisionFrame]]) -> None:
        """Set the capture function. Returns VisionFrame or None."""
        self._capture_fn = fn

    def set_ocr_backend(self, fn: Callable[[bytes, int, int], str]) -> None:
        """
        Set OCR backend.
        fn(gray_bytes, width, height) -> extracted text
        """
        self._ocr_fn = fn

    def set_describe_backend(self, fn: Callable[[bytes, int, int, str], str]) -> None:
        """
        Set image description backend (e.g., LLM vision).
        fn(rgb_bytes, width, height, prompt) -> description text
        """
        self._describe_fn = fn

    def enable(self) -> None:
        """Start vision capture loop."""
        if self._enabled:
            return
        self._enabled = True
        self._running = True
        self._set_state(VisionState.ACTIVE)
        self._thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="vision-capture",
        )
        self._thread.start()
        logger.info("Vision engine enabled (source=%s)", self.cfg.source)
        self._publish("vision_enabled", {"enabled": True, "source": self.cfg.source})

    def disable(self) -> None:
        """Stop vision capture."""
        self._enabled = False
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        self._set_state(VisionState.IDLE)
        self._prev_gray = None
        logger.info("Vision engine disabled")
        self._publish("vision_enabled", {"enabled": False})

    def process_frame(self, frame: VisionFrame) -> VisionAnalysis:
        """
        Process a single frame through the full analysis pipeline.
        Can be called directly (e.g., from clipboard paste).
        """
        self._set_state(VisionState.ANALYZING)
        self._total_frames += 1

        analysis = VisionAnalysis(frame=frame)

        try:
            # 1. Preprocess: convert to grayscale using C++
            if frame.rgb_data and frame.width > 0 and frame.height > 0:
                frame.gray_data = image_to_gray(
                    frame.rgb_data, frame.width, frame.height
                )

                # 2. Compute sharpness (blur detection)
                frame.sharpness = image_sharpness(
                    frame.gray_data, frame.width, frame.height
                )
                analysis.is_blurry = frame.sharpness < self.cfg.blur_threshold

                if analysis.is_blurry:
                    logger.debug("Frame blurry (sharpness=%.1f)", frame.sharpness)

                # 3. Change detection
                if self.cfg.enable_change_detect and self._prev_gray:
                    analysis.change_score = self._compute_change(
                        frame.gray_data, frame.width, frame.height
                    )
                    analysis.is_significant_change = (
                        analysis.change_score >= self.cfg.min_change_threshold
                    )

                # Store for next comparison
                self._prev_gray = frame.gray_data
                self._prev_width = frame.width
                self._prev_height = frame.height

                # 4. Only run expensive analysis on significant changes or first frame
                should_analyze = (
                    self._total_frames == 1
                    or analysis.is_significant_change
                    or not self.cfg.enable_change_detect
                )

                if should_analyze and not analysis.is_blurry:
                    # 5. OCR
                    if self.cfg.enable_ocr and self._ocr_fn:
                        ocr_input = frame.gray_data
                        if self.cfg.ocr_preprocess:
                            ocr_input = image_adaptive_threshold(
                                frame.gray_data,
                                frame.width,
                                frame.height,
                                self.cfg.ocr_block_size,
                                self.cfg.ocr_c,
                            )
                        try:
                            analysis.ocr_text = self._ocr_fn(
                                ocr_input, frame.width, frame.height
                            )
                        except Exception as e:
                            logger.warning("OCR failed: %s", e)

                    # 6. Image description
                    if self.cfg.enable_description and self._describe_fn:
                        try:
                            analysis.description = self._describe_fn(
                                frame.rgb_data, frame.width, frame.height,
                                "Describe what you see concisely.",
                            )
                        except Exception as e:
                            logger.warning("Vision describe failed: %s", e)

                    # 7. Emotion triggers from visual content
                    if self.cfg.enable_emotion_triggers:
                        self._detect_emotion_trigger(analysis)

                    self._total_analyses += 1

        except Exception as e:
            logger.error("Vision frame processing error: %s", e)
            self._set_state(VisionState.ERROR)
            self._publish("vision_error", {"error": str(e)})
            time.sleep(0.5)

        # Publish analysis result
        self._last_analysis = analysis
        self._publish("vision_analysis", {
            "ocr_text": analysis.ocr_text,
            "description": analysis.description,
            "objects": analysis.objects,
            "change_score": round(analysis.change_score, 3),
            "is_significant": analysis.is_significant_change,
            "is_blurry": analysis.is_blurry,
            "sharpness": round(frame.sharpness, 1),
            "emotional_trigger": analysis.emotional_trigger,
            "source": frame.source,
        })

        # Publish emotion trigger if detected
        if analysis.emotional_trigger and self.bus:
            self._publish("chat_stimulus", {
                "valence": analysis.trigger_valence,
                "arousal": analysis.trigger_arousal,
                "novelty": min(1.0, analysis.change_score * 2),
                "engagement": 0.6,
                "emotion_hint": analysis.emotional_trigger,
            })

        if self._enabled:
            self._set_state(VisionState.ACTIVE)

        return analysis

    def get_stats(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "enabled": self._enabled,
            "source": self.cfg.source,
            "total_frames": self._total_frames,
            "total_analyses": self._total_analyses,
            "has_native": HAS_NATIVE,
            "last_change_score": (
                round(self._last_analysis.change_score, 3)
                if self._last_analysis else 0.0
            ),
        }

    # ── Internal ─────────────────────────────────────────────

    def _capture_loop(self) -> None:
        """Main capture loop on dedicated thread."""
        interval_sec = self.cfg.capture_interval_ms / 1000.0
        while self._running:
            try:
                if self._capture_fn:
                    frame = self._capture_fn()
                    if frame:
                        self.process_frame(frame)
                time.sleep(interval_sec)
            except Exception as e:
                logger.error("Vision capture error: %s", e)
                self._set_state(VisionState.ERROR)
                self._publish("vision_error", {"error": str(e)})
                time.sleep(2.0)
                if self._running:
                    self._set_state(VisionState.ACTIVE)

    def _compute_change(self, gray: bytes, width: int, height: int) -> float:
        """
        Compute change score between current and previous frame.
        Uses mean absolute difference of pixel values.
        Returns 0.0 (identical) to 1.0 (completely different).
        """
        if (not self._prev_gray
                or self._prev_width != width
                or self._prev_height != height):
            return 1.0  # First frame or size changed = max change

        total = len(gray)
        if total == 0:
            return 0.0

        # Mean absolute difference normalised to [0,1]
        diff_sum = 0
        for i in range(total):
            diff_sum += abs(gray[i] - self._prev_gray[i])

        return diff_sum / (total * 255.0)

    def _detect_emotion_trigger(self, analysis: VisionAnalysis) -> None:
        """
        Analyse visual content for emotion-triggering patterns.
        Sets emotional_trigger, trigger_valence, trigger_arousal on analysis.
        """
        text = (analysis.ocr_text + " " + analysis.description).lower()
        if not text.strip():
            return

        # Pattern matching for emotional triggers
        triggers = [
            # (keywords, emotion, valence, arousal)
            (["error", "exception", "crash", "fatal", "failed"],
             "worried", -0.4, 0.6),
            (["success", "passed", "complete", "done", "deployed"],
             "happy", 0.7, 0.5),
            (["warning", "deprecated", "timeout"],
             "concerned", -0.2, 0.4),
            (["new message", "notification", "alert"],
             "curious", 0.2, 0.5),
            (["loading", "progress", "building", "compiling"],
             "interested", 0.1, 0.3),
            (["help", "question", "how to"],
             "engaged", 0.3, 0.4),
            (["delete", "remove", "destroy", "drop"],
             "nervous", -0.3, 0.5),
            (["beautiful", "amazing", "awesome", "great"],
             "joyful", 0.8, 0.6),
        ]

        best_match = None
        best_count = 0

        for keywords, emotion, val, aro in triggers:
            count = sum(1 for kw in keywords if kw in text)
            if count > best_count:
                best_count = count
                best_match = (emotion, val, aro)

        if best_match:
            analysis.emotional_trigger = best_match[0]
            analysis.trigger_valence = best_match[1]
            analysis.trigger_arousal = best_match[2]

        # High change = surprise/novelty
        if analysis.change_score > 0.5:
            if not analysis.emotional_trigger:
                analysis.emotional_trigger = "surprised"
                analysis.trigger_valence = 0.1
                analysis.trigger_arousal = 0.7

    # ── Event Handlers ───────────────────────────────────────

    def _on_toggle(self, data: Dict[str, Any]) -> None:
        if data.get("enabled"):
            self.enable()
        else:
            self.disable()

    def _on_config_changed(self, data: Dict[str, Any]) -> None:
        key = data.get("key", "")
        val = data.get("value")
        if key == "vision.source" and isinstance(val, str):
            self.cfg.source = val.lower().replace(" ", "_")
        elif key == "vision.interval_ms" and isinstance(val, (int, float)):
            self.cfg.capture_interval_ms = int(val)

    def _on_capture_now(self, data: Dict[str, Any]) -> None:
        """Trigger immediate capture outside the regular interval."""
        if self._capture_fn and self._enabled:
            frame = self._capture_fn()
            if frame:
                threading.Thread(
                    target=self.process_frame,
                    args=(frame,),
                    daemon=True,
                ).start()

    # ── Helpers ──────────────────────────────────────────────

    def _publish(self, event: str, data: Dict[str, Any]) -> None:
        if self.bus:
            try:
                self.bus.publish(event, data)
            except Exception:
                pass

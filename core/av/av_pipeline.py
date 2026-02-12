"""
core/av/av_pipeline.py — Audio-Visual Pipeline Orchestrator

Connects STT → Safety Filter → LLM → Safety Filter → TTS → Viseme → Avatar
with Vision input feeding into both LLM context and Emotion triggers.

Data Flow:
    ┌──────────┐    ┌────────────┐    ┌──────────────┐    ┌───────────┐
    │   STT    │───▶│  Safety    │───▶│     LLM      │───▶│  Safety   │
    │ (listen) │    │ (input)    │    │  (generate)  │    │ (output)  │
    └──────────┘    └────────────┘    └──────┬───────┘    └─────┬─────┘
                                             │                  │
    ┌──────────┐              ┌──────────┐   │           ┌──────▼──────┐
    │  Vision  │─────────────▶│ Context  │───┘           │    TTS      │
    │ (see)    │   emotion    │ Builder  │               │  (speak)    │
    └──────────┘   trigger    └──────────┘               └──────┬──────┘
         │                         ▲                            │
         │         ┌───────────┐   │                     ┌──────▼──────┐
         └────────▶│  Emotion  │───┘                     │   Viseme    │
                   │  Engine   │                         │  (animate)  │
                   └───────────┘                         └─────────────┘

State Machine:
    IDLE ──(start)──▶ RUNNING ──(stop)──▶ IDLE
                         │
                    ┌────┴────┐
                    │ Per-component states tracked independently │
                    └─────────┘
"""

from __future__ import annotations

import enum
import time
import threading
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..stt import STTEngine, STTConfig, STTState, TranscriptionResult
from ..tts import TTSEngine, TTSConfig, TTSState, ProsodyParams
from ..vision import VisionEngine, VisionConfig, VisionState

logger = logging.getLogger(__name__)


# ── Data Types ───────────────────────────────────────────────

class AVState(enum.Enum):
    IDLE    = "idle"
    RUNNING = "running"
    ERROR   = "error"


@dataclass
class AVPipelineConfig:
    """Configuration for the full AV pipeline."""
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)

    auto_respond: bool = True         # Automatically send STT to LLM
    auto_speak: bool = True           # Automatically speak LLM responses
    vision_context: bool = True       # Include vision in LLM context
    filter_tts_output: bool = True    # Safety filter before TTS
    filter_stt_input: bool = True     # Safety filter on STT text
    log_pipeline: bool = True         # Log pipeline events


# ── Pipeline Orchestrator ────────────────────────────────────

class AVPipeline:
    """
    Orchestrates STT, TTS, Vision, Safety, and Emotion into a
    coherent audio-visual pipeline.
    """

    def __init__(
        self,
        event_bus: Any,
        config: Optional[AVPipelineConfig] = None,
        emotion_engine: Any = None,
        safety_filter: Any = None,
    ):
        self.bus = event_bus
        self.cfg = config or AVPipelineConfig()

        # External engine references
        self._emotion_engine = emotion_engine
        self._safety_filter = safety_filter

        # Create sub-engines
        self.stt = STTEngine(event_bus, self.cfg.stt)
        self.tts = TTSEngine(event_bus, self.cfg.tts)
        self.vision = VisionEngine(event_bus, self.cfg.vision)

        # Wire safety filter into TTS
        if self._safety_filter:
            self.tts.set_safety_filter(self._safety_filter)

        # Pipeline state
        self._state = AVState.IDLE
        self._state_lock = threading.Lock()

        # LLM send callback: fn(text, context) -> response_text
        self._llm_fn: Optional[Callable[[str, Dict[str, Any]], str]] = None

        # Latest vision context for LLM
        self._vision_context: Dict[str, Any] = {}
        self._vision_lock = threading.Lock()

        # Subscribe to pipeline events
        self.bus.subscribe("stt_transcription", self._on_stt_transcription)
        self.bus.subscribe("vision_analysis", self._on_vision_analysis)
        self.bus.subscribe("av_speak", self._on_speak_request)
        self.bus.subscribe("av_pipeline_toggle", self._on_pipeline_toggle)

    # ── State ────────────────────────────────────────────────

    @property
    def state(self) -> AVState:
        with self._state_lock:
            return self._state

    def _set_state(self, new: AVState) -> None:
        with self._state_lock:
            old = self._state
            self._state = new
        if old != new:
            self._publish("av_state_changed", {
                "state": new.value,
                "stt": self.stt.state.value,
                "tts": self.tts.state.value,
                "vision": self.vision.state.value,
            })

    # ── Public API ───────────────────────────────────────────

    def set_llm_callback(self, fn: Callable[[str, Dict[str, Any]], str]) -> None:
        """
        Set the LLM inference callback.
        fn(user_text, context_dict) -> response_text
        """
        self._llm_fn = fn

    def start(self) -> None:
        """Start all pipeline components."""
        self._set_state(AVState.RUNNING)
        logger.info("AV pipeline started")
        self._publish("av_pipeline_started", {})

    def stop(self) -> None:
        """Stop all pipeline components."""
        self.stt.disable()
        self.tts.stop()
        self.vision.disable()
        self._set_state(AVState.IDLE)
        logger.info("AV pipeline stopped")
        self._publish("av_pipeline_stopped", {})

    def shutdown(self) -> None:
        """Full shutdown."""
        self.stop()
        self.tts.shutdown()

    def enable_stt(self) -> None:
        """Enable speech-to-text."""
        self.stt.enable()

    def disable_stt(self) -> None:
        """Disable speech-to-text."""
        self.stt.disable()

    def enable_vision(self) -> None:
        """Enable vision capture."""
        self.vision.enable()

    def disable_vision(self) -> None:
        """Disable vision capture."""
        self.vision.disable()

    def speak(self, text: str, priority: bool = False) -> None:
        """Queue text for TTS."""
        self.tts.speak(text, priority=priority)

    def process_text_input(self, text: str) -> Optional[str]:
        """
        Process text through the full pipeline:
        Safety Filter (input) → LLM → Safety Filter (output) → TTS
        Returns the LLM response text.
        """
        if self.cfg.log_pipeline:
            logger.info("Pipeline input: '%s'", text[:100])

        # 1. Input safety filter
        filtered_input = text
        if self.cfg.filter_stt_input and self._safety_filter:
            try:
                result = self._safety_filter.filter_input(text)
                if result.action == "block":
                    logger.info("Input blocked by safety filter")
                    self._publish("av_input_blocked", {"text": text})
                    if self.cfg.auto_speak:
                        self.tts.speak(result.text, priority=True)
                    return result.text
                filtered_input = result.text
            except Exception as e:
                logger.warning("Input filter error: %s", e)

        # 2. Build context
        context = self._build_context()

        # 3. LLM inference
        if not self._llm_fn:
            logger.warning("No LLM callback set")
            return None

        try:
            response = self._llm_fn(filtered_input, context)
        except Exception as e:
            logger.error("LLM inference error: %s", e)
            self._publish("av_llm_error", {"error": str(e)})
            return None

        if self.cfg.log_pipeline:
            logger.info("Pipeline output: '%s'", response[:100] if response else "")

        # 4. Output safety filter + TTS handled by TTS engine
        if response and self.cfg.auto_speak:
            self.tts.speak(response)

        return response

    def get_component_states(self) -> Dict[str, str]:
        """Return all component states for UI."""
        return {
            "pipeline": self.state.value,
            "stt": self.stt.state.value,
            "tts": self.tts.state.value,
            "vision": self.vision.state.value,
        }

    def get_full_stats(self) -> Dict[str, Any]:
        """Return comprehensive stats from all components."""
        return {
            "pipeline": self.state.value,
            "stt": self.stt.get_stats(),
            "tts": self.tts.get_stats(),
            "vision": self.vision.get_stats(),
        }

    # ── Context Builder ──────────────────────────────────────

    def _build_context(self) -> Dict[str, Any]:
        """Build context dict for LLM from all sources."""
        ctx: Dict[str, Any] = {}

        # Emotion context
        if self._emotion_engine:
            try:
                ctx["emotion"] = self._emotion_engine.get_llm_context()
            except Exception:
                pass

        # Vision context
        if self.cfg.vision_context:
            with self._vision_lock:
                if self._vision_context:
                    ctx["vision"] = dict(self._vision_context)

        # TTS prosody (so LLM knows how it will sound)
        ctx["current_prosody"] = self.tts.compute_prosody().__dict__

        return ctx

    # ── Event Handlers ───────────────────────────────────────

    def _on_stt_transcription(self, data: Dict[str, Any]) -> None:
        """Handle transcription from STT."""
        text = data.get("text", "")
        if not text:
            return

        self._publish("av_transcription_received", data)

        if self.cfg.auto_respond:
            # Process in background thread to not block EventBus
            threading.Thread(
                target=self.process_text_input,
                args=(text,),
                daemon=True,
                name="av-process",
            ).start()

    def _on_vision_analysis(self, data: Dict[str, Any]) -> None:
        """Cache latest vision analysis for LLM context."""
        with self._vision_lock:
            self._vision_context = {
                "ocr_text": data.get("ocr_text", ""),
                "description": data.get("description", ""),
                "objects": data.get("objects", []),
                "source": data.get("source", ""),
            }

    def _on_speak_request(self, data: Dict[str, Any]) -> None:
        """Handle external speak requests."""
        text = data.get("text", "")
        priority = data.get("priority", False)
        if text:
            self.tts.speak(text, priority=priority)

    def _on_pipeline_toggle(self, data: Dict[str, Any]) -> None:
        """Toggle entire pipeline."""
        if data.get("enabled"):
            self.start()
        else:
            self.stop()

    # ── Helpers ──────────────────────────────────────────────

    def _publish(self, event: str, data: Dict[str, Any]) -> None:
        try:
            self.bus.publish(event, data)
        except Exception:
            pass

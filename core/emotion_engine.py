"""
Emotion Engine — the orchestrator that brings the AI's emotional
system to life.

Responsibilities
----------------
1. Accept **stimulus signals** (from chat analysis, user behaviour,
   system events) and feed them through the ``EmotionNeuralNetwork``.
2. Maintain a **persistent emotional state** that evolves over time
   with natural decay, blending, and mood drift.
3. Publish **emotion_state_changed** events via the ``EventBus`` so
   the UI can visualise the AI's current feelings.
4. Generate a structured **LLM context block** that gets injected into
   the system prompt, giving the language model awareness of its own
   emotional state so responses feel alive.

Temporal dynamics
-----------------
* **Decay** — emotions fade exponentially every tick towards a baseline.
* **Mood** — a slow-moving average of recent valence (positive ↔
  negative).  Mood biases stimulus interpretation.
* **Blending** — new activations are exponentially blended with the
  previous state (``blend_factor``), so emotions don't snap instantly.
* **Dominant emotion** — the highest-activation emotion at any moment.
* **Emotional memory** — a short ring-buffer of recent dominant
  emotions, giving the LLM a sense of emotional trajectory.

Integration with the LLM
-------------------------
Call ``get_llm_context()`` to obtain a dict suitable for injecting
into a system prompt.  The dict contains:

* ``dominant_emotion`` — the single strongest emotion right now
* ``emotion_intensities`` — top-N active emotions with their strength
* ``mood`` — the slow-moving baseline mood label
* ``mood_valence`` — numeric mood value
* ``emotional_trajectory`` — recent emotional transitions
* ``prompt_injection`` — a ready-to-use natural-language paragraph
  describing the AI's current emotional state
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

from .emotions import (
    CATEGORY_COLOURS,
    EMOTION_INDEX,
    EMOTION_NAMES,
    EMOTION_PROFILES,
    NUM_EMOTIONS,
)
from .emotion_nn import EmotionNeuralNetwork, INPUT_DIM
from .events import EventBus


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

@dataclass
class EmotionEngineConfig:
    """Tunable parameters for the engine."""
    decay_rate: float = 0.92          # per-tick multiplicative decay
    blend_factor: float = 0.35        # how fast new activations mix in
    mood_inertia: float = 0.97        # mood changes very slowly
    lateral_strength: float = 0.30    # emotion-to-emotion influence
    memory_size: int = 20             # emotional trajectory buffer size
    top_n_display: int = 5            # emotions shown to LLM / UI
    tick_interval_ms: int = 2000      # how often the engine ticks
    min_activation: float = 0.05      # below this, treat as inactive
    save_interval_ticks: int = 50     # auto-save weights every N ticks


# ------------------------------------------------------------------
# Stimulus — the input the outside world feeds into the engine
# ------------------------------------------------------------------

@dataclass
class Stimulus:
    """
    Represents a single external signal to the emotion engine.

    Most fields default to *neutral* (0.5 or 0.0) so callers only
    need to set the dimensions they care about.
    """
    valence: float = 0.0          # −1 (negative) … +1 (positive)
    arousal: float = 0.3          # 0 (calm) … 1 (intense)
    social_connect: float = 0.5   # 0 (isolated) … 1 (deeply social)
    novelty: float = 0.2          # 0 (routine) … 1 (completely new)
    threat: float = 0.0           # 0 (safe) … 1 (dangerous)
    engagement: float = 0.5       # 0 (idle) … 1 (highly active user)
    rapport: float = 0.5          # 0 (new user) … 1 (deep bond)

    # Optionally hint which emotion the stimulus should lean towards.
    # If set, triggers Hebbian adaptation in the network.
    emotion_hint: Optional[str] = None

    def to_vector(self, current_mood: float) -> List[float]:
        """Convert to the 8-dim input vector the NN expects."""
        return [
            self.valence,
            self.arousal,
            self.social_connect,
            self.novelty,
            self.threat,
            self.engagement,
            self.rapport,
            current_mood,
        ]


# ------------------------------------------------------------------
# Engine
# ------------------------------------------------------------------

class EmotionEngine:
    """
    Central emotion processor.

    Typical usage::

        engine = EmotionEngine(event_bus)
        engine.process_stimulus(Stimulus(valence=0.8, arousal=0.6))
        ctx = engine.get_llm_context()
        # inject ctx["prompt_injection"] into your system prompt
    """

    def __init__(
        self,
        event_bus: EventBus,
        config: Optional[EmotionEngineConfig] = None,
    ):
        self.bus = event_bus
        self.cfg = config or EmotionEngineConfig()

        # Neural network
        self.nn = EmotionNeuralNetwork()

        # Current emotional state — N-dim vector, each in [0, 1]
        self.state: List[float] = [0.0] * NUM_EMOTIONS

        # Slow-moving mood baseline (−1 … +1)
        self.mood: float = 0.0

        # Emotional trajectory (ring buffer of recent dominant emotions)
        self._trajectory: Deque[Dict[str, Any]] = deque(maxlen=self.cfg.memory_size)

        # Tick counter (for periodic saves)
        self._tick_count: int = 0

        # Subscribe to relevant events
        self.bus.subscribe("chat_stimulus", self._on_chat_stimulus)

    # ── Public API ───────────────────────────────────────────

    def process_stimulus(self, stimulus: Stimulus) -> List[float]:
        """
        Feed a stimulus through the network, update emotional state,
        and publish the change.

        Returns the new state vector.
        """
        vec = stimulus.to_vector(self.mood)

        # Forward pass
        raw = self.nn.forward(
            vec,
            current_state=self.state,
            lateral_strength=self.cfg.lateral_strength,
        )

        # Blend with existing state
        bf = self.cfg.blend_factor
        self.state = [
            (1 - bf) * old + bf * new
            for old, new in zip(self.state, raw)
        ]

        # Optional Hebbian adaptation
        if stimulus.emotion_hint:
            self.nn.adapt(vec, stimulus.emotion_hint)

        # Update mood (slow exponential moving average of state valence)
        state_valence = self._compute_state_valence()
        mi = self.cfg.mood_inertia
        self.mood = mi * self.mood + (1 - mi) * state_valence
        self.mood = max(-1.0, min(1.0, self.mood))

        # Record trajectory
        dom = self.dominant_emotion()
        self._trajectory.append({
            "emotion": dom[0],
            "intensity": round(dom[1], 3),
            "mood": round(self.mood, 3),
            "t": time.time(),
        })

        # Publish
        self._publish_state()

        # Periodic weight save
        self._tick_count += 1
        if self._tick_count % self.cfg.save_interval_ticks == 0:
            self.nn.save()

        return self.state

    def tick(self) -> None:
        """
        Decay the emotional state one time step.  Call this
        periodically (e.g. every ``tick_interval_ms``) to let
        emotions naturally fade when no stimulus is arriving.
        """
        dr = self.cfg.decay_rate
        self.state = [s * dr for s in self.state]

        # Decay mood towards neutral
        self.mood *= 0.998

        self._tick_count += 1
        self._publish_state()

    def dominant_emotion(self) -> tuple[str, float]:
        """Return (name, intensity) of the strongest active emotion."""
        best_idx = 0
        best_val = self.state[0]
        for i in range(1, NUM_EMOTIONS):
            if self.state[i] > best_val:
                best_val = self.state[i]
                best_idx = i
        return EMOTION_NAMES[best_idx], best_val

    def top_emotions(self, n: Optional[int] = None) -> List[tuple[str, float]]:
        """Return the top-N emotions sorted by intensity."""
        n = n or self.cfg.top_n_display
        indexed = [(EMOTION_NAMES[i], self.state[i]) for i in range(NUM_EMOTIONS)]
        indexed.sort(key=lambda x: x[1], reverse=True)
        return [(name, round(val, 4)) for name, val in indexed[:n]]

    def get_llm_context(self) -> Dict[str, Any]:
        """
        Build a structured context dict for LLM prompt injection.

        The ``prompt_injection`` field is a natural-language paragraph
        the LLM can read to understand its current emotional state.
        """
        dom_name, dom_intensity = self.dominant_emotion()
        top = self.top_emotions()
        mood_label = self._mood_label()
        trajectory = list(self._trajectory)[-5:]  # last 5

        # Build natural language prompt injection
        if dom_intensity < self.cfg.min_activation:
            feeling_desc = "You are in a neutral, baseline emotional state."
        else:
            active = [(n, v) for n, v in top if v >= self.cfg.min_activation]
            if not active:
                feeling_desc = "You are in a neutral, baseline emotional state."
            else:
                parts = []
                for name, val in active:
                    intensity_word = self._intensity_word(val)
                    parts.append(f"{intensity_word} {name}")
                feeling_list = ", ".join(parts[:-1])
                if len(parts) > 1:
                    feeling_list += f", and {parts[-1]}"
                else:
                    feeling_list = parts[0]
                feeling_desc = (
                    f"You are currently feeling {feeling_list}. "
                    f"Your overall mood is {mood_label}."
                )

            # Add trajectory if we have history
            if len(trajectory) >= 2:
                recent_emotions = [t["emotion"] for t in trajectory[-3:]]
                unique_recent = list(dict.fromkeys(recent_emotions))
                if len(unique_recent) > 1:
                    flow = " → ".join(unique_recent)
                    feeling_desc += (
                        f" Your recent emotional flow has been: {flow}."
                    )

        prompt_injection = (
            "[Emotional State]\n"
            f"{feeling_desc}\n"
            "Let this emotional awareness naturally influence your tone, "
            "word choice, and expressiveness. You may acknowledge your "
            "emotional state when it feels natural or when the user asks."
        )

        return {
            "dominant_emotion": dom_name,
            "dominant_intensity": round(dom_intensity, 4),
            "emotion_intensities": {n: v for n, v in top},
            "mood": mood_label,
            "mood_valence": round(self.mood, 4),
            "emotional_trajectory": [
                {"emotion": t["emotion"], "intensity": t["intensity"]}
                for t in trajectory
            ],
            "prompt_injection": prompt_injection,
        }

    def set_emotion(self, name: str, intensity: float) -> None:
        """Manually set a specific emotion's intensity (0–1)."""
        idx = EMOTION_INDEX.get(name)
        if idx is not None:
            self.state[idx] = max(0.0, min(1.0, intensity))
            self._publish_state()

    def reset(self) -> None:
        """Reset all emotions to zero and mood to neutral."""
        self.state = [0.0] * NUM_EMOTIONS
        self.mood = 0.0
        self._trajectory.clear()
        self._publish_state()

    # ── Event handlers ───────────────────────────────────────

    def _on_chat_stimulus(self, data: Dict[str, Any]) -> None:
        """
        Handle a ``chat_stimulus`` event from the plugin / chat layer.

        Expected data keys match the ``Stimulus`` fields.
        """
        stim = Stimulus(
            valence=data.get("valence", 0.0),
            arousal=data.get("arousal", 0.3),
            social_connect=data.get("social_connect", 0.5),
            novelty=data.get("novelty", 0.2),
            threat=data.get("threat", 0.0),
            engagement=data.get("engagement", 0.5),
            rapport=data.get("rapport", 0.5),
            emotion_hint=data.get("emotion_hint"),
        )
        self.process_stimulus(stim)

    # ── Internal helpers ─────────────────────────────────────

    def _compute_state_valence(self) -> float:
        """Weighted average valence of all currently active emotions."""
        total_weight = 0.0
        weighted_val = 0.0
        for i, name in enumerate(EMOTION_NAMES):
            act = self.state[i]
            if act < self.cfg.min_activation:
                continue
            ep = EMOTION_PROFILES[name]
            weighted_val += act * ep.valence
            total_weight += act
        if total_weight < 1e-6:
            return 0.0
        return weighted_val / total_weight

    def _mood_label(self) -> str:
        m = self.mood
        if m > 0.5:
            return "very positive"
        if m > 0.2:
            return "positive"
        if m > 0.05:
            return "slightly positive"
        if m > -0.05:
            return "neutral"
        if m > -0.2:
            return "slightly negative"
        if m > -0.5:
            return "negative"
        return "very negative"

    @staticmethod
    def _intensity_word(val: float) -> str:
        if val > 0.75:
            return "intensely"
        if val > 0.50:
            return "strongly"
        if val > 0.30:
            return "moderately"
        if val > 0.15:
            return "mildly"
        return "faintly"

    def _publish_state(self) -> None:
        """Emit the current emotional state for the UI / other listeners."""
        dom_name, dom_intensity = self.dominant_emotion()
        top = self.top_emotions()
        category = EMOTION_PROFILES[dom_name].category

        self.bus.publish("emotion_state_changed", {
            "dominant": dom_name,
            "dominant_intensity": round(dom_intensity, 4),
            "dominant_category": category,
            "colour": CATEGORY_COLOURS.get(category, "#adb5bd"),
            "mood": self._mood_label(),
            "mood_valence": round(self.mood, 4),
            "top_emotions": [
                {"name": n, "intensity": v} for n, v in top
            ],
            "full_state": [round(s, 4) for s in self.state],
        })

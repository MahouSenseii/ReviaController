"""
Metacognition — self-monitoring and reflection for the AI.

Tracks whether responses are landing well by observing user reactions
after the AI speaks.  Compares predicted emotional outcomes to actual
outcomes, builds a confidence model, and generates self-reflection
notes that get injected into the system prompt so the AI can
course-correct in real time.

Key concepts
------------
* **Prediction** — before each response, the engine predicts what
  emotion the user will feel next.
* **Evaluation** — after the user replies, it compares prediction
  to reality and scores accuracy.
* **Reflection** — a short natural-language note about what went
  well or poorly, injected into the next prompt.
* **Confidence** — a rolling accuracy score (0–1) that tells the
  decision engine how much to trust its own emotional reads.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

from .emotion_engine import EmotionEngine
from .events import EventBus


# ------------------------------------------------------------------
# Prediction record
# ------------------------------------------------------------------

@dataclass
class Prediction:
    """A single predicted → actual emotional outcome."""
    predicted_valence: float = 0.0
    predicted_dominant: str = ""
    actual_valence: Optional[float] = None
    actual_dominant: Optional[str] = None
    accuracy: Optional[float] = None  # 0 = totally wrong, 1 = spot on
    timestamp: float = 0.0


# ------------------------------------------------------------------
# Metacognition Engine
# ------------------------------------------------------------------

class MetacognitionEngine:
    """
    Self-monitoring layer that tracks whether the AI's emotional
    reads and response strategies are actually working.
    """

    def __init__(
        self,
        event_bus: EventBus,
        emotion_engine: EmotionEngine,
    ):
        self.bus = event_bus
        self.emotion = emotion_engine

        # Prediction history
        self._predictions: Deque[Prediction] = deque(maxlen=50)
        self._pending_prediction: Optional[Prediction] = None

        # Rolling confidence score (0–1)
        self.confidence: float = 0.5

        # Reflection notes
        self._current_reflection: str = ""
        self._interaction_outcomes: List[Dict[str, Any]] = []

        # Pattern tracking
        self._strategy_outcomes: Dict[str, List[float]] = {}

        # Subscribe
        self.bus.subscribe("assistant_response", self._on_response)
        self.bus.subscribe("chat_stimulus", self._on_user_stimulus)
        self.bus.subscribe("decision_made", self._on_decision)

    # ── Public API ────────────────────────────────────────────

    def predict_outcome(self) -> None:
        """
        Record a prediction of what the user will feel after the
        AI's upcoming response, based on current emotional state.
        """
        dom_name, dom_intensity = self.emotion.dominant_emotion()
        mood = self.emotion.mood

        # Simple prediction: if mood is positive, user will stay positive
        # This gets refined as accuracy data accumulates
        predicted_valence = mood * 0.6 + dom_intensity * 0.2
        if dom_intensity < 0.05:
            predicted_valence = 0.0

        self._pending_prediction = Prediction(
            predicted_valence=predicted_valence,
            predicted_dominant=dom_name,
            timestamp=time.time(),
        )

    def evaluate(self, user_valence: float, user_dominant: str) -> float:
        """
        Compare the pending prediction to actual user reaction.

        Returns accuracy score (0–1).
        """
        if self._pending_prediction is None:
            return 0.5

        pred = self._pending_prediction

        # Valence accuracy: how close were we?
        valence_error = abs(pred.predicted_valence - user_valence)
        valence_acc = max(0.0, 1.0 - valence_error)

        # Emotion match bonus
        emotion_match = 0.2 if pred.predicted_dominant == user_dominant else 0.0

        accuracy = min(1.0, valence_acc * 0.8 + emotion_match)

        # Record
        pred.actual_valence = user_valence
        pred.actual_dominant = user_dominant
        pred.accuracy = accuracy
        self._predictions.append(pred)

        # Update rolling confidence
        self._update_confidence(accuracy)

        # Generate reflection
        self._generate_reflection(pred)

        # Clear pending
        self._pending_prediction = None

        # Publish for UI
        self.bus.publish("metacognition_update", {
            "confidence": round(self.confidence, 3),
            "last_accuracy": round(accuracy, 3),
            "prediction_count": len(self._predictions),
            "reflection": self._current_reflection,
        })

        return accuracy

    def get_reflection_block(self) -> str:
        """
        Return a prompt block with self-reflection notes.

        Injected into the system prompt to give the AI self-awareness.
        """
        if not self._current_reflection:
            return ""

        confidence_word = self._confidence_word()
        return (
            "[Self-Awareness]\n"
            f"Your confidence in reading the conversation is {confidence_word} "
            f"({self.confidence:.0%}). "
            f"{self._current_reflection}"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Return metacognition stats for UI display."""
        recent = list(self._predictions)[-10:]
        avg_accuracy = (
            sum(p.accuracy for p in recent if p.accuracy is not None)
            / max(len([p for p in recent if p.accuracy is not None]), 1)
        )
        return {
            "confidence": round(self.confidence, 3),
            "avg_accuracy": round(avg_accuracy, 3),
            "total_predictions": len(self._predictions),
            "reflection": self._current_reflection,
        }

    # ── Event handlers ────────────────────────────────────────

    def _on_response(self, data: Dict[str, Any]) -> None:
        """After AI responds, set up prediction for evaluation."""
        self.predict_outcome()

    def _on_user_stimulus(self, data: Dict[str, Any]) -> None:
        """When user speaks next, evaluate our prediction."""
        if self._pending_prediction is None:
            return

        valence = data.get("valence", 0.0)
        hint = data.get("emotion_hint", "")
        dominant = hint if hint else "neutral"

        self.evaluate(valence, dominant)

    def _on_decision(self, data: Dict[str, Any]) -> None:
        """Track which strategies lead to good outcomes."""
        reasoning = data.get("reasoning", "")
        if reasoning:
            # Store for correlation with next evaluation
            self._interaction_outcomes.append({
                "strategy": data,
                "timestamp": time.time(),
            })
            if len(self._interaction_outcomes) > 30:
                self._interaction_outcomes = self._interaction_outcomes[-30:]

    # ── Internal ──────────────────────────────────────────────

    def _update_confidence(self, accuracy: float) -> None:
        """Exponential moving average of accuracy scores."""
        inertia = 0.85
        self.confidence = inertia * self.confidence + (1 - inertia) * accuracy
        self.confidence = max(0.1, min(0.95, self.confidence))

    def _generate_reflection(self, pred: Prediction) -> None:
        """Create a natural-language reflection note."""
        if pred.accuracy is None:
            self._current_reflection = ""
            return

        if pred.accuracy > 0.7:
            self._current_reflection = (
                "Your emotional reads have been accurate recently. "
                "Continue with your current approach."
            )
        elif pred.accuracy > 0.4:
            self._current_reflection = (
                "Your emotional reads are partially accurate. "
                "Pay closer attention to the user's actual words "
                "and tone rather than assumptions."
            )
        else:
            # Check if we predicted positive but got negative
            if (pred.predicted_valence > 0 and
                    pred.actual_valence is not None and
                    pred.actual_valence < -0.2):
                self._current_reflection = (
                    "You misread the user's mood — they seem more "
                    "negative than expected. Adjust your tone to be "
                    "more attentive and empathetic."
                )
            elif (pred.predicted_valence < 0 and
                    pred.actual_valence is not None and
                    pred.actual_valence > 0.2):
                self._current_reflection = (
                    "The user is more positive than you expected. "
                    "You may have been too cautious — match their energy."
                )
            else:
                self._current_reflection = (
                    "Your emotional predictions have been off. "
                    "Focus on what the user is explicitly saying "
                    "rather than inferring mood."
                )

    def _confidence_word(self) -> str:
        if self.confidence > 0.75:
            return "high"
        if self.confidence > 0.5:
            return "moderate"
        if self.confidence > 0.3:
            return "low"
        return "very low"

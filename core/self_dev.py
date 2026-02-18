"""
Self-development — learning and growth from interactions.

This module closes the learning loop that was missing from the emotion
system.  It observes conversation outcomes, adjusts the neural network's
learning rate dynamically, tracks user preference patterns, and evolves
personality weights over time.

Growth axes
-----------
* **Emotional accuracy** — learns which stimulus patterns map to which
  emotions based on user feedback signals.
* **User preference modelling** — tracks what response styles the user
  prefers (verbose vs concise, warm vs professional, etc.).
* **Personality drift** — the AI's personality slowly evolves based on
  cumulative interaction patterns.
* **Neural network tuning** — dynamically adjusts learning rate and
  bypass strength based on prediction accuracy.

Persistence
-----------
Growth state saves to ``self_dev_state.json`` so progress is not lost
between sessions.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from .emotion_engine import EmotionEngine
from .emotion_nn import EmotionNeuralNetwork
from .events import EventBus


_STATE_PATH = Path("self_dev_state.json")


# ------------------------------------------------------------------
# User preference model
# ------------------------------------------------------------------

@dataclass
class UserPreferences:
    """Learned preferences — what style the user responds best to."""
    preferred_verbosity: float = 0.5
    preferred_warmth: float = 0.5
    preferred_assertiveness: float = 0.5
    preferred_empathy: float = 0.5
    interaction_count: int = 0

    def to_dict(self) -> Dict[str, float]:
        return {
            "preferred_verbosity": round(self.preferred_verbosity, 3),
            "preferred_warmth": round(self.preferred_warmth, 3),
            "preferred_assertiveness": round(self.preferred_assertiveness, 3),
            "preferred_empathy": round(self.preferred_empathy, 3),
            "interaction_count": self.interaction_count,
        }


# ------------------------------------------------------------------
# Growth record
# ------------------------------------------------------------------

@dataclass
class GrowthSnapshot:
    """A point-in-time snapshot of the AI's development."""
    timestamp: float = 0.0
    accuracy: float = 0.5
    learning_rate: float = 0.005
    bypass_strength: float = 0.7
    interaction_count: int = 0


# ------------------------------------------------------------------
# Self-Development Engine
# ------------------------------------------------------------------

class SelfDevelopmentEngine:
    """
    Observes interaction outcomes and evolves the AI's emotional
    intelligence over time.
    """

    def __init__(
        self,
        event_bus: EventBus,
        emotion_engine: EmotionEngine,
    ):
        self.bus = event_bus
        self.emotion = emotion_engine

        # User preference model
        self.user_prefs = UserPreferences()

        # Growth tracking
        self._growth_history: Deque[GrowthSnapshot] = deque(maxlen=100)
        self._recent_accuracies: Deque[float] = deque(maxlen=30)
        self._total_interactions: int = 0

        # Positive/negative reaction tracking for NN tuning
        self._positive_reactions: int = 0
        self._negative_reactions: int = 0
        self._reaction_window: Deque[float] = deque(maxlen=20)

        # Load saved state
        self._load_state()

        # Subscribe
        self.bus.subscribe("metacognition_update", self._on_metacognition)
        self.bus.subscribe("chat_stimulus", self._on_stimulus)
        self.bus.subscribe("decision_made", self._on_decision)

    # ── Public API ────────────────────────────────────────────

    def get_preference_hints(self) -> str:
        """
        Return a prompt block describing learned user preferences.

        Only activated after enough interactions to have signal.
        """
        if self.user_prefs.interaction_count < 5:
            return ""

        hints: list[str] = []

        if self.user_prefs.preferred_verbosity > 0.65:
            hints.append("This user prefers detailed, thorough responses.")
        elif self.user_prefs.preferred_verbosity < 0.35:
            hints.append("This user prefers short, concise responses.")

        if self.user_prefs.preferred_warmth > 0.65:
            hints.append("This user responds well to warm, friendly tone.")
        elif self.user_prefs.preferred_warmth < 0.35:
            hints.append("This user prefers a professional, direct tone.")

        if self.user_prefs.preferred_assertiveness > 0.65:
            hints.append("This user appreciates confident, decisive answers.")
        elif self.user_prefs.preferred_assertiveness < 0.35:
            hints.append("This user prefers options and gentle suggestions.")

        if not hints:
            return ""

        return (
            "[Learned Preferences]\n"
            + " ".join(hints)
        )

    def get_growth_summary(self) -> Dict[str, Any]:
        """Return growth stats for UI display."""
        nn = self.emotion.nn
        avg_acc = (
            sum(self._recent_accuracies) / max(len(self._recent_accuracies), 1)
        )
        return {
            "total_interactions": self._total_interactions,
            "learning_rate": round(nn.learning_rate, 4),
            "bypass_strength": round(nn.bypass_strength, 3),
            "avg_accuracy": round(avg_acc, 3),
            "positive_ratio": self._positive_ratio(),
            "preferences": self.user_prefs.to_dict(),
            "growth_points": len(self._growth_history),
        }

    def save_state(self) -> None:
        """Persist development state to disk."""
        data = {
            "user_prefs": self.user_prefs.to_dict(),
            "total_interactions": self._total_interactions,
            "recent_accuracies": list(self._recent_accuracies),
            "positive_reactions": self._positive_reactions,
            "negative_reactions": self._negative_reactions,
        }
        try:
            _STATE_PATH.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass

    # ── Event handlers ────────────────────────────────────────

    def _on_metacognition(self, data: Dict[str, Any]) -> None:
        """
        When metacognition evaluates a prediction, use the accuracy
        to tune the neural network and update growth stats.
        """
        accuracy = data.get("last_accuracy", 0.5)
        self._recent_accuracies.append(accuracy)
        self._total_interactions += 1

        # ── Dynamic learning rate adjustment ──────────────────
        nn = self.emotion.nn
        avg_acc = sum(self._recent_accuracies) / len(self._recent_accuracies)

        if avg_acc < 0.4:
            # Accuracy is poor — increase learning rate to adapt faster
            nn.learning_rate = min(0.05, nn.learning_rate * 1.1)
        elif avg_acc > 0.7:
            # Accuracy is good — decrease learning rate to stabilize
            nn.learning_rate = max(0.002, nn.learning_rate * 0.95)

        # ── Shift bypass strength over time ───────────────────
        # As the hidden layers learn, gradually trust them more
        if self._total_interactions > 20 and avg_acc > 0.5:
            nn.bypass_strength = max(0.3, nn.bypass_strength - 0.002)

        # ── Record growth snapshot ────────────────────────────
        self._growth_history.append(GrowthSnapshot(
            timestamp=time.time(),
            accuracy=avg_acc,
            learning_rate=nn.learning_rate,
            bypass_strength=nn.bypass_strength,
            interaction_count=self._total_interactions,
        ))

        # Publish growth stats
        self.bus.publish("self_dev_update", self.get_growth_summary())

        # Auto-save every 10 interactions
        if self._total_interactions % 10 == 0:
            self.save_state()
            nn.save()

    def _on_stimulus(self, data: Dict[str, Any]) -> None:
        """Track user sentiment for preference learning."""
        valence = data.get("valence", 0.0)
        self._reaction_window.append(valence)

        if valence > 0.2:
            self._positive_reactions += 1
        elif valence < -0.2:
            self._negative_reactions += 1

    def _on_decision(self, data: Dict[str, Any]) -> None:
        """
        Learn user preferences by correlating decision strategies
        with user reaction valence.
        """
        self.user_prefs.interaction_count += 1

        # We need at least 2 reactions to correlate
        if len(self._reaction_window) < 2:
            return

        # Use the most recent user valence as feedback signal
        recent_valence = self._reaction_window[-1]
        lr = 0.05  # preference learning rate

        # If user reacted positively, nudge preferences toward
        # the strategy that was used
        if recent_valence > 0.1:
            signal = min(recent_valence, 0.5)  # cap influence
            self.user_prefs.preferred_verbosity += lr * signal * (
                data.get("verbosity", 0.5) - self.user_prefs.preferred_verbosity
            )
            self.user_prefs.preferred_warmth += lr * signal * (
                data.get("warmth", 0.5) - self.user_prefs.preferred_warmth
            )
            self.user_prefs.preferred_assertiveness += lr * signal * (
                data.get("assertiveness", 0.5) - self.user_prefs.preferred_assertiveness
            )
            self.user_prefs.preferred_empathy += lr * signal * (
                data.get("empathy", 0.5) - self.user_prefs.preferred_empathy
            )

        # Clamp all preferences
        self.user_prefs.preferred_verbosity = _clamp(self.user_prefs.preferred_verbosity)
        self.user_prefs.preferred_warmth = _clamp(self.user_prefs.preferred_warmth)
        self.user_prefs.preferred_assertiveness = _clamp(self.user_prefs.preferred_assertiveness)
        self.user_prefs.preferred_empathy = _clamp(self.user_prefs.preferred_empathy)

    # ── Internal ──────────────────────────────────────────────

    def _positive_ratio(self) -> float:
        total = self._positive_reactions + self._negative_reactions
        if total == 0:
            return 0.5
        return round(self._positive_reactions / total, 3)

    def _load_state(self) -> None:
        """Load saved development state."""
        if not _STATE_PATH.exists():
            return
        try:
            data = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
            p = data.get("user_prefs", {})
            self.user_prefs.preferred_verbosity = p.get("preferred_verbosity", 0.5)
            self.user_prefs.preferred_warmth = p.get("preferred_warmth", 0.5)
            self.user_prefs.preferred_assertiveness = p.get("preferred_assertiveness", 0.5)
            self.user_prefs.preferred_empathy = p.get("preferred_empathy", 0.5)
            self.user_prefs.interaction_count = p.get("interaction_count", 0)
            self._total_interactions = data.get("total_interactions", 0)
            self._positive_reactions = data.get("positive_reactions", 0)
            self._negative_reactions = data.get("negative_reactions", 0)
            for a in data.get("recent_accuracies", []):
                self._recent_accuracies.append(a)
        except (json.JSONDecodeError, KeyError, OSError):
            pass


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

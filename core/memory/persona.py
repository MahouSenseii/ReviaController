"""
Persona consistency controller — persists tone, speaking style,
and boundaries per profile/channel and detects persona drift.

Tracks a stable persona fingerprint and compares it against recent
behaviour to flag drift.  Publishes ``persona_drift_detected`` when
the deviation exceeds a configurable threshold.

Integration
-----------
Subscribe the controller to ``chat_message`` events.  Call
``get_context()`` to get an LLM prompt injection block that
reminds the model of the persona baseline.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional


# ------------------------------------------------------------------
# Persona dimensions
# ------------------------------------------------------------------

@dataclass
class PersonaProfile:
    """
    Multi-dimensional persona fingerprint.

    Each dimension is a float in [0, 1] representing a behavioural
    axis.  The profile is the "intended" persona; the controller
    compares recent behaviour against it.
    """
    formality: float = 0.5       # 0 = casual, 1 = very formal
    warmth: float = 0.6          # 0 = cold/clinical, 1 = deeply warm
    verbosity: float = 0.5       # 0 = terse, 1 = very verbose
    humor: float = 0.3           # 0 = serious, 1 = humorous
    assertiveness: float = 0.5   # 0 = passive, 1 = very assertive
    empathy: float = 0.6         # 0 = detached, 1 = deeply empathetic
    creativity: float = 0.5      # 0 = literal/factual, 1 = creative
    patience: float = 0.7        # 0 = impatient, 1 = very patient

    # Explicit persona boundaries
    boundaries: List[str] = field(default_factory=list)
    # e.g. ["never discuss politics", "always use PG language"]

    # Preferred speaking style notes
    style_notes: str = ""
    # e.g. "Use short sentences. Prefer active voice."

    def to_vector(self) -> List[float]:
        return [
            self.formality, self.warmth, self.verbosity, self.humor,
            self.assertiveness, self.empathy, self.creativity, self.patience,
        ]

    @classmethod
    def from_vector(cls, vec: List[float]) -> "PersonaProfile":
        names = [
            "formality", "warmth", "verbosity", "humor",
            "assertiveness", "empathy", "creativity", "patience",
        ]
        kwargs = {n: max(0.0, min(1.0, v)) for n, v in zip(names, vec)}
        return cls(**kwargs)

    def distance(self, other: "PersonaProfile") -> float:
        """Euclidean distance between two profiles."""
        a = self.to_vector()
        b = other.to_vector()
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# ------------------------------------------------------------------
# Behaviour sample (sliding window entry)
# ------------------------------------------------------------------

@dataclass
class BehaviourSample:
    """A snapshot of observed persona dimensions from a single message."""
    profile: PersonaProfile
    timestamp: float = field(default_factory=time.time)
    source: str = "assistant"    # "user" or "assistant"


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

@dataclass
class PersonaConfig:
    drift_threshold: float = 0.35     # distance above which drift is flagged
    window_size: int = 20             # number of recent samples to track
    smoothing_factor: float = 0.3     # EMA blend for running average
    check_interval_msgs: int = 5      # check drift every N messages


# ------------------------------------------------------------------
# Controller
# ------------------------------------------------------------------

class PersonaController:
    """
    Monitors persona consistency and detects drift.

    Subscribes to:
        - ``chat_message`` — to observe assistant behaviour
        - ``persona_command`` — explicit persona management

    Publishes:
        - ``persona_drift_detected`` — when drift exceeds threshold
        - ``persona_updated`` — when baseline changes
    """

    def __init__(
        self,
        event_bus: Optional[Any] = None,
        config: Optional[PersonaConfig] = None,
        baseline: Optional[PersonaProfile] = None,
    ):
        self.bus = event_bus
        self.cfg = config or PersonaConfig()

        # The "intended" persona
        self.baseline = baseline or PersonaProfile()

        # Running EMA of observed persona
        self._running_avg = PersonaProfile.from_vector(self.baseline.to_vector())

        # Sliding window of recent samples
        self._window: Deque[BehaviourSample] = deque(maxlen=self.cfg.window_size)

        # Drift history
        self._drift_history: List[Dict[str, Any]] = []

        self._msg_count = 0

        if self.bus:
            self.bus.subscribe("chat_message", self._on_chat_message)
            self.bus.subscribe("persona_command", self._on_command)

    # ── Public API ────────────────────────────────────────────

    def set_baseline(self, profile: PersonaProfile) -> None:
        """Set or update the intended persona baseline."""
        self.baseline = profile
        self._running_avg = PersonaProfile.from_vector(profile.to_vector())
        self._publish_updated()

    def observe(self, sample: PersonaProfile, source: str = "assistant") -> None:
        """Record an observed behaviour sample."""
        self._window.append(BehaviourSample(profile=sample, source=source))

        # EMA update
        sf = self.cfg.smoothing_factor
        old = self._running_avg.to_vector()
        new = sample.to_vector()
        blended = [(1 - sf) * o + sf * n for o, n in zip(old, new)]
        self._running_avg = PersonaProfile.from_vector(blended)

        self._msg_count += 1
        if self._msg_count % self.cfg.check_interval_msgs == 0:
            self._check_drift()

    def current_drift(self) -> float:
        """Return the current drift distance from baseline."""
        return self.baseline.distance(self._running_avg)

    def get_context(self) -> Dict[str, Any]:
        """Build LLM prompt context for persona consistency."""
        drift = self.current_drift()
        drifting = drift > self.cfg.drift_threshold

        lines: List[str] = []

        # Describe baseline
        bp = self.baseline
        style_dims = [
            ("Formality", bp.formality),
            ("Warmth", bp.warmth),
            ("Verbosity", bp.verbosity),
            ("Humor", bp.humor),
            ("Assertiveness", bp.assertiveness),
            ("Empathy", bp.empathy),
            ("Creativity", bp.creativity),
            ("Patience", bp.patience),
        ]

        desc_parts = []
        for name, val in style_dims:
            if val > 0.7:
                desc_parts.append(f"high {name.lower()}")
            elif val < 0.3:
                desc_parts.append(f"low {name.lower()}")

        if desc_parts:
            lines.append(f"Your persona style: {', '.join(desc_parts)}.")

        if bp.style_notes:
            lines.append(f"Style notes: {bp.style_notes}")

        if bp.boundaries:
            lines.append("Persona boundaries:")
            for b in bp.boundaries:
                lines.append(f"  - {b}")

        if drifting:
            lines.append(
                "WARNING: Your recent responses show persona drift. "
                "Refocus on the style described above."
            )

        block = "\n".join(lines)
        prompt_injection = (
            "[Persona Consistency]\n"
            f"{block}\n"
            "Maintain this persona consistently across the conversation."
        )

        return {
            "drift": round(drift, 4),
            "drifting": drifting,
            "baseline": self.baseline.to_vector(),
            "running_avg": self._running_avg.to_vector(),
            "prompt_injection": prompt_injection,
        }

    def stats(self) -> Dict[str, Any]:
        return {
            "current_drift": round(self.current_drift(), 4),
            "drift_threshold": self.cfg.drift_threshold,
            "samples_collected": len(self._window),
            "drift_alerts": len(self._drift_history),
            "msg_count": self._msg_count,
        }

    # ── Event handlers ────────────────────────────────────────

    def _on_chat_message(self, data: Dict[str, Any]) -> None:
        role = data.get("role", "user")
        if role != "assistant":
            return

        # Estimate persona dimensions from message metadata if available
        sample = PersonaProfile(
            formality=data.get("persona_formality", self.baseline.formality),
            warmth=data.get("persona_warmth", self.baseline.warmth),
            verbosity=data.get("persona_verbosity", self.baseline.verbosity),
            humor=data.get("persona_humor", self.baseline.humor),
            assertiveness=data.get("persona_assertiveness", self.baseline.assertiveness),
            empathy=data.get("persona_empathy", self.baseline.empathy),
            creativity=data.get("persona_creativity", self.baseline.creativity),
            patience=data.get("persona_patience", self.baseline.patience),
        )
        self.observe(sample, source="assistant")

    def _on_command(self, data: Dict[str, Any]) -> None:
        action = data.get("action", "")
        if action == "set_baseline":
            vec = data.get("vector", self.baseline.to_vector())
            profile = PersonaProfile.from_vector(vec)
            profile.boundaries = data.get("boundaries", self.baseline.boundaries)
            profile.style_notes = data.get("style_notes", self.baseline.style_notes)
            self.set_baseline(profile)
        elif action == "add_boundary":
            boundary = data.get("boundary", "")
            if boundary and boundary not in self.baseline.boundaries:
                self.baseline.boundaries.append(boundary)
                self._publish_updated()
        elif action == "reset_drift":
            self._running_avg = PersonaProfile.from_vector(self.baseline.to_vector())
            self._window.clear()
            self._drift_history.clear()

    # ── Internal ──────────────────────────────────────────────

    def _check_drift(self) -> None:
        drift = self.current_drift()
        if drift > self.cfg.drift_threshold:
            entry = {
                "drift": round(drift, 4),
                "timestamp": time.time(),
                "running_avg": self._running_avg.to_vector(),
            }
            self._drift_history.append(entry)
            if self.bus:
                self.bus.publish("persona_drift_detected", entry)

    def _publish_updated(self) -> None:
        if self.bus:
            self.bus.publish("persona_updated", {
                "baseline": self.baseline.to_vector(),
                "drift": round(self.current_drift(), 4),
            })

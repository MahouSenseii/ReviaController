"""
Intent and need inference memory — persists inferred user goals,
motivations, urgency signals, and distinguishes confirmed facts
from inferred hypotheses.

Each inference is tagged with a confidence level and a source type
(``confirmed`` vs ``inferred``).  The LLM context block presents
high-confidence inferences and flags uncertain ones, preventing
overconfident recall.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ------------------------------------------------------------------
# Types
# ------------------------------------------------------------------

class InferenceType(str, Enum):
    GOAL = "goal"                 # what the user is trying to achieve
    MOTIVATION = "motivation"     # why they want it
    URGENCY = "urgency"           # how time-sensitive
    PREFERENCE = "preference"     # what they prefer
    NEED = "need"                 # something they require
    BOUNDARY = "boundary"         # something they want to avoid


class ConfidenceLevel(str, Enum):
    CONFIRMED = "confirmed"       # user explicitly stated
    HIGH = "high"                 # strongly inferred from context
    MEDIUM = "medium"             # reasonably inferred
    LOW = "low"                   # weakly inferred, could be wrong
    HYPOTHESIS = "hypothesis"     # speculative, needs validation


_CONFIDENCE_SCORES = {
    ConfidenceLevel.CONFIRMED: 1.0,
    ConfidenceLevel.HIGH: 0.85,
    ConfidenceLevel.MEDIUM: 0.65,
    ConfidenceLevel.LOW: 0.4,
    ConfidenceLevel.HYPOTHESIS: 0.2,
}


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class Inference:
    """A single inferred fact about the user."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    content: str = ""
    inference_type: InferenceType = InferenceType.GOAL
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    source_message: str = ""       # the message that triggered this inference
    user_id: str = ""
    session_id: str = ""
    superseded_by: Optional[str] = None  # if corrected, points to replacement
    active: bool = True

    @property
    def confidence_score(self) -> float:
        return _CONFIDENCE_SCORES.get(self.confidence, 0.5)


@dataclass
class IntentConfig:
    min_display_confidence: float = 0.4    # below this, don't inject into LLM
    max_active_inferences: int = 50        # cap on active inferences
    decay_confidence_days: float = 14.0    # hypotheses weaken over time
    top_n_context: int = 8                 # max inferences in LLM context


# ------------------------------------------------------------------
# Intent memory
# ------------------------------------------------------------------

class IntentMemory:
    """
    Manages inferred user intents, goals, and needs.

    Subscribes to:
        - ``intent_command`` — add/confirm/supersede inferences

    Publishes:
        - ``intent_updated`` — when inferences change
    """

    def __init__(
        self,
        event_bus: Optional[Any] = None,
        config: Optional[IntentConfig] = None,
    ):
        self.bus = event_bus
        self.cfg = config or IntentConfig()
        self._inferences: Dict[str, Inference] = {}

        if self.bus:
            self.bus.subscribe("intent_command", self._on_command)

    # ── Public API ────────────────────────────────────────────

    def add_inference(
        self,
        content: str,
        inference_type: InferenceType = InferenceType.GOAL,
        confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM,
        source_message: str = "",
        user_id: str = "",
        session_id: str = "",
    ) -> Inference:
        """Add a new inference about the user."""
        inf = Inference(
            content=content,
            inference_type=inference_type,
            confidence=confidence,
            source_message=source_message,
            user_id=user_id,
            session_id=session_id,
        )
        self._inferences[inf.id] = inf
        self._enforce_cap()
        self._publish()
        return inf

    def confirm(self, inference_id: str) -> bool:
        """Promote an inference to confirmed status."""
        inf = self._inferences.get(inference_id)
        if inf and inf.active:
            inf.confidence = ConfidenceLevel.CONFIRMED
            inf.updated_at = time.time()
            self._publish()
            return True
        return False

    def supersede(self, old_id: str, new_content: str) -> Optional[Inference]:
        """Replace an inference with a corrected version."""
        old = self._inferences.get(old_id)
        if not old:
            return None

        old.active = False
        old.superseded_by = None  # will be filled below

        new_inf = self.add_inference(
            content=new_content,
            inference_type=old.inference_type,
            confidence=ConfidenceLevel.CONFIRMED,
            user_id=old.user_id,
            session_id=old.session_id,
        )
        old.superseded_by = new_inf.id
        self._publish()
        return new_inf

    def deactivate(self, inference_id: str) -> bool:
        """Deactivate an inference (soft delete)."""
        inf = self._inferences.get(inference_id)
        if inf:
            inf.active = False
            self._publish()
            return True
        return False

    def get_active(
        self,
        inference_type: Optional[InferenceType] = None,
        user_id: Optional[str] = None,
        min_confidence: Optional[float] = None,
    ) -> List[Inference]:
        """Return active inferences, optionally filtered."""
        self._decay_hypotheses()

        results = [i for i in self._inferences.values() if i.active]

        if inference_type is not None:
            results = [i for i in results if i.inference_type == inference_type]
        if user_id is not None:
            results = [i for i in results if i.user_id == user_id]
        if min_confidence is not None:
            results = [i for i in results if i.confidence_score >= min_confidence]

        results.sort(key=lambda i: i.confidence_score, reverse=True)
        return results

    def get_context(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Build LLM prompt context for inferred intents."""
        active = self.get_active(
            user_id=user_id,
            min_confidence=self.cfg.min_display_confidence,
        )
        top = active[: self.cfg.top_n_context]

        lines: List[str] = []

        # Group by type
        confirmed = [i for i in top if i.confidence == ConfidenceLevel.CONFIRMED]
        inferred = [i for i in top if i.confidence != ConfidenceLevel.CONFIRMED]

        if confirmed:
            lines.append("Known user facts:")
            for i in confirmed:
                lines.append(f"  - [{i.inference_type.value}] {i.content}")

        if inferred:
            lines.append("Inferred (use cautiously):")
            for i in inferred:
                lines.append(
                    f"  - [{i.inference_type.value}] {i.content} "
                    f"(confidence: {i.confidence.value})"
                )

        if not lines:
            prompt_injection = (
                "[Intent Context]\n"
                "No known user goals or inferred needs at this time."
            )
        else:
            block = "\n".join(lines)
            prompt_injection = (
                "[Intent Context]\n"
                f"{block}\n"
                "Use confirmed facts freely. For inferred items, "
                "validate before acting on them."
            )

        return {
            "confirmed_count": len(confirmed),
            "inferred_count": len(inferred),
            "total_active": len(active),
            "prompt_injection": prompt_injection,
        }

    def stats(self) -> Dict[str, Any]:
        active = [i for i in self._inferences.values() if i.active]
        return {
            "total_inferences": len(self._inferences),
            "active_inferences": len(active),
            "confirmed": sum(1 for i in active if i.confidence == ConfidenceLevel.CONFIRMED),
            "hypotheses": sum(1 for i in active if i.confidence == ConfidenceLevel.HYPOTHESIS),
            "by_type": {
                t.value: sum(1 for i in active if i.inference_type == t)
                for t in InferenceType
            },
        }

    # ── Event handlers ────────────────────────────────────────

    def _on_command(self, data: Dict[str, Any]) -> None:
        action = data.get("action", "")
        if action == "add":
            itype = data.get("type", "goal")
            conf = data.get("confidence", "medium")
            self.add_inference(
                content=data.get("content", ""),
                inference_type=InferenceType(itype),
                confidence=ConfidenceLevel(conf),
                source_message=data.get("source_message", ""),
                user_id=data.get("user_id", ""),
                session_id=data.get("session_id", ""),
            )
        elif action == "confirm":
            self.confirm(data.get("inference_id", ""))
        elif action == "supersede":
            self.supersede(
                old_id=data.get("old_id", ""),
                new_content=data.get("new_content", ""),
            )
        elif action == "deactivate":
            self.deactivate(data.get("inference_id", ""))

    # ── Internal helpers ──────────────────────────────────────

    def _decay_hypotheses(self) -> None:
        """Weaken old hypotheses over time."""
        now = time.time()
        threshold_secs = self.cfg.decay_confidence_days * 86400
        for inf in self._inferences.values():
            if not inf.active:
                continue
            if inf.confidence == ConfidenceLevel.HYPOTHESIS:
                age = now - inf.created_at
                if age > threshold_secs:
                    inf.active = False

    def _enforce_cap(self) -> None:
        active = [i for i in self._inferences.values() if i.active]
        if len(active) > self.cfg.max_active_inferences:
            active.sort(key=lambda i: i.confidence_score)
            excess = len(active) - self.cfg.max_active_inferences
            for i in active[:excess]:
                i.active = False

    def _publish(self) -> None:
        if self.bus:
            self.bus.publish("intent_updated", self.stats())

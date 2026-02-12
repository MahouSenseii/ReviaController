"""
Repair and accountability memory — logs mistakes, corrections, and
user feedback loops.

When a correction is recorded, the system links it to the original
(incorrect) memory and ensures future recall prefers the corrected
version.  This prevents the AI from repeating known mistakes.

Integration
-----------
Subscribe to ``correction_command`` events to log corrections.
Call ``get_context()`` to inject correction-awareness into the LLM
prompt so it knows about prior mistakes and corrections.
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

class CorrectionType(str, Enum):
    FACTUAL = "factual"           # wrong fact recalled
    BEHAVIOURAL = "behavioural"   # inappropriate response style
    PREFERENCE = "preference"     # misremembered user preference
    BOUNDARY = "boundary"         # crossed a user boundary
    HALLUCINATION = "hallucination"  # fabricated information


class FeedbackSentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    CORRECTION = "correction"


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class Correction:
    """A recorded correction to a prior mistake."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    original_content: str = ""       # what was said incorrectly
    corrected_content: str = ""      # the correct version
    correction_type: CorrectionType = CorrectionType.FACTUAL
    reason: str = ""                 # why it was wrong
    created_at: float = field(default_factory=time.time)
    original_memory_id: Optional[str] = None  # link to the faulty memory
    session_id: str = ""
    user_id: str = ""
    acknowledged: bool = False       # whether the AI acknowledged the mistake


@dataclass
class FeedbackEntry:
    """User feedback on an interaction."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    sentiment: FeedbackSentiment = FeedbackSentiment.NEUTRAL
    content: str = ""                # optional feedback text
    context_message: str = ""        # the message the feedback is about
    created_at: float = field(default_factory=time.time)
    session_id: str = ""
    user_id: str = ""


@dataclass
class RepairConfig:
    max_corrections: int = 200        # cap on stored corrections
    max_feedback: int = 500           # cap on stored feedback
    correction_context_count: int = 5  # corrections shown to LLM
    feedback_window_hours: float = 48.0  # recent feedback window


# ------------------------------------------------------------------
# Repair memory
# ------------------------------------------------------------------

class RepairMemory:
    """
    Tracks corrections and feedback for accountability.

    Subscribes to:
        - ``correction_command`` — log corrections
        - ``feedback_command`` — log user feedback

    Publishes:
        - ``correction_logged`` — when a new correction is recorded
        - ``feedback_logged`` — when feedback is recorded
        - ``repair_stats_updated`` — when stats change
    """

    def __init__(
        self,
        event_bus: Optional[Any] = None,
        config: Optional[RepairConfig] = None,
    ):
        self.bus = event_bus
        self.cfg = config or RepairConfig()

        self._corrections: Dict[str, Correction] = {}
        self._feedback: Dict[str, FeedbackEntry] = {}

        # Quick lookup: original_memory_id → correction
        self._memory_corrections: Dict[str, str] = {}

        if self.bus:
            self.bus.subscribe("correction_command", self._on_correction)
            self.bus.subscribe("feedback_command", self._on_feedback)

    # ── Corrections ───────────────────────────────────────────

    def log_correction(
        self,
        original_content: str,
        corrected_content: str,
        correction_type: CorrectionType = CorrectionType.FACTUAL,
        reason: str = "",
        original_memory_id: Optional[str] = None,
        session_id: str = "",
        user_id: str = "",
    ) -> Correction:
        """Record a correction."""
        c = Correction(
            original_content=original_content,
            corrected_content=corrected_content,
            correction_type=correction_type,
            reason=reason,
            original_memory_id=original_memory_id,
            session_id=session_id,
            user_id=user_id,
        )
        self._corrections[c.id] = c

        if original_memory_id:
            self._memory_corrections[original_memory_id] = c.id

        self._enforce_correction_cap()

        if self.bus:
            self.bus.publish("correction_logged", {
                "id": c.id,
                "type": correction_type.value,
                "original": original_content[:100],
                "corrected": corrected_content[:100],
            })

        return c

    def get_correction_for_memory(self, memory_id: str) -> Optional[Correction]:
        """Check if a memory has been corrected."""
        cid = self._memory_corrections.get(memory_id)
        if cid:
            return self._corrections.get(cid)
        return None

    def get_recent_corrections(
        self,
        n: int = 10,
        correction_type: Optional[CorrectionType] = None,
    ) -> List[Correction]:
        """Return recent corrections, optionally filtered by type."""
        results = list(self._corrections.values())
        if correction_type is not None:
            results = [c for c in results if c.correction_type == correction_type]
        results.sort(key=lambda c: c.created_at, reverse=True)
        return results[:n]

    def acknowledge(self, correction_id: str) -> bool:
        """Mark a correction as acknowledged."""
        c = self._corrections.get(correction_id)
        if c:
            c.acknowledged = True
            return True
        return False

    # ── Feedback ──────────────────────────────────────────────

    def log_feedback(
        self,
        sentiment: FeedbackSentiment = FeedbackSentiment.NEUTRAL,
        content: str = "",
        context_message: str = "",
        session_id: str = "",
        user_id: str = "",
    ) -> FeedbackEntry:
        """Record user feedback."""
        f = FeedbackEntry(
            sentiment=sentiment,
            content=content,
            context_message=context_message,
            session_id=session_id,
            user_id=user_id,
        )
        self._feedback[f.id] = f
        self._enforce_feedback_cap()

        if self.bus:
            self.bus.publish("feedback_logged", {
                "id": f.id,
                "sentiment": sentiment.value,
            })

        return f

    def get_recent_feedback(
        self,
        hours: Optional[float] = None,
        sentiment: Optional[FeedbackSentiment] = None,
    ) -> List[FeedbackEntry]:
        """Return recent feedback entries."""
        window = (hours or self.cfg.feedback_window_hours) * 3600
        cutoff = time.time() - window

        results = [f for f in self._feedback.values() if f.created_at >= cutoff]
        if sentiment is not None:
            results = [f for f in results if f.sentiment == sentiment]
        results.sort(key=lambda f: f.created_at, reverse=True)
        return results

    def feedback_summary(self) -> Dict[str, int]:
        """Count feedback by sentiment."""
        counts: Dict[str, int] = {}
        for f in self._feedback.values():
            counts[f.sentiment.value] = counts.get(f.sentiment.value, 0) + 1
        return counts

    # ── Context for LLM ──────────────────────────────────────

    def get_context(self) -> Dict[str, Any]:
        """Build LLM prompt context for repair awareness."""
        recent = self.get_recent_corrections(n=self.cfg.correction_context_count)
        unacked = [c for c in recent if not c.acknowledged]

        lines: List[str] = []

        if unacked:
            lines.append("Recent corrections to be aware of:")
            for c in unacked:
                lines.append(
                    f"  - WRONG: \"{c.original_content[:80]}\"\n"
                    f"    CORRECT: \"{c.corrected_content[:80]}\""
                )
                if c.reason:
                    lines.append(f"    Reason: {c.reason}")

        # Recent negative feedback
        neg_feedback = self.get_recent_feedback(
            hours=self.cfg.feedback_window_hours,
            sentiment=FeedbackSentiment.NEGATIVE,
        )
        if neg_feedback:
            lines.append("Recent negative feedback:")
            for f in neg_feedback[:3]:
                text = f.content or f.context_message
                if text:
                    lines.append(f"  - {text[:100]}")

        if not lines:
            prompt_injection = (
                "[Accountability Context]\n"
                "No recent corrections or negative feedback. Keep up the good work."
            )
        else:
            block = "\n".join(lines)
            prompt_injection = (
                "[Accountability Context]\n"
                f"{block}\n"
                "Prefer corrected facts over prior incorrect recalls. "
                "Acknowledge past mistakes when relevant."
            )

        return {
            "total_corrections": len(self._corrections),
            "unacknowledged": len(unacked),
            "recent_negative_feedback": len(neg_feedback) if neg_feedback else 0,
            "prompt_injection": prompt_injection,
        }

    def stats(self) -> Dict[str, Any]:
        return {
            "total_corrections": len(self._corrections),
            "acknowledged": sum(1 for c in self._corrections.values() if c.acknowledged),
            "by_type": {
                t.value: sum(1 for c in self._corrections.values() if c.correction_type == t)
                for t in CorrectionType
            },
            "total_feedback": len(self._feedback),
            "feedback_by_sentiment": self.feedback_summary(),
        }

    # ── Event handlers ────────────────────────────────────────

    def _on_correction(self, data: Dict[str, Any]) -> None:
        ctype = data.get("type", "factual")
        self.log_correction(
            original_content=data.get("original", ""),
            corrected_content=data.get("corrected", ""),
            correction_type=CorrectionType(ctype),
            reason=data.get("reason", ""),
            original_memory_id=data.get("memory_id"),
            session_id=data.get("session_id", ""),
            user_id=data.get("user_id", ""),
        )

    def _on_feedback(self, data: Dict[str, Any]) -> None:
        sent = data.get("sentiment", "neutral")
        self.log_feedback(
            sentiment=FeedbackSentiment(sent),
            content=data.get("content", ""),
            context_message=data.get("context_message", ""),
            session_id=data.get("session_id", ""),
            user_id=data.get("user_id", ""),
        )

    # ── Internal ──────────────────────────────────────────────

    def _enforce_correction_cap(self) -> None:
        if len(self._corrections) > self.cfg.max_corrections:
            sorted_c = sorted(self._corrections.values(), key=lambda c: c.created_at)
            excess = len(self._corrections) - self.cfg.max_corrections
            for c in sorted_c[:excess]:
                del self._corrections[c.id]

    def _enforce_feedback_cap(self) -> None:
        if len(self._feedback) > self.cfg.max_feedback:
            sorted_f = sorted(self._feedback.values(), key=lambda f: f.created_at)
            excess = len(self._feedback) - self.cfg.max_feedback
            for f in sorted_f[:excess]:
                del self._feedback[f.id]

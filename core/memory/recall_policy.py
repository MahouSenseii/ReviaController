"""
Proactive recall policy — decides when and how to proactively
reference memory versus staying concise.

Controls:
    - Whether to proactively inject memories into context
    - Confidence thresholds per memory type
    - Verbosity of memory references
    - Rate limiting to avoid excessive memory citations

The policy outputs a ``RecallDecision`` that the orchestrator
uses to decide which memories to include and how to present them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ------------------------------------------------------------------
# Policy modes
# ------------------------------------------------------------------

class RecallMode(str, Enum):
    CONCISE = "concise"       # minimal memory injection
    BALANCED = "balanced"     # moderate memory injection
    PROACTIVE = "proactive"   # actively reference relevant memories
    SILENT = "silent"         # no memory injection


# ------------------------------------------------------------------
# Decision output
# ------------------------------------------------------------------

@dataclass
class RecallDecision:
    """The policy's decision on what to recall and how."""
    should_recall: bool = True
    mode: RecallMode = RecallMode.BALANCED
    max_memories: int = 5
    min_confidence: float = 0.3
    include_emotions: bool = True
    include_corrections: bool = True
    include_inferences: bool = False   # conservative by default
    verbosity: str = "normal"          # "minimal", "normal", "detailed"
    reason: str = ""


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

@dataclass
class RecallPolicyConfig:
    default_mode: RecallMode = RecallMode.BALANCED
    confidence_threshold_confirmed: float = 0.2   # low bar for confirmed facts
    confidence_threshold_inferred: float = 0.6    # higher bar for inferences
    max_recalls_per_turn: int = 7
    cooldown_seconds: float = 2.0     # min seconds between proactive recalls
    emotion_memory_threshold: float = 0.3  # min emotion match to include
    correction_always_include: bool = True  # always include relevant corrections
    proactive_similarity_boost: float = 0.15  # extra similarity needed in proactive mode


# ------------------------------------------------------------------
# Policy engine
# ------------------------------------------------------------------

class RecallPolicy:
    """
    Evaluates context to decide recall strategy.

    Subscribes to:
        - ``recall_policy_command`` — change mode, thresholds

    Publishes:
        - ``recall_policy_changed`` — when policy settings change
    """

    def __init__(
        self,
        event_bus: Optional[Any] = None,
        config: Optional[RecallPolicyConfig] = None,
    ):
        self.bus = event_bus
        self.cfg = config or RecallPolicyConfig()
        self._mode = self.cfg.default_mode
        self._last_recall_time: float = 0.0
        self._recall_count_this_session: int = 0

        if self.bus:
            self.bus.subscribe("recall_policy_command", self._on_command)

    # ── Public API ────────────────────────────────────────────

    @property
    def mode(self) -> RecallMode:
        return self._mode

    @mode.setter
    def mode(self, value: RecallMode) -> None:
        self._mode = value
        self._publish_changed()

    def evaluate(
        self,
        query: str = "",
        message_count: int = 0,
        emotion_intensity: float = 0.0,
        has_corrections: bool = False,
        user_explicitly_asked: bool = False,
    ) -> RecallDecision:
        """
        Evaluate the current context and return a recall decision.

        Parameters
        ----------
        query : str
            The user's message or query.
        message_count : int
            Number of messages in the current conversation.
        emotion_intensity : float
            Intensity of the dominant emotion (0-1).
        has_corrections : bool
            Whether there are relevant corrections.
        user_explicitly_asked : bool
            Whether the user asked about past context.
        """
        now = time.time()

        # Silent mode: never recall
        if self._mode == RecallMode.SILENT:
            return RecallDecision(
                should_recall=False,
                mode=RecallMode.SILENT,
                reason="Recall is disabled (silent mode).",
            )

        # User explicitly asked: always recall fully
        if user_explicitly_asked:
            return RecallDecision(
                should_recall=True,
                mode=RecallMode.PROACTIVE,
                max_memories=self.cfg.max_recalls_per_turn,
                min_confidence=0.1,
                include_emotions=True,
                include_corrections=True,
                include_inferences=True,
                verbosity="detailed",
                reason="User explicitly asked about past context.",
            )

        # Corrections always included if configured
        include_corrections = (
            has_corrections and self.cfg.correction_always_include
        )

        # Mode-specific decisions
        if self._mode == RecallMode.CONCISE:
            return RecallDecision(
                should_recall=True,
                mode=RecallMode.CONCISE,
                max_memories=3,
                min_confidence=self.cfg.confidence_threshold_confirmed,
                include_emotions=False,
                include_corrections=include_corrections,
                include_inferences=False,
                verbosity="minimal",
                reason="Concise mode: minimal context.",
            )

        if self._mode == RecallMode.PROACTIVE:
            # Rate-limit proactive recalls
            if (now - self._last_recall_time) < self.cfg.cooldown_seconds:
                return RecallDecision(
                    should_recall=True,
                    mode=RecallMode.BALANCED,
                    max_memories=self.cfg.max_recalls_per_turn,
                    min_confidence=self.cfg.confidence_threshold_confirmed,
                    include_emotions=True,
                    include_corrections=include_corrections,
                    include_inferences=False,
                    verbosity="normal",
                    reason="Proactive recall on cooldown, using balanced.",
                )

            self._last_recall_time = now
            return RecallDecision(
                should_recall=True,
                mode=RecallMode.PROACTIVE,
                max_memories=self.cfg.max_recalls_per_turn,
                min_confidence=self.cfg.confidence_threshold_inferred,
                include_emotions=True,
                include_corrections=include_corrections,
                include_inferences=True,
                verbosity="detailed",
                reason="Proactive recall: full context injection.",
            )

        # Balanced mode (default)
        # Increase detail when emotions are strong or early in conversation
        if emotion_intensity > 0.6 or message_count <= 3:
            verbosity = "detailed"
            max_mem = min(self.cfg.max_recalls_per_turn, 6)
        else:
            verbosity = "normal"
            max_mem = min(self.cfg.max_recalls_per_turn, 5)

        self._last_recall_time = now
        self._recall_count_this_session += 1

        return RecallDecision(
            should_recall=True,
            mode=RecallMode.BALANCED,
            max_memories=max_mem,
            min_confidence=self.cfg.confidence_threshold_confirmed,
            include_emotions=emotion_intensity > self.cfg.emotion_memory_threshold,
            include_corrections=include_corrections,
            include_inferences=False,
            verbosity=verbosity,
            reason="Balanced mode: moderate context.",
        )

    def reset_session(self) -> None:
        """Reset session-scoped counters."""
        self._recall_count_this_session = 0
        self._last_recall_time = 0.0

    def stats(self) -> Dict[str, Any]:
        return {
            "mode": self._mode.value,
            "recalls_this_session": self._recall_count_this_session,
            "last_recall_time": self._last_recall_time,
        }

    # ── Event handlers ────────────────────────────────────────

    def _on_command(self, data: Dict[str, Any]) -> None:
        action = data.get("action", "")
        if action == "set_mode":
            mode_str = data.get("mode", "balanced")
            try:
                self._mode = RecallMode(mode_str)
                self._publish_changed()
            except ValueError:
                pass
        elif action == "reset":
            self.reset_session()

    def _publish_changed(self) -> None:
        if self.bus:
            self.bus.publish("recall_policy_changed", {
                "mode": self._mode.value,
            })

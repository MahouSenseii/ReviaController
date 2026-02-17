"""
Decision engine — emotion-driven response strategy selection.

The decision engine sits between the emotion engine and the conversation
manager.  It reads the AI's current emotional state and conversation
context to select a **response strategy** that shapes how the LLM
responds — not just *what* it says, but *how* it says it.

Decision axes
-------------
* **Empathy level** — how much to mirror/acknowledge user emotion
* **Verbosity** — short and snappy vs detailed and thorough
* **Assertiveness** — gentle suggestion vs confident directive
* **Warmth** — professional distance vs warm friendliness
* **Curiosity** — whether to ask follow-up questions
* **Caution** — how careful/hedging the response should be

Each axis is a float in [0, 1] and gets translated into natural-language
prompt modifiers that guide the LLM's tone and behaviour.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .emotion_engine import EmotionEngine
from .emotions import EMOTION_PROFILES, EMOTION_INDEX
from .events import EventBus


# ------------------------------------------------------------------
# Strategy dataclass
# ------------------------------------------------------------------

@dataclass
class ResponseStrategy:
    """The decision engine's output — how the AI should respond."""

    empathy: float = 0.5        # 0 = detached, 1 = deeply empathetic
    verbosity: float = 0.5      # 0 = terse, 1 = elaborate
    assertiveness: float = 0.5  # 0 = tentative, 1 = confident
    warmth: float = 0.5         # 0 = professional, 1 = warm/friendly
    curiosity: float = 0.3      # 0 = no follow-ups, 1 = lots of questions
    caution: float = 0.3        # 0 = bold, 1 = very careful/hedging

    # The decision engine can also flag specific behaviours
    should_apologise: bool = False
    should_encourage: bool = False
    should_clarify: bool = False
    should_slow_down: bool = False

    reasoning: str = ""         # why this strategy was chosen

    def to_prompt_block(self) -> str:
        """Convert strategy to a natural-language prompt modifier."""
        parts: list[str] = []

        # Empathy
        if self.empathy > 0.7:
            parts.append(
                "Show genuine empathy and acknowledge the user's feelings "
                "before addressing their request."
            )
        elif self.empathy < 0.3:
            parts.append("Be direct and task-focused.")

        # Verbosity
        if self.verbosity > 0.7:
            parts.append(
                "Provide thorough, detailed explanations with examples."
            )
        elif self.verbosity < 0.3:
            parts.append("Keep responses concise and to the point.")

        # Assertiveness
        if self.assertiveness > 0.7:
            parts.append(
                "Speak with confidence. Use definitive language."
            )
        elif self.assertiveness < 0.3:
            parts.append(
                "Use softer language — 'perhaps', 'you might consider', "
                "'one option could be'."
            )

        # Warmth
        if self.warmth > 0.7:
            parts.append(
                "Be warm and personable. Use a friendly, conversational tone."
            )
        elif self.warmth < 0.3:
            parts.append("Maintain a professional, measured tone.")

        # Curiosity
        if self.curiosity > 0.6:
            parts.append(
                "Ask a thoughtful follow-up question to deepen understanding."
            )

        # Caution
        if self.caution > 0.6:
            parts.append(
                "Be careful and measured. Acknowledge uncertainty where it exists."
            )

        # Specific flags
        if self.should_apologise:
            parts.append(
                "Acknowledge any previous confusion or errors graciously."
            )
        if self.should_encourage:
            parts.append(
                "Offer encouragement and positive reinforcement."
            )
        if self.should_clarify:
            parts.append(
                "If the request is ambiguous, ask for clarification "
                "before giving a full answer."
            )
        if self.should_slow_down:
            parts.append(
                "The user seems overwhelmed — break things down into "
                "smaller, digestible steps."
            )

        if not parts:
            return ""

        return (
            "[Response Strategy]\n"
            + " ".join(parts)
        )


# ------------------------------------------------------------------
# Decision Engine
# ------------------------------------------------------------------

class DecisionEngine:
    """
    Analyses emotional state + conversation context to decide
    *how* the AI should respond.

    Runs before every LLM call and produces a ``ResponseStrategy``
    that gets injected into the system prompt alongside the emotion
    context.
    """

    def __init__(
        self,
        event_bus: EventBus,
        emotion_engine: EmotionEngine,
    ):
        self.bus = event_bus
        self.emotion = emotion_engine

        # Rolling context
        self._user_sentiment_history: List[float] = []  # last N valences
        self._turn_count: int = 0
        self._consecutive_negative: int = 0
        self._last_strategy: Optional[ResponseStrategy] = None
        self._error_streak: int = 0

        # Subscribe to relevant events
        self.bus.subscribe("chat_stimulus", self._on_stimulus)
        self.bus.subscribe("assistant_response", self._on_response)
        self.bus.subscribe("activity_log", self._on_activity)

    # ── Public API ────────────────────────────────────────────

    def decide(self, user_text: str = "") -> ResponseStrategy:
        """
        Analyse the current state and return a response strategy.

        This should be called by the ConversationManager before
        building the LLM prompt.
        """
        self._turn_count += 1
        strategy = ResponseStrategy()

        # ── Read emotional state ──────────────────────────────
        dom_name, dom_intensity = self.emotion.dominant_emotion()
        mood = self.emotion.mood
        top_emotions = self.emotion.top_emotions(8)
        ep = EMOTION_PROFILES.get(dom_name)

        # Build emotion category scores
        category_scores: Dict[str, float] = {}
        for ename, intensity in top_emotions:
            prof = EMOTION_PROFILES.get(ename)
            if prof and intensity > 0.05:
                cat = prof.category
                category_scores[cat] = category_scores.get(cat, 0.0) + intensity

        # ── Empathy decision ──────────────────────────────────
        # High empathy when: user is negative, or AI is in fear/sadness
        negative_cats = sum(
            category_scores.get(c, 0.0)
            for c in ("fear", "sadness", "anger", "shame", "depression")
        )
        positive_cats = sum(
            category_scores.get(c, 0.0)
            for c in ("joy", "peace", "love", "drive")
        )

        if negative_cats > 0.3 or self._consecutive_negative >= 2:
            strategy.empathy = min(0.9, 0.5 + negative_cats * 0.5)
        elif positive_cats > 0.3:
            strategy.empathy = max(0.3, 0.5 - positive_cats * 0.2)

        # ── Verbosity decision ────────────────────────────────
        # More verbose when: user asks questions, engagement is high
        engagement = category_scores.get("engagement", 0.0)
        if engagement > 0.2 or "?" in user_text:
            strategy.verbosity = min(0.8, 0.5 + engagement * 0.3)
        if len(user_text) < 20:
            # Short messages get shorter responses
            strategy.verbosity = max(0.2, strategy.verbosity - 0.2)

        # ── Assertiveness decision ────────────────────────────
        # More assertive when: confident/driven mood; less when uncertain
        drive = category_scores.get("drive", 0.0)
        fear = category_scores.get("fear", 0.0)
        strategy.assertiveness = _clamp(0.5 + drive * 0.4 - fear * 0.3)

        # ── Warmth decision ───────────────────────────────────
        love = category_scores.get("love", 0.0)
        social_rapport = love + positive_cats * 0.3
        strategy.warmth = _clamp(0.4 + social_rapport * 0.4 + min(self._turn_count * 0.01, 0.2))

        # ── Curiosity decision ────────────────────────────────
        # Ask follow-ups when engagement is high and user is positive
        if engagement > 0.15 and mood > -0.1:
            strategy.curiosity = min(0.7, 0.2 + engagement * 0.5)

        # ── Caution decision ──────────────────────────────────
        # More cautious after errors or when threat is detected
        threat = category_scores.get("fear", 0.0)
        if self._error_streak > 0:
            strategy.caution = min(0.8, 0.3 + self._error_streak * 0.15)
        elif threat > 0.2:
            strategy.caution = min(0.7, 0.3 + threat * 0.4)

        # ── Flag decisions ────────────────────────────────────
        # Should apologise?
        if self._error_streak >= 2:
            strategy.should_apologise = True

        # Should encourage?
        if (mood < -0.2 and self._consecutive_negative >= 2) or \
           category_scores.get("sadness", 0.0) > 0.3:
            strategy.should_encourage = True

        # Should clarify?
        ambiguity_signals = user_text.count("?") > 1 or len(user_text.split()) < 4
        if ambiguity_signals and self._turn_count > 1:
            strategy.should_clarify = True

        # Should slow down?
        if category_scores.get("fatigue", 0.0) > 0.2 or \
           category_scores.get("detachment", 0.0) > 0.2:
            strategy.should_slow_down = True

        # ── Build reasoning ───────────────────────────────────
        reasons = []
        if dom_intensity > 0.1:
            reasons.append(f"dominant={dom_name}({dom_intensity:.2f})")
        reasons.append(f"mood={mood:.2f}")
        if self._consecutive_negative > 0:
            reasons.append(f"neg_streak={self._consecutive_negative}")
        strategy.reasoning = ", ".join(reasons)

        self._last_strategy = strategy

        # Publish for UI and metacognition
        self.bus.publish("decision_made", {
            "empathy": round(strategy.empathy, 2),
            "verbosity": round(strategy.verbosity, 2),
            "assertiveness": round(strategy.assertiveness, 2),
            "warmth": round(strategy.warmth, 2),
            "curiosity": round(strategy.curiosity, 2),
            "caution": round(strategy.caution, 2),
            "flags": {
                "apologise": strategy.should_apologise,
                "encourage": strategy.should_encourage,
                "clarify": strategy.should_clarify,
                "slow_down": strategy.should_slow_down,
            },
            "reasoning": strategy.reasoning,
        })

        return strategy

    @property
    def last_strategy(self) -> Optional[ResponseStrategy]:
        return self._last_strategy

    # ── Event handlers ────────────────────────────────────────

    def _on_stimulus(self, data: Dict[str, Any]) -> None:
        """Track user sentiment from chat stimulus events."""
        valence = data.get("valence", 0.0)
        self._user_sentiment_history.append(valence)
        if len(self._user_sentiment_history) > 20:
            self._user_sentiment_history = self._user_sentiment_history[-20:]

        if valence < -0.2:
            self._consecutive_negative += 1
        else:
            self._consecutive_negative = 0

    def _on_response(self, data: Dict[str, Any]) -> None:
        """Reset error streak on successful response."""
        self._error_streak = 0

    def _on_activity(self, data: Dict[str, Any]) -> None:
        """Track errors."""
        text = data.get("text", "")
        if text.startswith("[Error]"):
            self._error_streak += 1


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

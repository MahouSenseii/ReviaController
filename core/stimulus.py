"""
Stimulus analyser — converts chat messages into emotion stimuli.

Since the project avoids external NLP libraries, this module uses
keyword / pattern matching to estimate the emotional dimensions of
user and assistant messages, then publishes ``chat_stimulus`` events
for the EmotionEngine to consume.

The analysis is intentionally simple but extensible.  Each message
is scored across the 7 stimulus dimensions:

    valence, arousal, social_connect, novelty, threat, engagement, rapport
"""

from __future__ import annotations

import re
from typing import Any, Dict

from .events import EventBus


# ── Keyword dictionaries ─────────────────────────────────────

_POSITIVE_WORDS = {
    "thank", "thanks", "love", "great", "awesome", "amazing", "wonderful",
    "happy", "glad", "nice", "good", "excellent", "perfect", "beautiful",
    "fantastic", "brilliant", "appreciate", "helpful", "kind", "sweet",
    "yes", "please", "enjoy", "fun", "excited", "cool", "wow", "yay",
    "haha", "lol", "lmao", ":)", ":D", "<3",
}

_NEGATIVE_WORDS = {
    "hate", "bad", "terrible", "awful", "horrible", "wrong", "stupid",
    "ugly", "angry", "sad", "annoyed", "frustrated", "disappointed",
    "confused", "worried", "scared", "afraid", "no", "stop", "don't",
    "can't", "won't", "never", "worst", "useless", "broken", "fail",
    "error", "bug", "crash", "damn", "hell", "ugh", ":(", ":/",
}

_THREAT_WORDS = {
    "error", "crash", "fail", "broken", "bug", "wrong", "problem",
    "issue", "danger", "warning", "critical", "urgent", "emergency",
    "scared", "afraid", "worried", "threat", "attack", "virus",
}

_CURIOSITY_WORDS = {
    "what", "how", "why", "when", "where", "who", "which", "could",
    "would", "should", "tell", "explain", "describe", "show", "help",
    "wonder", "curious", "interesting", "really", "?",
}

_SOCIAL_WORDS = {
    "we", "us", "our", "together", "friend", "buddy", "hey", "hi",
    "hello", "morning", "evening", "night", "bye", "goodbye", "miss",
    "love", "care", "hug", "please", "thank",
}

_EXCITEMENT_WORDS = {
    "!", "!!", "!!!", "wow", "amazing", "incredible", "omg", "oh",
    "yes", "yay", "excited", "awesome", "fantastic", "unbelievable",
    "urgent", "now", "hurry", "quick", "fast", "immediately",
}


def _count_matches(text: str, wordset: set[str]) -> int:
    """Count how many words from the set appear in the text."""
    words = set(re.findall(r"[a-z]+|[!?:;<>()]+", text.lower()))
    return len(words & wordset)


def _has_question(text: str) -> bool:
    return "?" in text


def _exclamation_density(text: str) -> float:
    count = text.count("!")
    return min(count / max(len(text.split()), 1), 1.0)


# ── Main analyser ─────────────────────────────────────────────

class StimulusAnalyser:
    """
    Analyses chat messages and publishes emotion stimuli.

    Subscribes to ``assistant_response`` and ``user_message`` events.
    Publishes ``chat_stimulus`` events for the EmotionEngine.
    """

    def __init__(self, event_bus: EventBus):
        self.bus = event_bus
        self._turn_count: int = 0

        self.bus.subscribe("user_message", self._on_user_message)
        self.bus.subscribe("assistant_response", self._on_assistant_response)

    def analyse(self, text: str, is_user: bool = True) -> Dict[str, Any]:
        """
        Analyse a message and return stimulus dimensions.

        Returns a dict matching the ``Stimulus`` fields.
        """
        self._turn_count += 1

        pos = _count_matches(text, _POSITIVE_WORDS)
        neg = _count_matches(text, _NEGATIVE_WORDS)
        threat = _count_matches(text, _THREAT_WORDS)
        curiosity = _count_matches(text, _CURIOSITY_WORDS)
        social = _count_matches(text, _SOCIAL_WORDS)
        excitement = _count_matches(text, _EXCITEMENT_WORDS)

        total_words = max(len(text.split()), 1)

        # ── Valence: -1 to +1 ────────────────────────────────
        if pos + neg > 0:
            valence = (pos - neg) / (pos + neg)
        else:
            valence = 0.0
        valence = max(-1.0, min(1.0, valence))

        # ── Arousal: 0 to 1 ──────────────────────────────────
        arousal_raw = (
            _exclamation_density(text) * 0.3
            + min(excitement / max(total_words, 1), 1.0) * 0.3
            + (0.2 if len(text) > 200 else 0.0)
            + (0.2 if _has_question(text) else 0.0)
        )
        arousal = max(0.1, min(1.0, arousal_raw + 0.2))

        # ── Social connection: 0 to 1 ────────────────────────
        social_score = min(social / max(total_words * 0.3, 1), 1.0)
        social_connect = max(0.2, min(1.0, social_score + 0.3))

        # ── Novelty: 0 to 1 ──────────────────────────────────
        has_q = _has_question(text)
        novelty_raw = (
            (0.4 if has_q else 0.0)
            + min(curiosity / max(total_words * 0.3, 1), 1.0) * 0.3
            + (0.3 if self._turn_count <= 3 else 0.0)  # early turns
        )
        novelty = max(0.1, min(1.0, novelty_raw))

        # ── Threat: 0 to 1 ───────────────────────────────────
        threat_score = min(threat / max(total_words * 0.2, 1), 1.0)
        threat_val = max(0.0, min(1.0, threat_score * 0.7))

        # ── Engagement: 0 to 1 ───────────────────────────────
        engagement = max(0.3, min(1.0,
            0.3  # base engagement (they're chatting)
            + (0.2 if len(text) > 50 else 0.0)
            + (0.2 if has_q else 0.0)
            + min(self._turn_count * 0.02, 0.3)
        ))

        # ── Rapport: 0 to 1 ──────────────────────────────────
        # Builds over conversation length
        rapport = max(0.2, min(1.0,
            0.2 + min(self._turn_count * 0.03, 0.5)
            + social_connect * 0.2
            + (0.1 if valence > 0 else 0.0)
        ))

        # ── Emotion hint ─────────────────────────────────────
        hint = None
        if valence > 0.5 and arousal > 0.5:
            hint = "happy"
        elif valence > 0.3:
            hint = "content"
        elif valence < -0.5 and threat_val > 0.3:
            hint = "afraid"
        elif valence < -0.5:
            hint = "sad"
        elif valence < -0.2:
            hint = "frustrated"
        elif has_q and curiosity > 1:
            hint = "curious"

        return {
            "valence": round(valence, 3),
            "arousal": round(arousal, 3),
            "social_connect": round(social_connect, 3),
            "novelty": round(novelty, 3),
            "threat": round(threat_val, 3),
            "engagement": round(engagement, 3),
            "rapport": round(rapport, 3),
            "emotion_hint": hint,
        }

    # ── Event handlers ────────────────────────────────────────

    def _on_user_message(self, data: Dict[str, Any]) -> None:
        text = data.get("text", "")
        if text:
            stim = self.analyse(text, is_user=True)
            self.bus.publish("chat_stimulus", stim)

    def _on_assistant_response(self, data: Dict[str, Any]) -> None:
        text = data.get("text", "")
        if text:
            # Assistant responses have gentler stimulus effect
            stim = self.analyse(text, is_user=False)
            # Dampen — assistant's own words shouldn't swing emotions hard
            stim["arousal"] *= 0.5
            stim["threat"] *= 0.3
            self.bus.publish("chat_stimulus", stim)

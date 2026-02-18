"""
Stimulus analyser — converts chat messages into emotion stimuli
driven by conversation context, memory, and user identity.

Rather than relying purely on keyword matching, this analyser
considers:

* **Conversation history** — sentiment trajectory across recent turns
  affects emotional momentum (e.g. multiple positive turns build joy).
* **User memory / rapport** — how many turns the user has had, how
  positive they've been historically, builds long-term rapport.
* **Who they're talking to** — the AI profile (from profile.json)
  sets baseline personality weights that bias which emotions fire.
* **Message content** — keyword / pattern analysis provides the
  immediate signal layer.

Publishes ``chat_stimulus`` events for the EmotionEngine.
"""

from __future__ import annotations

import json
import re
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

from .events import EventBus


# ── Keyword dictionaries (immediate signal layer) ────────────

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

_PERSONAL_WORDS = {
    "name", "who", "you", "your", "yourself",
    "feel", "think", "believe", "remember",
}

_PROFILE_PATH = Path("profile.json")


def _count_matches(text: str, wordset: set[str]) -> int:
    """Count how many words from the set appear in the text."""
    words = set(re.findall(r"[a-z]+|[!?:;<>()]+", text.lower()))
    return len(words & wordset)


def _has_question(text: str) -> bool:
    return "?" in text


def _exclamation_density(text: str) -> float:
    count = text.count("!")
    return min(count / max(len(text.split()), 1), 1.0)


# ── Personality bias from profile ────────────────────────────

def _load_profile_bias() -> Dict[str, float]:
    """
    Load the AI profile and derive personality-based emotion biases.

    Returns a dict of stimulus dimension adjustments.
    """
    defaults = {
        "valence_bias": 0.0,
        "arousal_bias": 0.0,
        "social_bias": 0.0,
        "engagement_bias": 0.0,
        "rapport_base": 0.2,
    }

    if not _PROFILE_PATH.exists():
        return defaults

    try:
        profile = json.loads(_PROFILE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return defaults

    persona = profile.get("persona", "").lower()
    traits = profile.get("personality_traits", "").lower()
    tone = profile.get("voice_tone", "").lower()
    combined = f"{persona} {traits} {tone}"

    biases = dict(defaults)

    # Warm / friendly persona → more positive, more social
    if any(w in combined for w in ("friendly", "warm", "kind", "caring", "sweet")):
        biases["valence_bias"] += 0.1
        biases["social_bias"] += 0.15
        biases["rapport_base"] = 0.35

    # Energetic / enthusiastic → higher arousal
    if any(w in combined for w in ("energetic", "excited", "enthusiastic", "lively")):
        biases["arousal_bias"] += 0.1
        biases["engagement_bias"] += 0.1

    # Calm / professional → lower arousal, more grounded
    if any(w in combined for w in ("calm", "professional", "serious", "composed")):
        biases["arousal_bias"] -= 0.08
        biases["valence_bias"] += 0.05

    # Curious / helpful → higher engagement
    if any(w in combined for w in ("curious", "helpful", "patient", "attentive")):
        biases["engagement_bias"] += 0.1
        biases["social_bias"] += 0.1

    return biases


# ── Main analyser ─────────────────────────────────────────────

class StimulusAnalyser:
    """
    Context-aware stimulus analyser.

    Considers conversation history, accumulated user rapport,
    the AI's personality profile, and message content to produce
    rich emotion stimuli.
    """

    def __init__(self, event_bus: EventBus):
        self.bus = event_bus
        self._turn_count: int = 0

        # Conversation memory — track recent sentiment trajectory
        self._recent_valences: deque[float] = deque(maxlen=15)
        self._recent_arousals: deque[float] = deque(maxlen=15)
        self._cumulative_rapport: float = 0.0
        self._consecutive_positive: int = 0
        self._consecutive_negative: int = 0

        # User identity tracking — who is talking to us?
        self._known_users: Dict[str, Dict[str, Any]] = {}
        self._current_user: Optional[str] = None

        # Load personality bias from AI profile
        self._profile_bias = _load_profile_bias()

        self.bus.subscribe("user_message", self._on_user_message)
        self.bus.subscribe("assistant_response", self._on_assistant_response)
        self.bus.subscribe("profile_saved", self._on_profile_saved)
        self.bus.subscribe("profile_selected", self._on_profile_selected)

    def analyse(self, text: str, is_user: bool = True) -> Dict[str, Any]:
        """
        Analyse a message using conversation context + content + profile.

        Returns a dict matching the ``Stimulus`` fields.
        """
        self._turn_count += 1
        bias = self._profile_bias

        # ── Content analysis (immediate signal) ────────────
        pos = _count_matches(text, _POSITIVE_WORDS)
        neg = _count_matches(text, _NEGATIVE_WORDS)
        threat = _count_matches(text, _THREAT_WORDS)
        curiosity = _count_matches(text, _CURIOSITY_WORDS)
        social = _count_matches(text, _SOCIAL_WORDS)
        excitement = _count_matches(text, _EXCITEMENT_WORDS)
        personal = _count_matches(text, _PERSONAL_WORDS)
        total_words = max(len(text.split()), 1)
        has_q = _has_question(text)

        # ── Valence: -1 to +1 (content + trajectory + profile) ──
        if pos + neg > 0:
            content_valence = (pos - neg) / (pos + neg)
        else:
            content_valence = 0.0

        # Conversation momentum — recent sentiment trajectory
        trajectory_valence = 0.0
        if self._recent_valences:
            # Weight recent turns more heavily
            weights = [0.5 ** i for i in range(len(self._recent_valences))]
            vals = list(self._recent_valences)
            vals.reverse()
            trajectory_valence = sum(w * v for w, v in zip(weights, vals)) / sum(weights)

        # Consecutive streak amplifier
        streak_boost = 0.0
        if self._consecutive_positive >= 3:
            streak_boost = min(self._consecutive_positive * 0.04, 0.15)
        elif self._consecutive_negative >= 3:
            streak_boost = max(-self._consecutive_negative * 0.04, -0.15)

        valence = (
            content_valence * 0.55          # immediate content
            + trajectory_valence * 0.25     # conversation momentum
            + streak_boost                   # streak amplification
            + bias["valence_bias"]           # personality baseline
        )
        valence = max(-1.0, min(1.0, valence))

        # Track for trajectory
        self._recent_valences.append(content_valence)
        if content_valence > 0.1:
            self._consecutive_positive += 1
            self._consecutive_negative = 0
        elif content_valence < -0.1:
            self._consecutive_negative += 1
            self._consecutive_positive = 0
        else:
            self._consecutive_positive = max(0, self._consecutive_positive - 1)
            self._consecutive_negative = max(0, self._consecutive_negative - 1)

        # ── Arousal: 0 to 1 (content + trajectory + profile) ────
        content_arousal = (
            _exclamation_density(text) * 0.3
            + min(excitement / max(total_words, 1), 1.0) * 0.3
            + (0.2 if len(text) > 200 else 0.0)
            + (0.15 if has_q else 0.0)
        )

        # Arousal momentum — escalating conversations raise intensity
        arousal_momentum = 0.0
        if self._recent_arousals:
            recent = list(self._recent_arousals)[-5:]
            arousal_momentum = sum(recent) / len(recent) * 0.2

        arousal = max(0.1, min(1.0,
            content_arousal + 0.15
            + arousal_momentum
            + bias["arousal_bias"]
        ))
        self._recent_arousals.append(content_arousal)

        # ── Social connection: 0 to 1 (content + rapport + profile) ──
        social_score = min(social / max(total_words * 0.3, 1), 1.0)
        personal_boost = min(personal / max(total_words * 0.3, 1), 0.3)
        social_connect = max(0.2, min(1.0,
            social_score * 0.4
            + personal_boost
            + min(self._cumulative_rapport * 0.15, 0.3)
            + bias["social_bias"]
            + 0.2  # base social (they're talking to us)
        ))

        # ── Novelty: 0 to 1 ────────────────────────────────
        novelty_raw = (
            (0.35 if has_q else 0.0)
            + min(curiosity / max(total_words * 0.3, 1), 1.0) * 0.3
            + (0.3 if self._turn_count <= 3 else 0.0)
            + (0.1 if len(text) > 100 else 0.0)
        )
        novelty = max(0.1, min(1.0, novelty_raw))

        # ── Threat: 0 to 1 ─────────────────────────────────
        threat_score = min(threat / max(total_words * 0.2, 1), 1.0)
        # Consecutive negative messages amplify perceived threat
        neg_amplifier = 1.0 + min(self._consecutive_negative * 0.1, 0.3)
        threat_val = max(0.0, min(1.0, threat_score * 0.7 * neg_amplifier))

        # ── Engagement: 0 to 1 (builds with conversation) ──
        engagement = max(0.3, min(1.0,
            0.25
            + (0.15 if len(text) > 50 else 0.0)
            + (0.15 if has_q else 0.0)
            + min(self._turn_count * 0.015, 0.25)
            + bias["engagement_bias"]
        ))

        # ── Rapport: 0 to 1 (builds over time, affected by who is talking) ──
        # Rapport accumulates — positive interactions build it up, negative erode it
        if is_user:
            if content_valence > 0.1:
                self._cumulative_rapport = min(
                    self._cumulative_rapport + 0.08, 5.0
                )
            elif content_valence < -0.2:
                self._cumulative_rapport = max(
                    self._cumulative_rapport - 0.03, 0.0
                )

        rapport = max(0.15, min(1.0,
            bias["rapport_base"]
            + min(self._turn_count * 0.02, 0.3)
            + min(self._cumulative_rapport * 0.1, 0.35)
            + social_connect * 0.15
        ))

        # ── Emotion hint (richer context-driven hints) ─────
        hint = self._derive_emotion_hint(
            valence, arousal, threat_val, has_q,
            curiosity, social_connect, rapport,
        )

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

    def _derive_emotion_hint(
        self,
        valence: float,
        arousal: float,
        threat: float,
        has_question: bool,
        curiosity: int,
        social_connect: float,
        rapport: float,
    ) -> Optional[str]:
        """
        Derive an emotion hint using the full dimensional analysis
        rather than simple keyword thresholds.
        """
        # High positive + high arousal → excitement/joy
        if valence > 0.5 and arousal > 0.6:
            return "joyful" if valence > 0.7 else "happy"

        # High positive + low arousal → contentment/peace
        if valence > 0.4 and arousal < 0.4:
            return "content" if rapport > 0.5 else "calm"

        # High positive + high social → warmth
        if valence > 0.3 and social_connect > 0.6:
            return "affectionate" if rapport > 0.6 else "thankful"

        # High arousal + question + curiosity → engagement
        if has_question and curiosity > 1 and arousal > 0.3:
            return "curious"

        # High threat + negative → fear
        if threat > 0.3 and valence < -0.3:
            return "afraid" if threat > 0.5 else "worried"

        # Negative + high arousal → anger/frustration
        if valence < -0.4 and arousal > 0.5:
            return "frustrated" if valence > -0.6 else "angry"

        # Negative + low arousal → sadness
        if valence < -0.4 and arousal < 0.4:
            return "sad" if valence < -0.6 else "disappointed"

        # Mildly negative → mild annoyance
        if valence < -0.2:
            return "annoyed"

        # Mildly positive with engagement → interest
        if valence > 0.1 and arousal > 0.3:
            return "interested"

        # High rapport + positive → motivated
        if rapport > 0.6 and valence > 0.2:
            return "motivated"

        return None

    # ── Event handlers ────────────────────────────────────────

    def _on_user_message(self, data: Dict[str, Any]) -> None:
        text = data.get("text", "")
        # Track who is talking if provided
        user = data.get("user", data.get("username"))
        if user:
            self._current_user = user
            if user not in self._known_users:
                self._known_users[user] = {"turns": 0, "avg_valence": 0.0}
            self._known_users[user]["turns"] += 1

        if text:
            stim = self.analyse(text, is_user=True)
            self.bus.publish("chat_stimulus", stim)

    def _on_assistant_response(self, data: Dict[str, Any]) -> None:
        text = data.get("text", "")
        if text:
            # Assistant responses have gentler stimulus effect
            stim = self.analyse(text, is_user=False)
            # Dampen — assistant's own words shouldn't swing emotions hard
            stim["arousal"] = round(stim["arousal"] * 0.5, 3)
            stim["threat"] = round(stim["threat"] * 0.3, 3)
            self.bus.publish("chat_stimulus", stim)

    def _on_profile_saved(self, _data: Dict[str, Any]) -> None:
        """Reload personality biases when the profile is saved."""
        self._profile_bias = _load_profile_bias()

    def _on_profile_selected(self, _data: Dict[str, Any]) -> None:
        """Reload personality biases when a different profile is selected."""
        self._profile_bias = _load_profile_bias()

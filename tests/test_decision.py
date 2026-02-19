"""
Tests for core/decision.py — emotion-driven response strategy engine.

Covers:
* ResponseStrategy.to_prompt_block() — every conditional branch
* _clamp() helper
* DecisionEngine.decide() — strategy values in range, flag logic,
  event-driven state tracking (error streak, negative streak)
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from core.decision import DecisionEngine, ResponseStrategy, _clamp


# ── Helpers ───────────────────────────────────────────────────────────

def make_mock_emotion(
    dominant_name: str = "calm",
    dominant_intensity: float = 0.3,
    mood: float = 0.1,
    top: list | None = None,
) -> MagicMock:
    """Create a minimal EmotionEngine mock with controllable state."""
    mock = MagicMock()
    mock.dominant_emotion.return_value = (dominant_name, dominant_intensity)
    mock.mood = mood
    mock.top_emotions.return_value = top or [(dominant_name, dominant_intensity)]
    return mock


@pytest.fixture
def neutral_emotion():
    return make_mock_emotion()


@pytest.fixture
def engine(bus, neutral_emotion):
    return DecisionEngine(bus, neutral_emotion)


# ── _clamp ────────────────────────────────────────────────────────────

class TestClamp:
    def test_within_bounds(self):
        assert _clamp(0.5) == 0.5

    def test_below_zero(self):
        assert _clamp(-0.1) == 0.0

    def test_above_one(self):
        assert _clamp(1.5) == 1.0

    def test_exactly_at_bounds(self):
        assert _clamp(0.0) == 0.0
        assert _clamp(1.0) == 1.0


# ── ResponseStrategy.to_prompt_block() ───────────────────────────────

class TestToPromptBlock:
    def test_all_defaults_returns_empty_string(self):
        s = ResponseStrategy()
        assert s.to_prompt_block() == ""

    def test_high_empathy_mentions_empathy(self):
        s = ResponseStrategy(empathy=0.8)
        block = s.to_prompt_block()
        assert "empathy" in block.lower() or "feeling" in block.lower()

    def test_low_empathy_says_direct(self):
        s = ResponseStrategy(empathy=0.2)
        assert "direct" in s.to_prompt_block().lower()

    def test_high_verbosity_says_detailed(self):
        s = ResponseStrategy(verbosity=0.8)
        block = s.to_prompt_block().lower()
        assert "thorough" in block or "detailed" in block

    def test_low_verbosity_says_concise(self):
        s = ResponseStrategy(verbosity=0.2)
        assert "concise" in s.to_prompt_block().lower()

    def test_high_assertiveness_says_confident(self):
        s = ResponseStrategy(assertiveness=0.8)
        block = s.to_prompt_block().lower()
        assert "confidence" in block or "confident" in block or "definitive" in block

    def test_low_assertiveness_uses_hedge_words(self):
        s = ResponseStrategy(assertiveness=0.2)
        block = s.to_prompt_block().lower()
        assert "softer" in block or "perhaps" in block or "might" in block

    def test_high_warmth_is_friendly(self):
        s = ResponseStrategy(warmth=0.8)
        block = s.to_prompt_block().lower()
        assert "warm" in block or "friendly" in block

    def test_low_warmth_is_professional(self):
        s = ResponseStrategy(warmth=0.2)
        assert "professional" in s.to_prompt_block().lower()

    def test_high_curiosity_asks_follow_up(self):
        s = ResponseStrategy(curiosity=0.7)
        block = s.to_prompt_block().lower()
        assert "follow-up" in block or "question" in block

    def test_high_caution_acknowledges_uncertainty(self):
        s = ResponseStrategy(caution=0.7)
        block = s.to_prompt_block().lower()
        assert "careful" in block or "uncertainty" in block or "measured" in block

    def test_should_apologise_flag(self):
        s = ResponseStrategy(should_apologise=True)
        block = s.to_prompt_block().lower()
        assert "confusion" in block or "acknowledge" in block or "error" in block

    def test_should_encourage_flag(self):
        s = ResponseStrategy(should_encourage=True)
        block = s.to_prompt_block().lower()
        assert "encouragement" in block or "positive" in block

    def test_should_clarify_flag(self):
        s = ResponseStrategy(should_clarify=True)
        assert "clarif" in s.to_prompt_block().lower()

    def test_should_slow_down_flag(self):
        s = ResponseStrategy(should_slow_down=True)
        block = s.to_prompt_block().lower()
        assert "overwhelmed" in block or "smaller" in block or "step" in block

    def test_active_strategy_starts_with_header(self):
        s = ResponseStrategy(empathy=0.9)
        assert s.to_prompt_block().startswith("[Response Strategy]")

    def test_multiple_flags_all_included(self):
        s = ResponseStrategy(
            empathy=0.9, verbosity=0.9,
            should_apologise=True, should_encourage=True,
        )
        block = s.to_prompt_block()
        assert len(block) > 50  # multiple instructions present


# ── DecisionEngine.decide() ───────────────────────────────────────────

class TestDecideReturnType:
    def test_returns_response_strategy_instance(self, engine):
        assert isinstance(engine.decide("hello"), ResponseStrategy)

    def test_all_float_fields_in_range(self, engine):
        s = engine.decide("hello")
        for attr in ("empathy", "verbosity", "assertiveness",
                     "warmth", "curiosity", "caution"):
            val = getattr(s, attr)
            assert 0.0 <= val <= 1.0, f"{attr}={val} out of [0,1]"

    def test_stores_last_strategy(self, engine):
        assert engine.last_strategy is None
        engine.decide("hi")
        assert isinstance(engine.last_strategy, ResponseStrategy)

    def test_reasoning_is_string(self, engine):
        s = engine.decide("hello")
        assert isinstance(s.reasoning, str)


class TestDecideEmpathy:
    def test_fear_emotions_raise_empathy(self, bus):
        emotion = make_mock_emotion(
            dominant_name="afraid",
            dominant_intensity=0.8,
            mood=-0.4,
            top=[("afraid", 0.8), ("anxious", 0.6), ("worried", 0.4)],
        )
        de = DecisionEngine(bus, emotion)
        s = de.decide("I am very scared")
        assert s.empathy > 0.5

    def test_positive_emotions_reduce_empathy(self, bus):
        emotion = make_mock_emotion(
            dominant_name="joyful",
            dominant_intensity=0.8,
            mood=0.7,
            top=[("joyful", 0.8), ("happy", 0.6)],
        )
        de = DecisionEngine(bus, emotion)
        s = de.decide("great day")
        assert s.empathy <= 0.5


class TestDecideVerbosity:
    def test_short_message_reduces_verbosity(self, bus):
        emotion = make_mock_emotion()
        de = DecisionEngine(bus, emotion)
        s_short = de.decide("hi")
        s_long = de.decide(
            "Can you explain in great detail how this whole thing works "
            "and what I should do next to make it better?"
        )
        assert s_short.verbosity <= s_long.verbosity

    def test_question_can_increase_verbosity(self, bus):
        emotion = make_mock_emotion()
        de = DecisionEngine(bus, emotion)
        s_no_q = de.decide("hello there friend")
        s_q = de.decide("Can you explain everything?")
        assert s_q.verbosity >= s_no_q.verbosity


class TestDecideFlags:
    def test_apologise_after_two_errors(self, bus):
        de = DecisionEngine(bus, make_mock_emotion())
        bus.publish("activity_log", {"text": "[Error] first failure"})
        bus.publish("activity_log", {"text": "[Error] second failure"})
        s = de.decide("")
        assert s.should_apologise is True

    def test_no_apologise_after_one_error(self, bus):
        de = DecisionEngine(bus, make_mock_emotion())
        bus.publish("activity_log", {"text": "[Error] only one"})
        s = de.decide("")
        assert s.should_apologise is False

    def test_error_streak_resets_on_success(self, bus):
        de = DecisionEngine(bus, make_mock_emotion())
        bus.publish("activity_log", {"text": "[Error] err1"})
        bus.publish("activity_log", {"text": "[Error] err2"})
        bus.publish("assistant_response", {"text": "all good"})
        s = de.decide("")
        assert s.should_apologise is False

    def test_non_error_logs_dont_increment_streak(self, bus):
        de = DecisionEngine(bus, make_mock_emotion())
        bus.publish("activity_log", {"text": "Connected to server"})
        bus.publish("activity_log", {"text": "Model loaded"})
        s = de.decide("")
        assert s.should_apologise is False

    def test_clarify_on_very_short_ambiguous_text(self, bus):
        de = DecisionEngine(bus, make_mock_emotion())
        # Trigger turn_count > 1 first
        de.decide("hello there how are you doing today")
        s = de.decide("??")  # very short + multiple question marks
        assert s.should_clarify is True

    def test_consecutive_negative_stimulus_tracked(self, bus):
        emotion = make_mock_emotion(
            dominant_name="afraid",
            dominant_intensity=0.5,
            mood=-0.4,
            top=[("afraid", 0.5), ("anxious", 0.3)],
        )
        de = DecisionEngine(bus, emotion)
        bus.publish("chat_stimulus", {"valence": -0.5})
        bus.publish("chat_stimulus", {"valence": -0.6})
        bus.publish("chat_stimulus", {"valence": -0.7})
        s = de.decide("")
        # consecutive_negative >= 2 → empathy branch fires
        assert s.empathy > 0.5

    def test_positive_stimulus_resets_negative_streak(self, bus):
        de = DecisionEngine(bus, make_mock_emotion())
        bus.publish("chat_stimulus", {"valence": -0.5})
        bus.publish("chat_stimulus", {"valence": -0.6})
        bus.publish("chat_stimulus", {"valence": 0.8})  # positive resets streak
        s = de.decide("")
        # Streak reset — empathy should be at default (no streak boost)
        assert s.empathy <= 0.6

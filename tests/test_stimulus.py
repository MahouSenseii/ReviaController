"""
Tests for core/stimulus.py — context-aware emotion stimulus analyser.

Covers:
* Helper functions: _count_matches, _has_question, _exclamation_density
* StimulusAnalyser.analyse() — valence, arousal, threat, output shape
* Trajectory / streak mechanics
* _derive_emotion_hint branches
* Assistant-response dampening via event bus
"""

from __future__ import annotations

import pytest

from core.stimulus import (
    StimulusAnalyser,
    _count_matches,
    _exclamation_density,
    _has_question,
)


# ── _count_matches ────────────────────────────────────────────────────

class TestCountMatches:
    def test_single_match(self):
        assert _count_matches("I love this", {"love"}) == 1

    def test_no_match(self):
        assert _count_matches("hello world", {"xyz"}) == 0

    def test_case_insensitive(self):
        assert _count_matches("LOVE IT", {"love"}) == 1

    def test_multiple_words_from_set(self):
        assert _count_matches("I love and hate this", {"love", "hate"}) == 2

    def test_repeated_word_counted_once(self):
        # Set intersection deduplicates — "love love love" still intersects to 1
        assert _count_matches("love love love", {"love"}) == 1

    def test_empty_text(self):
        assert _count_matches("", {"love"}) == 0

    def test_empty_wordset(self):
        assert _count_matches("hello world", set()) == 0

    def test_punctuation_tokens(self):
        # "?" is tokenised by the regex as a punctuation token
        assert _count_matches("really?", {"?"}) == 1


# ── _has_question ─────────────────────────────────────────────────────

class TestHasQuestion:
    def test_with_question_mark(self):
        assert _has_question("how are you?") is True

    def test_without_question_mark(self):
        assert _has_question("hello world") is False

    def test_empty_string(self):
        assert _has_question("") is False


# ── _exclamation_density ──────────────────────────────────────────────

class TestExclamationDensity:
    def test_no_exclamation_is_zero(self):
        assert _exclamation_density("hello world") == 0.0

    def test_single_exclamation(self):
        result = _exclamation_density("great!")
        assert result > 0.0

    def test_clamped_to_one(self):
        assert _exclamation_density("!!!!!") <= 1.0

    def test_more_exclamations_higher_density(self):
        low = _exclamation_density("good")
        high = _exclamation_density("wow!!! amazing!!! yes!!!")
        assert high >= low


# ── StimulusAnalyser fixture ──────────────────────────────────────────

@pytest.fixture
def analyser(bus):
    return StimulusAnalyser(bus)


# ── analyse() — output shape ──────────────────────────────────────────

class TestAnalyseOutputShape:
    def test_all_required_keys_present(self, analyser):
        result = analyser.analyse("hello")
        for key in (
            "valence", "arousal", "social_connect", "novelty",
            "threat", "engagement", "rapport", "emotion_hint",
        ):
            assert key in result

    def test_numeric_fields_are_floats(self, analyser):
        result = analyser.analyse("hello")
        for key in ("valence", "arousal", "social_connect",
                    "novelty", "threat", "engagement", "rapport"):
            assert isinstance(result[key], float), f"{key} is not a float"

    def test_emotion_hint_is_none_or_str(self, analyser):
        result = analyser.analyse("the cat sat")
        assert result["emotion_hint"] is None or isinstance(result["emotion_hint"], str)


# ── analyse() — valence ───────────────────────────────────────────────

class TestAnalyseValence:
    def test_positive_message_gives_positive_valence(self, analyser):
        result = analyser.analyse("thank you so much I love this amazing work")
        assert result["valence"] > 0.0

    def test_negative_message_gives_negative_valence(self, analyser):
        result = analyser.analyse("this is terrible awful broken and bad wrong")
        assert result["valence"] < 0.0

    def test_valence_clamped_to_unit_range(self, analyser):
        result = analyser.analyse("amazing wonderful fantastic brilliant excellent perfect")
        assert -1.0 <= result["valence"] <= 1.0

    def test_neutral_message_near_zero(self, analyser):
        # "the cat sat on the mat" has no sentiment words
        result = analyser.analyse("the cat sat on the mat")
        assert -0.4 < result["valence"] < 0.4


# ── analyse() — arousal ───────────────────────────────────────────────

class TestAnalyseArousal:
    def test_arousal_in_unit_range(self, analyser):
        result = analyser.analyse("hello")
        assert 0.0 <= result["arousal"] <= 1.0

    def test_exclamations_raise_arousal(self, analyser):
        low = analyser.analyse("okay that is fine")
        high = analyser.analyse("wow!!! amazing!!! incredible!!!")
        assert high["arousal"] >= low["arousal"]


# ── analyse() — threat ────────────────────────────────────────────────

class TestAnalyseThreat:
    def test_threat_words_raise_threat_score(self, analyser):
        low = analyser.analyse("hello friend how are you")
        high = analyser.analyse("error crash broken danger bug fail warning critical")
        assert high["threat"] >= low["threat"]

    def test_threat_clamped(self, analyser):
        result = analyser.analyse(
            "error crash fail bug broken danger warning critical urgent attack virus"
        )
        assert 0.0 <= result["threat"] <= 1.0


# ── analyse() — trajectory / streak ──────────────────────────────────

class TestTrajectoryMechanics:
    def test_consecutive_positives_sustain_valence(self, bus):
        a = StimulusAnalyser(bus)
        r1 = a.analyse("great thanks awesome love this")
        r2 = a.analyse("wonderful brilliant fantastic perfect")
        r3 = a.analyse("amazing excellent superb beautiful")
        # Trajectory momentum means later turns stay positive or rise
        assert r3["valence"] >= r1["valence"] - 0.1

    def test_consecutive_negatives_sustain_negative_valence(self, bus):
        a = StimulusAnalyser(bus)
        r1 = a.analyse("bad terrible broken awful wrong")
        r2 = a.analyse("horrible stupid useless fail crash")
        r3 = a.analyse("worst ever annoyed frustrated angry")
        assert r3["valence"] <= r1["valence"] + 0.1

    def test_rapport_builds_over_turns(self, bus):
        a = StimulusAnalyser(bus)
        r_early = a.analyse("hi")
        for _ in range(10):
            a.analyse("thank you love this great awesome")
        r_late = a.analyse("hello again")
        assert r_late["rapport"] >= r_early["rapport"]


# ── _derive_emotion_hint ──────────────────────────────────────────────

class TestDeriveEmotionHint:
    def test_high_positive_arousal_gives_positive_hint(self, analyser):
        result = analyser.analyse("wow amazing fantastic love it!!!")
        # Should hit joyful/happy/interested branch
        positive_hints = {
            "joyful", "happy", "interested", "affectionate",
            "thankful", "motivated", "curious", "content", "calm",
        }
        hint = result["emotion_hint"]
        assert hint is None or hint in positive_hints

    def test_negative_threat_gives_fear_or_negative_hint(self, analyser):
        result = analyser.analyse(
            "error crash danger attack virus fail broken wrong bad awful"
        )
        negative_hints = {
            "afraid", "worried", "annoyed", "frustrated", "angry",
            "sad", "disappointed",
        }
        hint = result["emotion_hint"]
        assert hint is None or hint in negative_hints

    def test_question_with_curiosity_may_hint_curious(self, analyser):
        result = analyser.analyse("what do you think? how does this work? why?")
        # High curiosity + question → may hint "curious"
        hint = result["emotion_hint"]
        assert hint is None or isinstance(hint, str)


# ── Assistant-response dampening ──────────────────────────────────────

class TestAssistantResponseDampening:
    def test_arousal_is_dampened(self, bus):
        # Hold a reference — without it the QObject's slots are GC'd
        analyser = StimulusAnalyser(bus)
        stimuli = []
        bus.subscribe("chat_stimulus", stimuli.append)
        bus.publish("assistant_response", {
            "text": "wow amazing fantastic great!!!"
        })
        assert len(stimuli) == 1
        # Arousal is multiplied by 0.5 — so always ≤ 0.6 for typical input
        assert stimuli[0]["arousal"] <= 0.6

    def test_threat_is_dampened(self, bus):
        analyser = StimulusAnalyser(bus)  # keep reference
        stimuli = []
        bus.subscribe("chat_stimulus", stimuli.append)
        bus.publish("assistant_response", {
            "text": "error crash broken danger fail warning"
        })
        assert len(stimuli) == 1
        # Threat multiplied by 0.3 — should be well below 0.5
        assert stimuli[0]["threat"] <= 0.5

    def test_user_message_event_triggers_stimulus(self, bus):
        analyser = StimulusAnalyser(bus)  # keep reference
        stimuli = []
        bus.subscribe("chat_stimulus", stimuli.append)
        bus.publish("user_message", {"text": "hello there"})
        assert len(stimuli) == 1

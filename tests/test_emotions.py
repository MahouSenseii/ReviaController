"""
Tests for core/emotions.py — VAD profiles and lateral influence matrix.

These tests protect against:
* Hand-authored VAD values drifting outside their valid ranges.
* EMOTION_NAMES / EMOTION_PROFILES / EMOTION_INDEX falling out of sync.
* build_influence_matrix() producing wrong shapes or out-of-range weights.
"""

from __future__ import annotations

from core.emotions import (
    EMOTION_PROFILES,
    EMOTION_NAMES,
    EMOTION_INDEX,
    NUM_EMOTIONS,
    build_influence_matrix,
)


# ── Catalogue consistency ─────────────────────────────────────────────

class TestCatalogueConsistency:
    def test_num_emotions_matches_profiles(self):
        assert len(EMOTION_PROFILES) == NUM_EMOTIONS

    def test_num_emotions_matches_names(self):
        assert len(EMOTION_NAMES) == NUM_EMOTIONS

    def test_emotion_index_length(self):
        assert len(EMOTION_INDEX) == NUM_EMOTIONS

    def test_all_names_in_profiles(self):
        for name in EMOTION_NAMES:
            assert name in EMOTION_PROFILES, f"'{name}' in EMOTION_NAMES but not in EMOTION_PROFILES"

    def test_all_profile_keys_in_names(self):
        for name in EMOTION_PROFILES:
            assert name in EMOTION_NAMES, f"'{name}' in EMOTION_PROFILES but not in EMOTION_NAMES"

    def test_index_values_are_valid(self):
        for name, idx in EMOTION_INDEX.items():
            assert 0 <= idx < NUM_EMOTIONS, f"Index {idx} for '{name}' out of range"

    def test_index_is_bijective(self):
        indices = list(EMOTION_INDEX.values())
        assert len(set(indices)) == len(indices), "Duplicate indices in EMOTION_INDEX"


# ── VAD range validation ──────────────────────────────────────────────

class TestVADRanges:
    def test_valence_in_range(self):
        for name, ep in EMOTION_PROFILES.items():
            assert -1.0 <= ep.valence <= 1.0, (
                f"'{name}' valence={ep.valence} outside [-1, 1]"
            )

    def test_arousal_in_range(self):
        for name, ep in EMOTION_PROFILES.items():
            assert 0.0 <= ep.arousal <= 1.0, (
                f"'{name}' arousal={ep.arousal} outside [0, 1]"
            )

    def test_dominance_in_range(self):
        for name, ep in EMOTION_PROFILES.items():
            assert 0.0 <= ep.dominance <= 1.0, (
                f"'{name}' dominance={ep.dominance} outside [0, 1]"
            )

    def test_category_is_string(self):
        for name, ep in EMOTION_PROFILES.items():
            assert isinstance(ep.category, str) and ep.category, (
                f"'{name}' has empty or non-string category"
            )


# ── EmotionProfile helpers ────────────────────────────────────────────

class TestEmotionProfileHelpers:
    def test_as_vector_returns_vad_tuple(self):
        ep = EMOTION_PROFILES["happy"]
        v, a, d = ep.as_vector()
        assert v == ep.valence
        assert a == ep.arousal
        assert d == ep.dominance

    def test_profile_is_frozen(self):
        ep = EMOTION_PROFILES["calm"]
        try:
            ep.valence = 0.0  # type: ignore[misc]
            assert False, "Expected FrozenInstanceError"
        except Exception:
            pass  # frozen dataclass raises on mutation


# ── Lateral influence matrix ──────────────────────────────────────────

class TestInfluenceMatrix:
    def setup_method(self):
        self.mat = build_influence_matrix()

    def test_matrix_has_correct_rows(self):
        assert len(self.mat) == NUM_EMOTIONS

    def test_matrix_has_correct_cols(self):
        for row in self.mat:
            assert len(row) == NUM_EMOTIONS

    def test_all_values_in_range(self):
        for r, row in enumerate(self.mat):
            for c, val in enumerate(row):
                assert -1.0 <= val <= 1.0, (
                    f"mat[{r}][{c}] = {val} outside [-1, 1]"
                )

    def test_known_amplification(self):
        # afraid → anxious (weight 0.60)
        afraid_idx = EMOTION_INDEX["afraid"]
        anxious_idx = EMOTION_INDEX["anxious"]
        assert self.mat[anxious_idx][afraid_idx] > 0

    def test_known_suppression(self):
        # happy suppresses sad (weight -0.60)
        happy_idx = EMOTION_INDEX["happy"]
        sad_idx = EMOTION_INDEX["sad"]
        assert self.mat[sad_idx][happy_idx] < 0

    def test_another_amplification(self):
        # frustrated → angry (weight 0.55)
        frustrated_idx = EMOTION_INDEX["frustrated"]
        angry_idx = EMOTION_INDEX["angry"]
        assert self.mat[angry_idx][frustrated_idx] > 0

    def test_another_suppression(self):
        # calm suppresses anxious (weight -0.50)
        calm_idx = EMOTION_INDEX["calm"]
        anxious_idx = EMOTION_INDEX["anxious"]
        assert self.mat[anxious_idx][calm_idx] < 0

    def test_build_is_deterministic(self):
        mat2 = build_influence_matrix()
        assert self.mat == mat2

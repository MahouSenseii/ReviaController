"""
Tests for core/emotion_nn.py — pure-Python neural network.

Covers:
* Math helpers (_relu, _sigmoid, _clamp, _mat_vec, _vec_add)
* Weight initialisation helpers
* EmotionNeuralNetwork.forward() — shape, value range, lateral influence
* EmotionNeuralNetwork.adapt() — weights shift toward target
* Persistence — save/load round-trip, corrupt file, missing file
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from core.emotion_nn import (
    INPUT_DIM,
    EmotionNeuralNetwork,
    _clamp,
    _init_bias,
    _init_weights,
    _mat_vec,
    _relu,
    _sigmoid,
    _vec_add,
)
from core.emotions import EMOTION_INDEX, NUM_EMOTIONS


# ── _relu ─────────────────────────────────────────────────────────────

class TestRelu:
    def test_positive_passthrough(self):
        assert _relu(3.7) == 3.7

    def test_zero_passthrough(self):
        assert _relu(0.0) == 0.0

    def test_negative_clamped_to_zero(self):
        assert _relu(-5.0) == 0.0

    def test_very_negative(self):
        assert _relu(-1e9) == 0.0


# ── _sigmoid ──────────────────────────────────────────────────────────

class TestSigmoid:
    def test_zero_gives_half(self):
        assert abs(_sigmoid(0.0) - 0.5) < 1e-9

    def test_output_always_in_unit_range(self):
        for x in [-1000.0, -12.0, -1.0, 0.0, 1.0, 12.0, 1000.0]:
            result = _sigmoid(x)
            assert 0.0 <= result <= 1.0

    def test_clamping_at_positive_extreme(self):
        # Values beyond ±12 are clamped before exp, so sigmoid(13)==sigmoid(12)
        assert _sigmoid(13.0) == _sigmoid(12.0)

    def test_clamping_at_negative_extreme(self):
        assert _sigmoid(-13.0) == _sigmoid(-12.0)

    def test_monotone(self):
        assert _sigmoid(1.0) > _sigmoid(0.0) > _sigmoid(-1.0)


# ── _clamp ────────────────────────────────────────────────────────────

class TestClamp:
    def test_within_default_range(self):
        assert _clamp(0.5) == 0.5

    def test_below_default_min(self):
        assert _clamp(-0.1) == 0.0

    def test_above_default_max(self):
        assert _clamp(1.1) == 1.0

    def test_exactly_at_bounds(self):
        assert _clamp(0.0) == 0.0
        assert _clamp(1.0) == 1.0

    def test_custom_bounds_low(self):
        assert _clamp(-15.0, lo=-10.0, hi=3.0) == -10.0

    def test_custom_bounds_high(self):
        assert _clamp(5.0, lo=-10.0, hi=3.0) == 3.0

    def test_custom_bounds_within(self):
        assert _clamp(0.0, lo=-10.0, hi=3.0) == 0.0


# ── _mat_vec ──────────────────────────────────────────────────────────

class TestMatVec:
    def test_identity_matrix(self):
        mat = [[1.0, 0.0], [0.0, 1.0]]
        vec = [3.0, 7.0]
        assert _mat_vec(mat, vec) == [3.0, 7.0]

    def test_scaling(self):
        mat = [[2.0, 0.0], [0.0, 3.0]]
        vec = [1.0, 1.0]
        assert _mat_vec(mat, vec) == [2.0, 3.0]

    def test_general(self):
        mat = [[1.0, 2.0], [3.0, 4.0]]
        vec = [1.0, 1.0]
        result = _mat_vec(mat, vec)
        assert abs(result[0] - 3.0) < 1e-9
        assert abs(result[1] - 7.0) < 1e-9

    def test_zero_vector(self):
        mat = [[1.0, 2.0], [3.0, 4.0]]
        vec = [0.0, 0.0]
        assert _mat_vec(mat, vec) == [0.0, 0.0]


# ── _vec_add ──────────────────────────────────────────────────────────

class TestVecAdd:
    def test_basic_addition(self):
        assert _vec_add([1.0, 2.0], [3.0, 4.0]) == [4.0, 6.0]

    def test_with_negatives(self):
        result = _vec_add([1.0, -1.0], [-1.0, 1.0])
        assert result == [0.0, 0.0]

    def test_empty_vectors(self):
        assert _vec_add([], []) == []


# ── _init_weights ─────────────────────────────────────────────────────

class TestInitWeights:
    def test_correct_shape(self):
        w = _init_weights(4, 3, seed=0)
        assert len(w) == 4
        assert all(len(row) == 3 for row in w)

    def test_deterministic_with_same_seed(self):
        w1 = _init_weights(8, 32, seed=42)
        w2 = _init_weights(8, 32, seed=42)
        assert w1 == w2

    def test_different_seeds_differ(self):
        w1 = _init_weights(8, 32, seed=1)
        w2 = _init_weights(8, 32, seed=2)
        assert w1 != w2

    def test_xavier_bounds(self):
        rows, cols = 8, 32
        limit = math.sqrt(6.0 / (rows + cols))
        w = _init_weights(rows, cols, seed=42)
        for row in w:
            for val in row:
                assert -limit <= val <= limit

    def test_init_bias_zeros(self):
        b = _init_bias(10)
        assert b == [0.0] * 10


# ── EmotionNeuralNetwork — forward pass ──────────────────────────────

# Use a path that never exists so the network always initialises fresh.
_NO_FILE = Path("/tmp/_test_nn_no_file_UNUSED.json")


class TestForward:
    def setup_method(self):
        self.nn = EmotionNeuralNetwork(weight_path=_NO_FILE)

    def _stim(self, val: float = 0.0) -> list[float]:
        return [val] * INPUT_DIM

    def test_output_length_equals_num_emotions(self):
        out = self.nn.forward(self._stim())
        assert len(out) == NUM_EMOTIONS

    def test_all_outputs_in_unit_range(self):
        out = self.nn.forward(self._stim(0.5))
        assert all(0.0 <= v <= 1.0 for v in out), (
            f"Values out of [0,1]: {[v for v in out if not (0<=v<=1)]}"
        )

    def test_wrong_stimulus_dim_raises(self):
        with pytest.raises(AssertionError):
            self.nn.forward([0.0] * (INPUT_DIM - 1))

    def test_wrong_stimulus_too_long_raises(self):
        with pytest.raises(AssertionError):
            self.nn.forward([0.0] * (INPUT_DIM + 1))

    def test_positive_valence_boosts_happy(self):
        # Strongly positive stimulus (valence=1, mood=1) should score
        # 'happy' higher than a strongly negative stimulus.
        pos = self.nn.forward([1.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 1.0])
        neg = self.nn.forward([-1.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, -1.0])
        assert pos[EMOTION_INDEX["happy"]] > neg[EMOTION_INDEX["happy"]]

    def test_negative_valence_boosts_sad(self):
        neg = self.nn.forward([-1.0, 0.3, 0.2, 0.1, 0.8, 0.2, 0.2, -1.0])
        pos = self.nn.forward([1.0, 0.3, 0.2, 0.1, 0.0, 0.2, 0.2, 1.0])
        assert neg[EMOTION_INDEX["sad"]] > pos[EMOTION_INDEX["sad"]]

    def test_lateral_influence_changes_output(self):
        stim = self._stim(0.3)
        no_lateral = self.nn.forward(stim, current_state=None)
        current = [0.9] * NUM_EMOTIONS
        with_lateral = self.nn.forward(stim, current_state=current)
        assert no_lateral != with_lateral

    def test_wrong_length_current_state_is_ignored(self):
        stim = self._stim(0.3)
        no_lateral = self.nn.forward(stim, current_state=None)
        bad_state = self.nn.forward(stim, current_state=[0.5] * (NUM_EMOTIONS - 1))
        assert no_lateral == bad_state

    def test_lateral_strength_zero_matches_no_state(self):
        stim = self._stim(0.3)
        current = [0.9] * NUM_EMOTIONS
        # lateral_strength=0 means influence term is 0 → same as no state
        no_influence = self.nn.forward(stim, current_state=current, lateral_strength=0.0)
        no_state = self.nn.forward(stim, current_state=None)
        for a, b in zip(no_influence, no_state):
            assert abs(a - b) < 1e-9


# ── EmotionNeuralNetwork — adapt ──────────────────────────────────────

class TestAdapt:
    def setup_method(self):
        self.nn = EmotionNeuralNetwork(weight_path=_NO_FILE)

    def test_adapt_raises_target_emotion(self):
        stim = [0.5] * INPUT_DIM
        target = "happy"
        tidx = EMOTION_INDEX[target]
        before = self.nn.forward(stim)[tidx]
        for _ in range(100):
            self.nn.adapt(stim, target, strength=1.0)
        after = self.nn.forward(stim)[tidx]
        assert after > before

    def test_adapt_unknown_emotion_is_noop(self):
        stim = [0.3] * INPUT_DIM
        before = self.nn.forward(stim)
        self.nn.adapt(stim, "EMOTION_THAT_DOES_NOT_EXIST")
        after = self.nn.forward(stim)
        assert before == after

    def test_adapt_strength_zero_is_noop(self):
        stim = [0.5] * INPUT_DIM
        before_bypass = [row[:] for row in self.nn.w_bypass]
        self.nn.adapt(stim, "happy", strength=0.0)
        assert self.nn.w_bypass == before_bypass


# ── EmotionNeuralNetwork — persistence ───────────────────────────────

class TestPersistence:
    def test_save_and_reload_preserves_hyperparams(self, tmp_path):
        path = tmp_path / "weights.json"
        nn1 = EmotionNeuralNetwork(weight_path=path)
        nn1.bypass_strength = 0.42
        nn1.learning_rate = 0.0123
        nn1.save()

        nn2 = EmotionNeuralNetwork(weight_path=path)
        assert abs(nn2.bypass_strength - 0.42) < 1e-9
        assert abs(nn2.learning_rate - 0.0123) < 1e-9

    def test_save_and_reload_preserves_weights(self, tmp_path):
        path = tmp_path / "weights.json"
        nn1 = EmotionNeuralNetwork(weight_path=path)
        # Mutate a weight so we can detect if it was restored
        nn1.w_bypass[0][0] = 9.87654
        nn1.save()

        nn2 = EmotionNeuralNetwork(weight_path=path)
        assert abs(nn2.w_bypass[0][0] - 9.87654) < 1e-9

    def test_missing_file_initialises_fresh(self, tmp_path):
        path = tmp_path / "does_not_exist.json"
        nn = EmotionNeuralNetwork(weight_path=path)
        # Fresh init produces correctly shaped weights
        assert len(nn.w1) == 32        # HIDDEN_1
        assert len(nn.w1[0]) == INPUT_DIM

    def test_corrupt_file_falls_back_to_fresh(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("this is not valid json", encoding="utf-8")
        nn = EmotionNeuralNetwork(weight_path=path)
        assert len(nn.w1) == 32

    def test_save_to_bad_path_does_not_raise(self):
        # Writing to a non-existent directory should be silently swallowed
        nn = EmotionNeuralNetwork(weight_path=Path("/nonexistent_dir/weights.json"))
        nn.save()  # must not raise

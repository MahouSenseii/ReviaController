"""
Pure-Python neural network for mapping stimulus vectors to emotion
activations.

No external dependencies (no NumPy / PyTorch) — only the ``math``
standard-library module — so the controller stays lightweight and
can run anywhere PyQt6 runs.

Architecture
------------
::

    Stimulus (8-dim)
         │
         ├──────────────────────────────────────┐
         │                                      │
    ┌────▼────┐                           ┌─────▼─────┐
    │ Dense 1  │  8 → 32  (ReLU)          │ VAD bypass │ 8 → N (direct)
    └────┬────┘                           └─────┬─────┘
    ┌────▼────┐                                 │
    │ Dense 2  │  32 → 48 (ReLU)                │
    └────┬────┘                                 │
    ┌────▼────┐                                 │
    │ Dense 3  │  48 → N  (pre-activation)      │
    └────┬────┘                                 │
         │              ┌───────────────────────┘
         └──────┬───────┘
                ▼
         sigmoid(hidden + vad_bypass)
                │
    lateral influence bias (from current emotional state)
                │
         clamp to [0, 1]
                ▼
        Emotion output (N-dim)

The **VAD bypass** is a direct projection from the stimulus to each
emotion, weighted by how well the stimulus matches that emotion's
Valence-Arousal-Dominance profile.  This guarantees psychologically
plausible behaviour from the first forward pass, while the hidden
layers learn subtler patterns over time through Hebbian adaptation.

Stimulus dimensions (defined by EmotionEngine):
    0  valence        (−1 … +1)  sentiment of the interaction
    1  arousal        (0 … 1)    energy / intensity
    2  social_connect (0 … 1)    social engagement level
    3  novelty        (0 … 1)    surprise / unexpectedness
    4  threat         (0 … 1)    perceived negativity / danger
    5  engagement     (0 … 1)    user activity level
    6  rapport        (0 … 1)    relationship quality over time
    7  current_mood   (−1 … +1)  slow-moving baseline mood
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import List, Optional

from .emotions import (
    EMOTION_PROFILES,
    EMOTION_NAMES,
    NUM_EMOTIONS,
    build_influence_matrix,
)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

INPUT_DIM = 8                   # stimulus vector size
HIDDEN_1 = 32
HIDDEN_2 = 48
OUTPUT_DIM = NUM_EMOTIONS       # one activation per emotion

_WEIGHT_PATH = Path("emotion_weights.json")


# ------------------------------------------------------------------
# Math helpers (pure Python)
# ------------------------------------------------------------------

def _relu(x: float) -> float:
    return x if x > 0.0 else 0.0


def _sigmoid(x: float) -> float:
    x = max(-12.0, min(12.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# ------------------------------------------------------------------
# Vector / matrix ops (list-of-lists)
# ------------------------------------------------------------------

def _mat_vec(mat: List[List[float]], vec: List[float]) -> List[float]:
    """Matrix × vector (no numpy)."""
    return [sum(row[j] * vec[j] for j in range(len(vec))) for row in mat]


def _vec_add(a: List[float], b: List[float]) -> List[float]:
    return [ai + bi for ai, bi in zip(a, b)]


# ------------------------------------------------------------------
# Weight initialisation
# ------------------------------------------------------------------

def _init_weights(rows: int, cols: int, seed: int) -> List[List[float]]:
    """Xavier-uniform init with deterministic seed."""
    rng = random.Random(seed)
    limit = math.sqrt(6.0 / (rows + cols))
    return [[rng.uniform(-limit, limit) for _ in range(cols)] for _ in range(rows)]


def _init_bias(size: int) -> List[float]:
    return [0.0] * size


def _build_vad_bypass() -> List[List[float]]:
    """
    Build the VAD bypass weight matrix (N_emotions × INPUT_DIM).

    Each emotion gets a direct linear projection from the 8-dim
    stimulus.  The weights encode how each stimulus dimension maps
    to each emotion based on its psychological profile:

    * **valence**  (dim 0) — positive emotions get positive weight,
      negative emotions get negative weight.
    * **arousal**  (dim 1) — high-arousal emotions get positive
      weight, low-arousal emotions get negative.
    * **social**   (dim 2) — warmth/love emotions boosted by social.
    * **novelty**  (dim 3) — curiosity/surprise boosted.
    * **threat**   (dim 4) — fear/anxiety emotions boosted.
    * **engagement** (dim 5) — drive/engagement emotions boosted.
    * **rapport**  (dim 6) — love/warmth emotions boosted.
    * **mood**     (dim 7) — acts like a second valence channel.
    """
    bypass: List[List[float]] = []

    # Category-specific tuning for stimulus dimensions beyond VAD
    _cat_social    = {"love": 0.6, "joy": 0.3, "shame": -0.2}
    _cat_novelty   = {"engagement": 0.5, "fear": 0.2}
    _cat_threat    = {"fear": 0.8, "anger": 0.3, "sadness": 0.2,
                      "joy": -0.3, "peace": -0.4, "love": -0.2}
    _cat_engage    = {"drive": 0.5, "engagement": 0.4, "fatigue": -0.4,
                      "detachment": -0.5}
    _cat_rapport   = {"love": 0.6, "peace": 0.3, "joy": 0.3,
                      "sadness": -0.1}

    for ename in EMOTION_NAMES:
        ep = EMOTION_PROFILES[ename]
        cat = ep.category
        row = [0.0] * INPUT_DIM

        # dim 0: valence — direct match
        row[0] = ep.valence * 1.5

        # dim 1: arousal — centred around 0.5
        row[1] = (ep.arousal - 0.5) * 1.2

        # dim 2: social connection
        row[2] = _cat_social.get(cat, 0.0)

        # dim 3: novelty
        row[3] = _cat_novelty.get(cat, 0.0)

        # dim 4: threat
        row[4] = _cat_threat.get(cat, 0.0)

        # dim 5: engagement
        row[5] = _cat_engage.get(cat, 0.0)

        # dim 6: rapport
        row[6] = _cat_rapport.get(cat, 0.0)

        # dim 7: mood — similar to valence but gentler
        row[7] = ep.valence * 0.8

        bypass.append(row)

    return bypass


def _vad_seeded_output_bias() -> List[float]:
    """
    Set output biases so that extreme emotions need stronger stimuli.
    """
    biases: List[float] = []
    for ename in EMOTION_NAMES:
        ep = EMOTION_PROFILES[ename]
        dist = math.sqrt(
            ep.valence ** 2
            + (ep.arousal - 0.5) ** 2
            + (ep.dominance - 0.5) ** 2
        )
        biases.append(-dist * 0.5)
    return biases


# ------------------------------------------------------------------
# Network class
# ------------------------------------------------------------------

class EmotionNeuralNetwork:
    """
    Feed-forward network with a VAD bypass that maps stimulus vectors
    to emotion activations.

    The VAD bypass ensures psychologically plausible output from the
    start, while the hidden layers learn nuanced patterns through
    Hebbian adaptation over time.

    All state is plain Python lists — no external tensor libraries.
    Weights can be saved / loaded to JSON for persistence.
    """

    def __init__(self, weight_path: Optional[Path] = None):
        self._path = weight_path or _WEIGHT_PATH

        # Hidden-path weights
        self.w1: List[List[float]] = []
        self.b1: List[float] = []
        self.w2: List[List[float]] = []
        self.b2: List[float] = []
        self.w3: List[List[float]] = []   # hidden → output
        self.b3: List[float] = []

        # VAD bypass weights (stimulus → output directly)
        self.w_bypass: List[List[float]] = []

        # Lateral influence matrix (emotion→emotion)
        self.lateral: List[List[float]] = build_influence_matrix()

        # Balance between hidden path and VAD bypass
        # Starts at 0.7 (bypass dominant), shifts towards hidden as
        # the network adapts.
        self.bypass_strength: float = 0.7

        # Learning rate for online Hebbian-style adaptation
        self.learning_rate: float = 0.005

        # Try loading saved weights, otherwise initialise fresh
        if not self._load():
            self._init_fresh()

    # ── Initialisation ───────────────────────────────────────

    def _init_fresh(self) -> None:
        seed = 42
        self.w1 = _init_weights(HIDDEN_1, INPUT_DIM, seed)
        self.b1 = _init_bias(HIDDEN_1)

        self.w2 = _init_weights(HIDDEN_2, HIDDEN_1, seed + 1)
        self.b2 = _init_bias(HIDDEN_2)

        self.w3 = _init_weights(OUTPUT_DIM, HIDDEN_2, seed + 2)
        self.b3 = _vad_seeded_output_bias()

        self.w_bypass = _build_vad_bypass()

    # ── Forward pass ─────────────────────────────────────────

    def forward(
        self,
        stimulus: List[float],
        current_state: Optional[List[float]] = None,
        lateral_strength: float = 0.3,
    ) -> List[float]:
        """
        Run the forward pass.

        Parameters
        ----------
        stimulus : list[float]
            8-dimensional stimulus vector.
        current_state : list[float] | None
            Current emotion activations (N-dim).  Used to compute
            lateral influence so the AI's mood colours its responses.
        lateral_strength : float
            How much the lateral connections influence the output.

        Returns
        -------
        list[float]
            Emotion activations, each in [0, 1].
        """
        assert len(stimulus) == INPUT_DIM, (
            f"Expected {INPUT_DIM}-dim stimulus, got {len(stimulus)}"
        )

        # ── Hidden path ──────────────────────────────────────
        z1 = _vec_add(_mat_vec(self.w1, stimulus), self.b1)
        a1 = [_relu(z) for z in z1]

        z2 = _vec_add(_mat_vec(self.w2, a1), self.b2)
        a2 = [_relu(z) for z in z2]

        hidden_out = _vec_add(_mat_vec(self.w3, a2), self.b3)

        # ── VAD bypass path ──────────────────────────────────
        bypass_out = _mat_vec(self.w_bypass, stimulus)

        # ── Combine and activate ─────────────────────────────
        bs = self.bypass_strength
        combined = [
            _sigmoid(bs * bp + (1.0 - bs) * ho)
            for ho, bp in zip(hidden_out, bypass_out)
        ]

        # ── Lateral influence ────────────────────────────────
        if current_state is not None and len(current_state) == OUTPUT_DIM:
            influence = _mat_vec(self.lateral, current_state)
            combined = [
                _clamp(c + lateral_strength * inf)
                for c, inf in zip(combined, influence)
            ]

        return combined

    # ── Online adaptation (Hebbian-inspired) ─────────────────

    def adapt(
        self,
        stimulus: List[float],
        target_emotion: str,
        strength: float = 1.0,
    ) -> None:
        """
        Nudge weights towards activating *target_emotion* for this
        stimulus pattern.

        Adapts both the bypass weights (fast, direct) and the output
        layer of the hidden path (slower, contextual).
        """
        from .emotions import EMOTION_INDEX

        idx = EMOTION_INDEX.get(target_emotion)
        if idx is None:
            return

        lr = self.learning_rate * strength

        # Adapt bypass weights — direct stimulus → emotion mapping
        for j in range(INPUT_DIM):
            self.w_bypass[idx][j] += lr * stimulus[j] * 0.5

        # Adapt hidden-path output weights
        z1 = _vec_add(_mat_vec(self.w1, stimulus), self.b1)
        a1 = [_relu(z) for z in z1]
        z2 = _vec_add(_mat_vec(self.w2, a1), self.b2)
        a2 = [_relu(z) for z in z2]

        for j in range(len(a2)):
            self.w3[idx][j] += lr * a2[j]
        self.b3[idx] += lr * 0.05

    # ── Persistence ──────────────────────────────────────────

    def save(self) -> None:
        """Persist all weights to JSON."""
        data = {
            "w1": self.w1, "b1": self.b1,
            "w2": self.w2, "b2": self.b2,
            "w3": self.w3, "b3": self.b3,
            "w_bypass": self.w_bypass,
            "bypass_strength": self.bypass_strength,
            "learning_rate": self.learning_rate,
        }
        try:
            self._path.write_text(
                json.dumps(data, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass

    def _load(self) -> bool:
        if not self._path.exists():
            return False
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self.w1 = data["w1"]
            self.b1 = data["b1"]
            self.w2 = data["w2"]
            self.b2 = data["b2"]
            self.w3 = data["w3"]
            self.b3 = data["b3"]
            self.w_bypass = data["w_bypass"]
            self.bypass_strength = data.get("bypass_strength", 0.7)
            self.learning_rate = data.get("learning_rate", 0.005)
            return True
        except (json.JSONDecodeError, KeyError, OSError):
            return False

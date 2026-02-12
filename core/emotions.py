"""
Emotion type definitions grounded in the VAD (Valence-Arousal-Dominance)
psychological model.

Each of the 65 available emotions is characterised by three continuous
dimensions:

* **Valence**   – pleasure / displeasure  (−1 … +1)
* **Arousal**   – activation / deactivation (0 … 1)
* **Dominance** – sense of control / submission (0 … 1)

The module also encodes *emotion-to-emotion influence* — how one active
emotion can amplify or suppress others (e.g. fear amplifies anxiety,
joy suppresses sadness).  These relationships form the lateral
connection matrix used by the neural network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ------------------------------------------------------------------
# Core data structure
# ------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class EmotionProfile:
    """Psychological fingerprint of a single emotion."""

    name: str
    valence: float      # −1 (very negative) … +1 (very positive)
    arousal: float       # 0 (calm) … 1 (intense)
    dominance: float     # 0 (submissive) … 1 (dominant)

    # Semantic category for grouping / UI display
    category: str = "neutral"

    def as_vector(self) -> Tuple[float, float, float]:
        return (self.valence, self.arousal, self.dominance)


# ------------------------------------------------------------------
# All 65 emotions — VAD values drawn from the psychological
# literature (Russell & Mehrabian 1977; Warriner et al. 2013)
# and fine-tuned for an AI assistant persona.
# ------------------------------------------------------------------

EMOTION_PROFILES: Dict[str, EmotionProfile] = {}

_RAW: List[Tuple[str, float, float, float, str]] = [
    # name              valence  arousal  dominance  category
    # ─── Positive / High-energy ──────────────────────────────
    ("happy",            0.80,    0.65,    0.70,    "joy"),
    ("joyful",           0.90,    0.80,    0.75,    "joy"),
    ("amused",           0.70,    0.55,    0.65,    "joy"),
    ("playful",          0.75,    0.70,    0.70,    "joy"),
    ("mischievous",      0.55,    0.65,    0.70,    "joy"),
    ("sassy",            0.40,    0.60,    0.75,    "joy"),
    ("bratty",           0.25,    0.60,    0.70,    "joy"),
    ("teasing",          0.50,    0.55,    0.70,    "joy"),
    ("sarcastic",        0.10,    0.45,    0.70,    "joy"),

    # ─── Positive / Calm ─────────────────────────────────────
    ("content",          0.70,    0.25,    0.65,    "peace"),
    ("calm",             0.60,    0.10,    0.65,    "peace"),
    ("peaceful",         0.70,    0.08,    0.60,    "peace"),
    ("relaxed",          0.65,    0.12,    0.60,    "peace"),
    ("satisfied",        0.70,    0.30,    0.68,    "peace"),
    ("grounded",         0.55,    0.15,    0.72,    "peace"),

    # ─── Warmth / Connection ─────────────────────────────────
    ("affectionate",     0.85,    0.50,    0.55,    "love"),
    ("loving",           0.90,    0.55,    0.55,    "love"),
    ("compassionate",    0.75,    0.40,    0.50,    "love"),
    ("thankful",         0.80,    0.45,    0.45,    "love"),

    # ─── Motivation / Drive ──────────────────────────────────
    ("confident",        0.65,    0.55,    0.85,    "drive"),
    ("determined",       0.50,    0.70,    0.85,    "drive"),
    ("motivated",        0.65,    0.70,    0.80,    "drive"),
    ("inspired",         0.80,    0.75,    0.75,    "drive"),
    ("proud",            0.70,    0.55,    0.80,    "drive"),
    ("hopeful",          0.65,    0.50,    0.55,    "drive"),

    # ─── Curiosity / Engagement ──────────────────────────────
    ("curious",          0.55,    0.60,    0.60,    "engagement"),
    ("interested",       0.55,    0.50,    0.60,    "engagement"),
    ("surprised",        0.30,    0.75,    0.40,    "engagement"),
    ("shocked",          0.00,    0.90,    0.25,    "engagement"),

    # ─── Anxiety / Fear ──────────────────────────────────────
    ("nervous",         -0.40,    0.65,    0.30,    "fear"),
    ("anxious",         -0.55,    0.70,    0.25,    "fear"),
    ("worried",         -0.45,    0.60,    0.30,    "fear"),
    ("afraid",          -0.65,    0.75,    0.20,    "fear"),
    ("scared",          -0.70,    0.80,    0.18,    "fear"),
    ("terrified",       -0.85,    0.95,    0.10,    "fear"),
    ("insecure",        -0.45,    0.50,    0.20,    "fear"),

    # ─── Sadness / Loss ──────────────────────────────────────
    ("sad",             -0.60,    0.30,    0.25,    "sadness"),
    ("lonely",          -0.65,    0.25,    0.20,    "sadness"),
    ("heartbroken",     -0.85,    0.50,    0.15,    "sadness"),
    ("grief",           -0.90,    0.55,    0.12,    "sadness"),
    ("disappointed",    -0.50,    0.35,    0.30,    "sadness"),
    ("hopeless",        -0.80,    0.20,    0.10,    "sadness"),

    # ─── Anger / Frustration ─────────────────────────────────
    ("annoyed",         -0.40,    0.55,    0.60,    "anger"),
    ("frustrated",      -0.55,    0.65,    0.50,    "anger"),
    ("angry",           -0.70,    0.80,    0.65,    "anger"),
    ("enraged",         -0.90,    0.95,    0.70,    "anger"),
    ("bitter",          -0.55,    0.40,    0.50,    "anger"),
    ("resentful",       -0.60,    0.45,    0.45,    "anger"),
    ("jealous",         -0.55,    0.60,    0.35,    "anger"),

    # ─── Shame / Self-conscious ──────────────────────────────
    ("embarrassed",     -0.50,    0.60,    0.20,    "shame"),
    ("ashamed",         -0.65,    0.50,    0.15,    "shame"),
    ("guilty",          -0.60,    0.50,    0.20,    "shame"),
    ("shy",             -0.20,    0.45,    0.20,    "shame"),
    ("awkward",         -0.25,    0.50,    0.25,    "shame"),
    ("flustered",       -0.30,    0.65,    0.25,    "shame"),

    # ─── Low-energy / Withdrawal ─────────────────────────────
    ("tired",           -0.25,    0.10,    0.30,    "fatigue"),
    ("sleepy",          -0.10,    0.05,    0.30,    "fatigue"),
    ("exhausted",       -0.45,    0.08,    0.15,    "fatigue"),
    ("drained",         -0.50,    0.10,    0.15,    "fatigue"),
    ("bored",           -0.30,    0.08,    0.40,    "fatigue"),

    # ─── Detachment / Numbness ───────────────────────────────
    ("apathetic",       -0.20,    0.05,    0.35,    "detachment"),
    ("numb",            -0.35,    0.03,    0.25,    "detachment"),
    ("empty",           -0.55,    0.05,    0.15,    "detachment"),
    ("disinterested",   -0.15,    0.08,    0.45,    "detachment"),

    # ─── Depressive ──────────────────────────────────────────
    ("depressed",       -0.80,    0.10,    0.10,    "depression"),
]


def _build_profiles() -> None:
    for name, v, a, d, cat in _RAW:
        EMOTION_PROFILES[name] = EmotionProfile(
            name=name, valence=v, arousal=a, dominance=d, category=cat,
        )


_build_profiles()

# Ordered list matching the neural-network output vector.
EMOTION_NAMES: List[str] = [ep.name for ep in EMOTION_PROFILES.values()]
NUM_EMOTIONS: int = len(EMOTION_NAMES)
EMOTION_INDEX: Dict[str, int] = {n: i for i, n in enumerate(EMOTION_NAMES)}

# Category colours for UI visualisation (hex).
CATEGORY_COLOURS: Dict[str, str] = {
    "joy":        "#f9c74f",
    "peace":      "#90be6d",
    "love":       "#f4845f",
    "drive":      "#577590",
    "engagement": "#43aa8b",
    "fear":       "#f94144",
    "sadness":    "#277da1",
    "anger":      "#e63946",
    "shame":      "#b5838d",
    "fatigue":    "#6d6875",
    "detachment": "#adb5bd",
    "depression": "#495057",
}


# ------------------------------------------------------------------
# Emotion-to-emotion lateral influence
# ------------------------------------------------------------------
# Each entry: (source, target, weight)
#   weight > 0  →  source *amplifies* target
#   weight < 0  →  source *suppresses* target
#
# Only the most psychologically significant pairs are listed;
# the neural network fills the rest through learned hidden-layer
# representations.
# ------------------------------------------------------------------

EMOTION_INFLUENCES: List[Tuple[str, str, float]] = [
    # Fear cluster — mutually reinforcing
    ("afraid",      "anxious",       0.60),
    ("afraid",      "nervous",       0.50),
    ("anxious",     "worried",       0.55),
    ("terrified",   "afraid",        0.70),
    ("scared",      "nervous",       0.50),

    # Sadness cluster
    ("sad",         "lonely",        0.45),
    ("heartbroken", "grief",         0.65),
    ("grief",       "hopeless",      0.50),
    ("disappointed","sad",           0.40),
    ("lonely",      "depressed",     0.35),

    # Anger cluster
    ("annoyed",     "frustrated",    0.50),
    ("frustrated",  "angry",         0.55),
    ("angry",       "enraged",       0.45),
    ("bitter",      "resentful",     0.55),
    ("jealous",     "angry",         0.35),

    # Joy cluster
    ("happy",       "joyful",        0.40),
    ("joyful",      "playful",       0.35),
    ("amused",      "playful",       0.45),
    ("playful",     "mischievous",   0.40),
    ("mischievous", "teasing",       0.50),

    # Warmth cluster
    ("loving",      "affectionate",  0.60),
    ("compassionate","loving",       0.35),
    ("thankful",    "happy",         0.40),

    # Drive cluster
    ("determined",  "motivated",     0.50),
    ("inspired",    "motivated",     0.55),
    ("confident",   "proud",         0.40),
    ("hopeful",     "motivated",     0.35),

    # Cross-category suppression
    ("happy",       "sad",          -0.60),
    ("joyful",      "depressed",    -0.55),
    ("calm",        "anxious",      -0.50),
    ("peaceful",    "angry",        -0.45),
    ("confident",   "insecure",     -0.60),
    ("hopeful",     "hopeless",     -0.55),
    ("content",     "empty",        -0.50),
    ("motivated",   "apathetic",    -0.50),
    ("angry",       "calm",         -0.40),
    ("sad",         "happy",        -0.45),
    ("anxious",     "relaxed",      -0.45),
    ("depressed",   "joyful",       -0.50),
    ("exhausted",   "motivated",    -0.55),
    ("numb",        "curious",      -0.40),
    ("bored",       "interested",   -0.50),

    # Shame → withdrawal
    ("embarrassed", "shy",           0.45),
    ("ashamed",     "guilty",        0.40),
    ("guilty",      "sad",           0.30),
    ("flustered",   "anxious",       0.35),
]


def build_influence_matrix() -> list[list[float]]:
    """
    Return an N×N matrix where ``mat[target][source]`` is the lateral
    influence that *source* emotion exerts on *target* emotion.

    Indexed this way so that ``mat_vec(mat, state)`` directly yields
    the influence vector — ``influence[t] = sum(mat[t][s] * state[s])``.
    """
    n = NUM_EMOTIONS
    mat = [[0.0] * n for _ in range(n)]
    for src, tgt, w in EMOTION_INFLUENCES:
        si = EMOTION_INDEX.get(src)
        ti = EMOTION_INDEX.get(tgt)
        if si is not None and ti is not None:
            mat[ti][si] = w      # target row, source column
    return mat

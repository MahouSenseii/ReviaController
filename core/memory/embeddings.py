"""
Lightweight text embeddings using random indexing with character
n-grams for sub-word similarity.

Produces fixed-size dense vectors from text with **zero external
dependencies** — only ``math``, ``hashlib``, and ``re`` from the
standard library.

How it works
------------
1. Tokenise text into **word unigrams** (stopwords removed) and
   **character n-grams** (tri-grams and quad-grams within each word).
   Character n-grams let semantically related words (e.g. "program",
   "programming", "programmer") share vector components.
2. For each feature, derive a deterministic pseudo-random unit vector
   from its SHA-256 hash (random indexing).
3. Sum all feature vectors (words weighted higher than n-grams) and
   L2-normalise the result.

This approach gives meaningful similarity for short-to-medium texts
without any vocabulary, training data, or external model.  When a
plugin with ``PluginCapability.EMBEDDING`` is available, the
``RAGEngine`` can transparently swap in real transformer embeddings.
"""

from __future__ import annotations

import hashlib
import math
import re
import struct
from collections import Counter
from typing import List

# Dimension of the embedding vectors.
EMBED_DIM = 384

# ------------------------------------------------------------------
# Tokenisation
# ------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9']+")
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "about", "between", "through", "during", "before",
    "after", "above", "below", "and", "but", "or", "nor", "not",
    "so", "yet", "both", "either", "neither", "each", "every",
    "this", "that", "these", "those", "it", "its", "i", "me", "my",
    "we", "our", "you", "your", "he", "him", "his", "she", "her",
    "they", "them", "their", "what", "which", "who", "whom",
    "said", "just",
})


def _char_ngrams(word: str, ns: tuple[int, ...] = (3, 4)) -> List[str]:
    """Extract character n-grams from a word, including boundary markers."""
    padded = f"#{word}#"
    grams: List[str] = []
    for n in ns:
        for i in range(len(padded) - n + 1):
            grams.append(padded[i:i + n])
    return grams


def tokenise(text: str) -> tuple[List[str], List[str]]:
    """
    Return ``(words, char_ngrams)`` from *text*.

    Words are unigrams with stopwords removed.
    Char n-grams are extracted from all words (including stopwords)
    to capture sub-word patterns.
    """
    all_words = _WORD_RE.findall(text.lower())

    words = [w for w in all_words if w not in _STOPWORDS and len(w) > 1]

    ngrams: List[str] = []
    for w in all_words:
        if len(w) > 2:
            ngrams.extend(_char_ngrams(w))

    return words, ngrams


# ------------------------------------------------------------------
# Hash-based pseudo-random vector generation
# ------------------------------------------------------------------

# Cache to avoid recomputing vectors for repeated features.
_vector_cache: dict[str, List[float]] = {}
_CACHE_MAX = 10000


def _feature_vector(feature: str) -> List[float]:
    """
    Deterministic unit vector for *feature* using SHA-256 seeding.

    Uses multiple hash rounds to fill all EMBED_DIM dimensions with
    good pseudo-randomness.
    """
    cached = _vector_cache.get(feature)
    if cached is not None:
        return cached

    raw: List[float] = []
    # Generate enough bytes: 4 bytes per float, multiple rounds
    rounds_needed = math.ceil(EMBED_DIM * 4 / 32)  # SHA-256 = 32 bytes
    all_bytes = b""
    seed = feature.encode("utf-8")
    for r in range(rounds_needed):
        seed = hashlib.sha256(seed).digest()
        all_bytes += seed

    # Convert consecutive 4-byte chunks into floats in [-1, +1]
    for i in range(EMBED_DIM):
        # Unpack as unsigned 32-bit int, map to [-1, 1]
        val = struct.unpack_from(">I", all_bytes, i * 4)[0]
        raw.append((val / 2147483647.5) - 1.0)

    # Normalise to unit length
    norm = math.sqrt(sum(x * x for x in raw))
    if norm < 1e-10:
        result = [0.0] * EMBED_DIM
    else:
        result = [x / norm for x in raw]

    # Cache (with eviction)
    if len(_vector_cache) >= _CACHE_MAX:
        _vector_cache.clear()
    _vector_cache[feature] = result

    return result


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def embed_text(text: str) -> List[float]:
    """
    Compute a fixed-size embedding vector for *text*.

    Words contribute with weight 3.0, character n-grams with 1.0.
    This ensures whole-word matches dominate while n-grams provide
    fuzzy sub-word similarity.

    Returns a list of ``EMBED_DIM`` floats, L2-normalised.
    """
    words, ngrams = tokenise(text)
    if not words and not ngrams:
        return [0.0] * EMBED_DIM

    vec = [0.0] * EMBED_DIM

    # Word features (higher weight — exact word matches matter most)
    word_counts = Counter(words)
    for word, count in word_counts.items():
        weight = 3.0 * (1.0 + math.log(count))
        fv = _feature_vector(f"w:{word}")
        for d in range(EMBED_DIM):
            vec[d] += weight * fv[d]

    # Character n-gram features (lower weight — fuzzy matching)
    ngram_counts = Counter(ngrams)
    for ngram, count in ngram_counts.items():
        weight = 1.0 * (1.0 + math.log(count))
        fv = _feature_vector(f"c:{ngram}")
        for d in range(EMBED_DIM):
            vec[d] += weight * fv[d]

    # L2-normalise
    norm = math.sqrt(sum(x * x for x in vec))
    if norm < 1e-10:
        return [0.0] * EMBED_DIM
    return [x / norm for x in vec]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Cosine similarity between two vectors.

    Both should already be L2-normalised (from ``embed_text``),
    so this is just a dot product.
    """
    return sum(ai * bi for ai, bi in zip(a, b))


def embed_batch(texts: List[str]) -> List[List[float]]:
    """Embed multiple texts.  Convenience wrapper."""
    return [embed_text(t) for t in texts]

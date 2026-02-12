"""
Text scoring engine for the safety filter.

Scans input text against every enabled category's keyword stems,
phrases, and regex patterns.  Returns per-category severity scores
(0–5) along with the matched evidence.

The scorer is stateless — it takes text in and returns scores out.
All configuration (which categories are enabled, what thresholds
apply) is handled by the ``FilterEngine``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .categories import CATEGORIES, CATEGORY_ORDER, SafetyCategory


# ------------------------------------------------------------------
# Result structures
# ------------------------------------------------------------------

@dataclass
class CategoryScore:
    """Score result for a single category."""
    slug: str
    name: str
    score: int = 0                          # 0–5
    matched_keywords: List[str] = field(default_factory=list)
    matched_phrases: List[str] = field(default_factory=list)
    matched_patterns: List[str] = field(default_factory=list)

    @property
    def triggered(self) -> bool:
        return self.score > 0


@dataclass
class ScoringResult:
    """Complete scoring result across all categories."""
    text: str
    scores: Dict[str, CategoryScore] = field(default_factory=dict)

    @property
    def max_score(self) -> int:
        if not self.scores:
            return 0
        return max(cs.score for cs in self.scores.values())

    @property
    def triggered_categories(self) -> List[CategoryScore]:
        return [cs for cs in self.scores.values() if cs.triggered]

    @property
    def is_clean(self) -> bool:
        return self.max_score == 0

    def category_score(self, slug: str) -> int:
        cs = self.scores.get(slug)
        return cs.score if cs else 0


# ------------------------------------------------------------------
# Word normalisation for keyword matching
# ------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9*]+")
_LEET_MAP = str.maketrans({
    "0": "o", "1": "i", "3": "e", "4": "a", "5": "s",
    "7": "t", "@": "a", "$": "s", "!": "i",
})


def _normalise(text: str) -> str:
    """
    Lowercase + basic leet-speak reversal + collapse whitespace.

    This catches things like "k1ll" → "kill", "h@te" → "hate".
    """
    return text.lower().translate(_LEET_MAP)


def _extract_words(text: str) -> List[str]:
    """Split normalised text into word tokens."""
    return _WORD_RE.findall(text)


# ------------------------------------------------------------------
# Scorer
# ------------------------------------------------------------------

class SafetyScorer:
    """
    Stateless text scorer.

    Call ``score(text)`` to get per-category severity scores.
    Call ``score_categories(text, slugs)`` to check only specific
    categories (slightly faster if you know what you need).
    """

    def score(
        self,
        text: str,
        enabled_slugs: Optional[List[str]] = None,
    ) -> ScoringResult:
        """
        Score *text* against all (or specified) categories.

        Parameters
        ----------
        text : str
            The raw message text to evaluate.
        enabled_slugs : list[str] | None
            If given, only these categories are checked.

        Returns
        -------
        ScoringResult
        """
        norm = _normalise(text)
        words = _extract_words(norm)
        slugs = enabled_slugs or CATEGORY_ORDER

        result = ScoringResult(text=text)

        for slug in slugs:
            cat = CATEGORIES.get(slug)
            if cat is None:
                continue
            cs = self._score_category(cat, text, norm, words)
            result.scores[slug] = cs

        return result

    # ── Internal ─────────────────────────────────────────────

    def _score_category(
        self,
        cat: SafetyCategory,
        raw_text: str,
        norm_text: str,
        words: List[str],
    ) -> CategoryScore:
        cs = CategoryScore(slug=cat.slug, name=cat.name)

        # 1. Keyword stem matching
        for stem, severity in cat.keywords:
            stem_norm = _normalise(stem)
            # Check if any word starts with the stem
            for w in words:
                if w.startswith(stem_norm) or stem_norm in w:
                    cs.score = max(cs.score, severity)
                    cs.matched_keywords.append(stem)
                    break
            # Also check as substring in normalised text (catches multi-word stems)
            if stem_norm in norm_text and stem not in cs.matched_keywords:
                cs.score = max(cs.score, severity)
                cs.matched_keywords.append(stem)

        # 2. Phrase matching
        for phrase, severity in cat.phrases:
            phrase_norm = _normalise(phrase)
            if phrase_norm in norm_text:
                cs.score = max(cs.score, severity)
                cs.matched_phrases.append(phrase)

        # 3. Regex pattern matching
        for pattern, severity in cat.patterns:
            match = pattern.search(raw_text)
            if match is None:
                match = pattern.search(norm_text)
            if match:
                cs.score = max(cs.score, severity)
                cs.matched_patterns.append(match.group(0)[:80])

        return cs

    def find_matches(
        self,
        text: str,
        slug: str,
    ) -> List[Tuple[int, int, str, int]]:
        """
        Return all match spans for a specific category.

        Returns list of ``(start, end, matched_text, severity)`` tuples.
        Used by the rewriter to know exactly what to replace.
        """
        cat = CATEGORIES.get(slug)
        if cat is None:
            return []

        norm = _normalise(text)
        matches: List[Tuple[int, int, str, int]] = []

        # Keyword matches — find spans in the original text
        for stem, severity in cat.keywords:
            stem_norm = _normalise(stem)
            for m in re.finditer(re.escape(stem_norm), norm):
                matches.append((m.start(), m.end(), m.group(0), severity))

        # Phrase matches
        for phrase, severity in cat.phrases:
            phrase_norm = _normalise(phrase)
            for m in re.finditer(re.escape(phrase_norm), norm):
                matches.append((m.start(), m.end(), m.group(0), severity))

        # Pattern matches
        for pattern, severity in cat.patterns:
            for m in pattern.finditer(text):
                matches.append((m.start(), m.end(), m.group(0), severity))
            for m in pattern.finditer(norm):
                matches.append((m.start(), m.end(), m.group(0), severity))

        # Deduplicate overlapping spans
        matches.sort(key=lambda x: (x[0], -x[1]))
        deduped: List[Tuple[int, int, str, int]] = []
        last_end = -1
        for start, end, txt, sev in matches:
            if start >= last_end:
                deduped.append((start, end, txt, sev))
                last_end = end

        return deduped

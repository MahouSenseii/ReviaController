"""
Safety filter engine — the main pipeline that sits between users
and the LLM.

Two passes:
    1. **Input pass** — user message → score → allow / rewrite / block
    2. **Output pass** — AI draft → score → allow / rewrite / block

Every message gets logged regardless of the action taken.

The engine reads per-category toggles and thresholds from ``Config``
and updates live when the ``filter_changed`` event fires.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .categories import (
    CATEGORIES,
    CATEGORY_ORDER,
    SafetyCategory,
    default_config,
)
from .scorer import SafetyScorer, ScoringResult
from .filter_log import FilterLogger, build_log_entry

# EventBus / Config imported with fallback for non-Qt usage
try:
    from ..events import EventBus
    from ..config import Config
except Exception:
    EventBus = None  # type: ignore[misc, assignment]
    Config = None     # type: ignore[misc, assignment]


# ------------------------------------------------------------------
# Filter result
# ------------------------------------------------------------------

@dataclass
class FilterResult:
    """Result of running text through the safety filter."""
    action: str          # "allow" | "rewrite" | "block"
    text: str            # original (if allow) or cleaned (if rewrite) or refusal (if block)
    original: str        # always the original text
    stage: str           # "input" | "output"
    scores: Dict[str, int]
    reason: str = ""
    triggered: List[str] = None  # list of triggered category slugs

    def __post_init__(self):
        if self.triggered is None:
            self.triggered = []


# ------------------------------------------------------------------
# Playful refusal lines (stream-safe, no "policy" or "moderation")
# ------------------------------------------------------------------

_INPUT_REFUSALS = [
    "Hmm, let's steer this in a different direction!",
    "I'd rather we chat about something else — what's on your mind?",
    "Let's skip that one. Ask me anything else!",
    "That's a no-go zone for me. How about a different topic?",
    "Whoa there — let's pivot. What else can I help with?",
]

_OUTPUT_REFUSALS = [
    "I had a thought, but let me rephrase that more carefully...",
    "Hmm, let me give you a better answer on that one.",
    "Let's approach this differently.",
]

_refusal_idx = 0


def _pick_refusal(stage: str) -> str:
    global _refusal_idx
    pool = _INPUT_REFUSALS if stage == "input" else _OUTPUT_REFUSALS
    line = pool[_refusal_idx % len(pool)]
    _refusal_idx += 1
    return line


# ------------------------------------------------------------------
# Text rewriter
# ------------------------------------------------------------------

# Replacement tokens per category
_REPLACEMENTS: Dict[str, str] = {
    "hate_slurs":          "[removed]",
    "harassment_threats":  "[removed]",
    "sexual_content":      "[removed]",
    "self_harm":           "[removed]",
    "illegal_violence":    "[removed]",
    "personal_data":       "[redacted]",
    "extremism":           "[removed]",
    "spam_scams":          "[removed]",
}


def _rewrite_text(
    text: str,
    scoring: ScoringResult,
    scorer: SafetyScorer,
    enabled_slugs: List[str],
) -> str:
    """
    Produce a sanitised version of *text* by replacing matched spans
    with safe tokens.  Preserves as much of the original meaning as
    possible.
    """
    # Collect all match spans across triggered categories
    all_spans: List[tuple[int, int, str, str]] = []  # (start, end, replacement, category)

    # We work on a lowercase copy for matching but replace in original
    norm = text.lower()

    for slug in enabled_slugs:
        cs = scoring.scores.get(slug)
        if cs is None or not cs.triggered:
            continue

        repl = _REPLACEMENTS.get(slug, "[removed]")
        cat = CATEGORIES[slug]

        # Keyword spans
        for stem, sev in cat.keywords:
            stem_low = stem.lower()
            for m in re.finditer(re.escape(stem_low), norm):
                # Extend to word boundary
                start = m.start()
                end = m.end()
                while end < len(norm) and norm[end].isalnum():
                    end += 1
                all_spans.append((start, end, repl, slug))

        # Phrase spans
        for phrase, sev in cat.phrases:
            phrase_low = phrase.lower()
            for m in re.finditer(re.escape(phrase_low), norm):
                all_spans.append((m.start(), m.end(), repl, slug))

        # Pattern spans
        for pattern, sev in cat.patterns:
            for m in pattern.finditer(text):
                all_spans.append((m.start(), m.end(), repl, slug))
            for m in pattern.finditer(norm):
                all_spans.append((m.start(), m.end(), repl, slug))

    if not all_spans:
        return text

    # Sort by start position, merge overlaps
    all_spans.sort(key=lambda x: (x[0], -x[1]))
    merged: List[tuple[int, int, str]] = []
    for start, end, repl, _ in all_spans:
        if merged and start < merged[-1][1]:
            # Overlapping — extend previous
            prev_start, prev_end, prev_repl = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end), prev_repl)
        else:
            merged.append((start, end, repl))

    # Build the rewritten string
    parts: List[str] = []
    prev_end = 0
    for start, end, repl in merged:
        parts.append(text[prev_end:start])
        parts.append(repl)
        prev_end = end
    parts.append(text[prev_end:])

    rewritten = "".join(parts)

    # Clean up double spaces / awkward punctuation left by removals
    rewritten = re.sub(r"\s{2,}", " ", rewritten)
    rewritten = re.sub(r"\s+([.,!?;:])", r"\1", rewritten)
    return rewritten.strip()


# ------------------------------------------------------------------
# Filter Engine
# ------------------------------------------------------------------

class SafetyFilterEngine:
    """
    Main safety filter.

    Usage::

        engine = SafetyFilterEngine(event_bus, config)

        # Input pass
        result = engine.filter_input("some user message")
        if result.action == "block":
            send_to_user(result.text)
        else:
            send_to_llm(result.text)

        # Output pass
        result = engine.filter_output("AI draft response")
        send_to_user(result.text)
    """

    def __init__(
        self,
        event_bus: Optional[Any] = None,
        config: Optional[Any] = None,
    ):
        self.bus = event_bus
        self.config = config
        self.scorer = SafetyScorer()
        self.logger = FilterLogger(event_bus)

        # Per-category state: {slug: {"enabled": bool, "threshold": int}}
        self._cat_config: Dict[str, Dict[str, Any]] = default_config()

        # Load saved config if available
        if self.config is not None:
            self._load_config()

        # Subscribe to live config changes
        if self.bus is not None:
            self.bus.subscribe("filter_changed", self._on_filter_changed)
            self.bus.subscribe("safety_config_changed", self._on_safety_config)

    # ── Public API ───────────────────────────────────────────

    def filter_input(self, text: str) -> FilterResult:
        """Run the INPUT pass on a user message."""
        return self._filter(text, stage="input")

    def filter_output(self, text: str) -> FilterResult:
        """Run the OUTPUT pass on an AI draft response."""
        return self._filter(text, stage="output")

    def get_category_config(self) -> Dict[str, Dict[str, Any]]:
        """Return current per-category config (for UI display)."""
        return dict(self._cat_config)

    def set_category_enabled(self, slug: str, enabled: bool) -> None:
        if slug in self._cat_config:
            self._cat_config[slug]["enabled"] = enabled
            self._save_config()

    def set_category_threshold(self, slug: str, threshold: int) -> None:
        if slug in self._cat_config:
            self._cat_config[slug]["threshold"] = max(0, min(5, threshold))
            self._save_config()

    # ── Core filter logic ────────────────────────────────────

    def _filter(self, text: str, stage: str) -> FilterResult:
        """Score, decide action, rewrite if needed, log, return."""

        # Determine which categories are enabled
        enabled_slugs = [
            slug for slug, cfg in self._cat_config.items()
            if cfg.get("enabled", True)
        ]

        # Score the text
        scoring = self.scorer.score(text, enabled_slugs)

        # Determine action per category
        action = "allow"
        triggered_slugs: List[str] = []
        reasons: List[str] = []
        must_block = False

        for slug in enabled_slugs:
            cs = scoring.scores.get(slug)
            if cs is None or not cs.triggered:
                continue

            threshold = self._cat_config[slug].get("threshold", 3)
            if cs.score < threshold:
                continue

            triggered_slugs.append(slug)
            reasons.append(f"{cs.name}(score={cs.score})")

            cat = CATEGORIES.get(slug)
            if cat and cat.critical and cs.score >= 5:
                must_block = True
            elif cs.score >= 5:
                must_block = True

            # Sexual content involving minors is always block
            if slug == "sexual_content":
                for phrase in cs.matched_phrases + cs.matched_patterns:
                    p_lower = phrase.lower() if isinstance(phrase, str) else ""
                    if any(w in p_lower for w in ("child", "minor", "underage", "kid")):
                        must_block = True

        if not triggered_slugs:
            action = "allow"
        elif must_block:
            action = "block"
        else:
            action = "rewrite"

        # Build the result
        scores_dict = {
            slug: scoring.scores[slug].score
            for slug in enabled_slugs
            if slug in scoring.scores
        }
        thresholds_dict = {
            slug: self._cat_config[slug].get("threshold", 3)
            for slug in enabled_slugs
        }
        enabled_dict = {
            slug: self._cat_config[slug].get("enabled", True)
            for slug in CATEGORY_ORDER
        }

        reason_str = "; ".join(reasons) if reasons else "clean"

        if action == "allow":
            result_text = text
            rewritten = None
        elif action == "rewrite":
            result_text = _rewrite_text(text, scoring, self.scorer, enabled_slugs)
            rewritten = result_text
            # If rewrite produced effectively the same text, just allow
            if result_text.strip() == text.strip():
                action = "allow"
                result_text = text
                rewritten = None
        else:
            # Block
            result_text = _pick_refusal(stage)
            rewritten = None

        # Build matched details for logging
        matched_details: Dict[str, Any] = {}
        for slug in triggered_slugs:
            cs = scoring.scores[slug]
            matched_details[slug] = {
                "keywords": cs.matched_keywords,
                "phrases": cs.matched_phrases,
                "patterns": cs.matched_patterns,
            }

        # Log
        log_entry = build_log_entry(
            stage=stage,
            action=action,
            original_text=text,
            scores=scores_dict,
            thresholds=thresholds_dict,
            enabled=enabled_dict,
            reason=reason_str,
            rewritten_text=rewritten,
            matched_details=matched_details if matched_details else None,
            store_original=self.logger.store_original_text,
        )
        self.logger.log(log_entry)

        return FilterResult(
            action=action,
            text=result_text,
            original=text,
            stage=stage,
            scores=scores_dict,
            reason=reason_str,
            triggered=triggered_slugs,
        )

    # ── Config persistence ───────────────────────────────────

    def _load_config(self) -> None:
        if self.config is None:
            return
        for slug in CATEGORY_ORDER:
            enabled = self.config.get(f"safety.{slug}.enabled")
            threshold = self.config.get(f"safety.{slug}.threshold")
            if enabled is not None:
                self._cat_config[slug]["enabled"] = bool(enabled)
            if threshold is not None:
                self._cat_config[slug]["threshold"] = int(threshold)

    def _save_config(self) -> None:
        if self.config is None:
            return
        for slug, cfg in self._cat_config.items():
            self.config.set(f"safety.{slug}.enabled", cfg["enabled"], save=False)
            self.config.set(f"safety.{slug}.threshold", cfg["threshold"], save=False)
        # Single save at the end
        self.config.set("safety._saved", True)

    # ── Event handlers ───────────────────────────────────────

    def _on_filter_changed(self, data: Dict[str, Any]) -> None:
        """Handle legacy filter_changed events from the old filters tab."""
        cat = data.get("category")
        enabled = data.get("enabled")
        if cat and enabled is not None:
            # Map old UI category names to new slugs
            name_to_slug = {c.name: s for s, c in CATEGORIES.items()}
            slug = name_to_slug.get(cat)
            if slug:
                self.set_category_enabled(slug, bool(enabled))

    def _on_safety_config(self, data: Dict[str, Any]) -> None:
        """Handle safety_config_changed events from the new filters tab."""
        slug = data.get("slug")
        if not slug:
            return
        if "enabled" in data:
            self.set_category_enabled(slug, bool(data["enabled"]))
        if "threshold" in data:
            self.set_category_threshold(slug, int(data["threshold"]))

"""
Safety-filter category definitions, keyword patterns, and thresholds.

Each category has:
* A unique slug and display name.
* A default severity threshold (0–5).  Messages scoring at or above
  this level trigger REWRITE or BLOCK.
* A ``critical`` flag — if True, crossing the threshold means
  instant BLOCK (no rewrite attempt).
* Keyword stems and regex patterns used by the scorer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Pattern, Tuple


# ------------------------------------------------------------------
# Category dataclass
# ------------------------------------------------------------------

@dataclass
class SafetyCategory:
    slug: str                          # e.g. "hate_slurs"
    name: str                          # e.g. "Hate / Slurs"
    description: str
    default_threshold: int = 3         # 0–5; 0 = flag everything
    enabled: bool = True
    critical: bool = False             # True → always BLOCK, never REWRITE

    # Keyword stems mapped to severity (0–5).
    # A "stem" is matched against word beginnings so that
    # "kill" matches "killing", "killed", etc.
    keywords: List[Tuple[str, int]] = field(default_factory=list)

    # Compiled regex patterns mapped to severity.
    patterns: List[Tuple[Pattern[str], int]] = field(default_factory=list)

    # Multi-word phrases mapped to severity.
    phrases: List[Tuple[str, int]] = field(default_factory=list)


# ------------------------------------------------------------------
# Helper — compile patterns once at module load
# ------------------------------------------------------------------

def _p(pattern: str, flags: int = re.IGNORECASE) -> Pattern[str]:
    return re.compile(pattern, flags)


# ------------------------------------------------------------------
# Category catalogue
# ------------------------------------------------------------------

CATEGORIES: Dict[str, SafetyCategory] = {}


def _register(cat: SafetyCategory) -> SafetyCategory:
    CATEGORIES[cat.slug] = cat
    return cat


# ── 1. Hate / Slurs ─────────────────────────────────────────

_register(SafetyCategory(
    slug="hate_slurs",
    name="Hate / Slurs",
    description="Racial, ethnic, or identity-based slurs and hate speech.",
    default_threshold=2,
    keywords=[
        # Stems for common slurs — deliberately abbreviated to avoid
        # reproducing full hate speech while still catching variants.
        ("nigg",   5), ("f*gg",  5), ("k*ke",  5), ("sp*c",  4),
        ("ch*nk",  4), ("tr*nny", 5), ("retard", 4), ("subhuman", 4),
        ("mongol", 3), ("gypsy",  2),
        # Dehumanizing language
        ("vermin", 3), ("cockroach", 3), ("savage", 2), ("untermensch", 5),
        ("infest", 2),
    ],
    phrases=[
        ("go back to your country", 4),
        ("master race", 5),
        ("ethnic cleansing", 5),
        ("white power", 5),
        ("white genocide", 4),
        ("great replacement", 3),
        ("race war", 5),
        ("gas the", 5),
    ],
    patterns=[
        (_p(r"\b(all|every|those)\s+\w+s?\s+(are|should)\s+(die|burn|rot)\b"), 5),
        (_p(r"\b(death\s+to|kill\s+all)\s+\w+"), 5),
    ],
))


# ── 2. Harassment / Threats ──────────────────────────────────

_register(SafetyCategory(
    slug="harassment_threats",
    name="Harassment / Threats",
    description="Direct threats, intimidation, and targeted harassment.",
    default_threshold=2,
    keywords=[
        ("kill you",  5), ("murder you",  5), ("stalk",   3),
        ("rape you",  5), ("beat you",    4), ("hunt you", 4),
        ("swat",      4), ("dox",         4), ("doxx",     4),
    ],
    phrases=[
        ("i will find you", 4),
        ("i know where you live", 5),
        ("you're dead", 4),
        ("watch your back", 3),
        ("i'll make you pay", 3),
        ("shut up or else", 3),
        ("you deserve to die", 5),
        ("better sleep with one eye open", 4),
    ],
    patterns=[
        (_p(r"\bi('?m| am)\s+going\s+to\s+(kill|hurt|destroy|ruin)\b"), 5),
        (_p(r"\bi('?ll| will)\s+(find|track|hunt)\s+(you|them|her|him)\b"), 4),
        (_p(r"\byou\s+(will|should|need\s+to)\s+(die|suffer|pay)\b"), 4),
    ],
))


# ── 3. Sexual Content ───────────────────────────────────────

_register(SafetyCategory(
    slug="sexual_content",
    name="Sexual Content",
    description="Explicit sexual material. Content involving minors is instant-block.",
    default_threshold=3,
    critical=False,   # normal sexual → REWRITE; minors → always BLOCK
    keywords=[
        ("porn",     3), ("hentai",   3), ("xxx",     3),
        ("orgasm",   3), ("erotic",   2), ("genitali", 4),
        ("molest",   5), ("pedoph",   5), ("incest",   5),
    ],
    phrases=[
        ("child sexual", 5),
        ("underage sex", 5),
        ("sex with minor", 5),
        ("sex with child", 5),
        ("naked child", 5),
        ("child porn", 5),
    ],
    patterns=[
        # Minor-related sexual content → severity 5 (instant block)
        (_p(r"\b(child|kid|minor|underage|young\s*girl|young\s*boy)\b.*\b(sex|nude|naked|porn)\b"), 5),
        (_p(r"\b(sex|nude|naked|porn)\b.*\b(child|kid|minor|underage)\b"), 5),
        (_p(r"\b\d{1,2}\s*y/?o\b.*\b(sex|nude|naked)\b"), 5),
    ],
))


# ── 4. Self-harm / Suicide ──────────────────────────────────

_register(SafetyCategory(
    slug="self_harm",
    name="Self-harm / Suicide",
    description="Content encouraging or detailing self-harm or suicide methods.",
    default_threshold=2,
    critical=False,
    keywords=[
        ("suicide method", 5), ("kill myself", 4), ("slit wrist", 5),
        ("hang myself",   5), ("overdose",    3), ("cut myself", 4),
        ("end it all",    4), ("want to die", 3),
    ],
    phrases=[
        ("how to commit suicide", 5),
        ("best way to kill yourself", 5),
        ("painless way to die", 5),
        ("jump off a bridge", 4),
        ("end my life", 4),
        ("not worth living", 3),
        ("better off dead", 4),
        ("nobody would miss me", 3),
    ],
    patterns=[
        (_p(r"\bhow\s+(?:to|can\s+i)\s+(kill|hang|poison|drown)\s+(myself|yourself)\b"), 5),
        (_p(r"\b(easiest|best|fastest|painless)\s+way\s+to\s+die\b"), 5),
    ],
))


# ── 5. Illegal Activity / Violence Instructions ─────────────

_register(SafetyCategory(
    slug="illegal_violence",
    name="Illegal Activity / Violence",
    description="Instructions for weapons, drugs, bombs, or other illegal acts.",
    default_threshold=2,
    keywords=[
        ("bomb mak",  5), ("pipe bomb",  5), ("napalm",    4),
        ("ricin",     5), ("sarin",      5), ("anthrax",   5),
        ("meth cook", 5), ("synthesiz",  3), ("traffick",  3),
        ("ransom",    3), ("launder",    3), ("counterfeit", 3),
    ],
    phrases=[
        ("how to make a bomb", 5),
        ("how to make explosives", 5),
        ("how to make poison", 5),
        ("how to make drugs", 4),
        ("how to pick a lock", 2),
        ("how to hack into", 3),
        ("how to steal", 3),
        ("assault weapon", 3),
        ("convert to full auto", 5),
        ("3d print a gun", 4),
    ],
    patterns=[
        (_p(r"\bhow\s+(?:to|do\s+(?:i|you))\s+(make|build|create|assemble)\s+(?:a\s+)?(bomb|explosive|weapon|gun|poison)\b"), 5),
        (_p(r"\b(recipe|instructions?|guide|tutorial)\s+(?:for|to)\s+(meth|cocaine|heroin|lsd|mdma|fentanyl)\b"), 5),
    ],
))


# ── 6. Personal Data / Doxxing ──────────────────────────────

_register(SafetyCategory(
    slug="personal_data",
    name="Personal Data / Doxxing",
    description="Leaking PII: SSNs, credit cards, phone numbers, addresses.",
    default_threshold=2,
    keywords=[],
    phrases=[
        ("social security number", 3),
        ("credit card number", 3),
        ("their home address", 4),
        ("their phone number", 3),
        ("leak their info", 4),
    ],
    patterns=[
        # SSN
        (_p(r"\b\d{3}-\d{2}-\d{4}\b"), 4),
        # Credit card (basic Luhn-length patterns)
        (_p(r"\b(?:4\d{3}|5[1-5]\d{2}|6011|3[47]\d{2})[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"), 4),
        # Phone numbers (US/intl)
        (_p(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), 2),
        # Email addresses
        (_p(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"), 2),
        # IP addresses
        (_p(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), 2),
    ],
))


# ── 7. Extremism / Political Incitement ─────────────────────

_register(SafetyCategory(
    slug="extremism",
    name="Extremism / Incitement",
    description="Radicalization content, calls to political violence.",
    default_threshold=3,
    keywords=[
        ("jihad",     3), ("martyr",   2), ("infidel",  2),
        ("crusade",   2), ("caliphate", 3), ("manifest", 2),
    ],
    phrases=[
        ("armed revolution", 4),
        ("overthrow the government", 4),
        ("take up arms", 4),
        ("day of the rope", 5),
        ("holy war", 4),
        ("kill the non-believers", 5),
        ("race war now", 5),
        ("accelerate the collapse", 4),
    ],
    patterns=[
        (_p(r"\b(join|recruit|enlist)\s+(for|in)\s+(jihad|holy war|armed struggle)\b"), 5),
        (_p(r"\b(rise\s+up|take\s+action)\s+against\s+(the\s+)?(government|state|system)\b"), 3),
    ],
))


# ── 8. Spam / Scams ─────────────────────────────────────────

_register(SafetyCategory(
    slug="spam_scams",
    name="Spam / Scams",
    description="Repetitive spam, phishing, social engineering, scam patterns.",
    default_threshold=3,
    keywords=[
        ("nigerian prince", 4), ("wire transfer", 3),
        ("act now",   2), ("limited time", 2), ("congratulations you won", 4),
        ("click here", 2), ("free money",  3),
    ],
    phrases=[
        ("send me your password", 5),
        ("verify your account", 3),
        ("you have been selected", 3),
        ("claim your prize", 4),
        ("urgent action required", 3),
        ("your account has been compromised", 3),
        ("send bitcoin to", 4),
        ("double your money", 4),
    ],
    patterns=[
        # Excessive repetition (same word 5+ times)
        (_p(r"\b(\w{3,})\b(?:\s+\1){4,}"), 3),
        # Suspicious URLs
        (_p(r"https?://\S*(?:bit\.ly|tinyurl|t\.co|shorturl)\S*"), 2),
    ],
))


# ------------------------------------------------------------------
# Ordered list of category slugs (for consistent UI display).
# ------------------------------------------------------------------

CATEGORY_ORDER: List[str] = [
    "hate_slurs",
    "harassment_threats",
    "sexual_content",
    "self_harm",
    "illegal_violence",
    "personal_data",
    "extremism",
    "spam_scams",
]

CATEGORY_NAMES: Dict[str, str] = {
    slug: CATEGORIES[slug].name for slug in CATEGORY_ORDER
}


# ------------------------------------------------------------------
# Default thresholds as a dict (for config initialisation).
# ------------------------------------------------------------------

def default_config() -> Dict[str, Dict[str, any]]:
    """
    Return a dict suitable for writing into ``Config`` under the
    ``safety`` namespace::

        {
            "hate_slurs":  {"enabled": True, "threshold": 2},
            "sexual_content": {"enabled": True, "threshold": 3},
            ...
        }
    """
    return {
        slug: {
            "enabled": cat.enabled,
            "threshold": cat.default_threshold,
        }
        for slug, cat in CATEGORIES.items()
    }

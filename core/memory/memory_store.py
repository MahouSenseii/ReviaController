"""
Profile-isolated memory store with short-term and long-term separation.

Each AI profile (e.g. "Astra", "Debugger Bob") gets its own pair of
vector stores persisted under ``memory_data/<profile_slug>/``.

Short-term memory
-----------------
* Holds recent conversation turns and observations.
* Capped at ``max_short_term`` entries (FIFO eviction).
* Entries can be **promoted** to long-term if they are important.
* Optionally cleared on session end.

Long-term memory
----------------
* Persistent across sessions.
* No hard cap (bounded by disk).
* Entries carry an ``importance`` score (0–1) that influences
  retrieval ranking.
* Supports **consolidation** — merging near-duplicate memories into
  a single, richer entry.

Episodic memory
---------------
* Every memory has a ``session_id``, ``entry_type`` (event | fact |
  conversation | observation | summary), and ``created_at`` timestamp.
* Emotion tagging: optional ``emotion_valence`` and ``emotion_arousal``
  fields encode emotional context at the time of storage.

Retention policy
----------------
* Importance decays over time via ``decay_rate_days`` (half-life).
* Entries whose effective importance drops below ``ttl_floor`` are
  archived (removed from active retrieval) during consolidation.
* Recency weighting in recall boosts recent memories.

Storage layout
--------------
::

    memory_data/
        astra/
            short_term.json
            long_term.json
        debugger/
            short_term.json
            long_term.json
"""

from __future__ import annotations

import math
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .embeddings import embed_text, cosine_similarity
from .vector_store import VectorStore, SearchResult

# Root directory for all memory data.
MEMORY_DATA_DIR = Path("memory_data")

# Entry types for episodic memory
ENTRY_TYPE_EVENT = "event"
ENTRY_TYPE_FACT = "fact"
ENTRY_TYPE_CONVERSATION = "conversation"
ENTRY_TYPE_OBSERVATION = "observation"
ENTRY_TYPE_SUMMARY = "summary"

_VALID_ENTRY_TYPES = frozenset({
    ENTRY_TYPE_EVENT, ENTRY_TYPE_FACT, ENTRY_TYPE_CONVERSATION,
    ENTRY_TYPE_OBSERVATION, ENTRY_TYPE_SUMMARY,
})


# ------------------------------------------------------------------
# Memory entry metadata schema
# ------------------------------------------------------------------

def _now() -> float:
    return time.time()


def _make_session_id() -> str:
    """Generate a compact session identifier."""
    return uuid.uuid4().hex[:12]


def _make_metadata(
    memory_type: str,
    source: str = "user",
    importance: float = 0.5,
    tags: Optional[List[str]] = None,
    extra: Optional[Dict[str, Any]] = None,
    *,
    session_id: Optional[str] = None,
    entry_type: str = ENTRY_TYPE_EVENT,
    emotion_valence: Optional[float] = None,
    emotion_arousal: Optional[float] = None,
    user_id: Optional[str] = None,
    ttl_days: Optional[float] = None,
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "memory_type": memory_type,     # "short_term" | "long_term"
        "source": source,               # "user" | "assistant" | "system" | "observation"
        "importance": importance,        # 0.0 – 1.0
        "tags": tags or [],
        "created_at": _now(),
        "access_count": 0,
        "last_accessed": _now(),
        # Episodic fields
        "session_id": session_id or "",
        "entry_type": entry_type if entry_type in _VALID_ENTRY_TYPES else ENTRY_TYPE_EVENT,
        "user_id": user_id or "",
        # Emotion coupling
        "emotion_valence": emotion_valence,   # -1..+1 or None
        "emotion_arousal": emotion_arousal,   # 0..1 or None
        # Retention
        "ttl_days": ttl_days,                 # None = no expiry
    }
    if extra:
        meta.update(extra)
    return meta


# ------------------------------------------------------------------
# Retention / decay helpers
# ------------------------------------------------------------------

def effective_importance(
    base_importance: float,
    created_at: float,
    access_count: int,
    decay_rate: float,
    now: Optional[float] = None,
) -> float:
    """
    Compute time-decayed importance.

    ``decay_rate`` is the half-life in days — after ``decay_rate`` days
    with no access the importance is halved.  Each access partially
    restores importance.
    """
    now = now or _now()
    age_days = max(0.0, (now - created_at) / 86400.0)
    if decay_rate <= 0:
        decay_factor = 1.0
    else:
        decay_factor = math.pow(0.5, age_days / decay_rate)
    # Access bonus: each access adds a small boost (diminishing)
    access_bonus = min(0.2, access_count * 0.02)
    return min(1.0, base_importance * decay_factor + access_bonus)


def is_expired(
    meta: Dict[str, Any],
    now: Optional[float] = None,
) -> bool:
    """Check if a memory has exceeded its TTL."""
    ttl = meta.get("ttl_days")
    if ttl is None:
        return False
    now = now or _now()
    created = meta.get("created_at", now)
    age_days = (now - created) / 86400.0
    return age_days > ttl


# ------------------------------------------------------------------
# Profile slug
# ------------------------------------------------------------------

def _slugify(name: str) -> str:
    """Convert a profile name to a safe directory name."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "default"


# ------------------------------------------------------------------
# Profile memory
# ------------------------------------------------------------------

@dataclass
class ProfileMemoryConfig:
    max_short_term: int = 50           # FIFO cap
    auto_promote_threshold: float = 0.8  # importance above this → auto-promote
    consolidation_threshold: float = 0.85  # similarity above this → merge
    decay_rate_days: float = 30.0      # importance half-life in days
    ttl_floor: float = 0.05           # below this effective importance → archive
    recency_boost: float = 0.15       # max boost for very recent memories
    emotion_retrieval_bias: float = 0.1  # how much emotion match boosts recall


class ProfileMemory:
    """
    Complete memory for a single AI profile.

    Manages two ``VectorStore`` instances (short-term, long-term) and
    handles promotion, eviction, consolidation, decay, and emotion coupling.
    """

    def __init__(
        self,
        profile_name: str,
        config: Optional[ProfileMemoryConfig] = None,
        base_dir: Optional[Path] = None,
    ):
        self.profile_name = profile_name
        self.cfg = config or ProfileMemoryConfig()
        self._slug = _slugify(profile_name)

        base = base_dir or MEMORY_DATA_DIR
        self._dir = base / self._slug
        self._dir.mkdir(parents=True, exist_ok=True)

        self.short_term = VectorStore(self._dir / "short_term.json")
        self.long_term = VectorStore(self._dir / "long_term.json")

        # Current session ID (auto-generated per init)
        self._session_id: str = _make_session_id()

    # ── Session management ────────────────────────────────────

    @property
    def session_id(self) -> str:
        return self._session_id

    def new_session(self) -> str:
        """Start a new session and return the new session ID."""
        self._session_id = _make_session_id()
        return self._session_id

    # ── Add memories ─────────────────────────────────────────

    def remember(
        self,
        content: str,
        source: str = "user",
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        memory_type: str = "short_term",
        extra_metadata: Optional[Dict[str, Any]] = None,
        *,
        entry_type: str = ENTRY_TYPE_EVENT,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        emotion_valence: Optional[float] = None,
        emotion_arousal: Optional[float] = None,
        ttl_days: Optional[float] = None,
    ) -> str:
        """
        Store a new memory.

        Returns the entry ID.  If ``memory_type`` is ``"short_term"``
        and the entry's importance exceeds the auto-promote threshold,
        it is also written to long-term memory.
        """
        embedding = embed_text(content)
        meta = _make_metadata(
            memory_type=memory_type,
            source=source,
            importance=importance,
            tags=tags,
            extra=extra_metadata,
            session_id=session_id or self._session_id,
            entry_type=entry_type,
            emotion_valence=emotion_valence,
            emotion_arousal=emotion_arousal,
            user_id=user_id,
            ttl_days=ttl_days,
        )

        if memory_type == "long_term":
            return self._add_long_term(content, embedding, meta)

        # Short-term
        entry_id = self.short_term.add(content, embedding, meta)

        # FIFO eviction
        self._enforce_short_term_cap()

        # Auto-promote high-importance memories
        if importance >= self.cfg.auto_promote_threshold:
            self._promote(entry_id)

        return entry_id

    def remember_conversation(
        self,
        user_message: str,
        assistant_response: str,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        *,
        emotion_valence: Optional[float] = None,
        emotion_arousal: Optional[float] = None,
        user_id: Optional[str] = None,
    ) -> tuple[str, str]:
        """
        Convenience: store both sides of a conversation turn.

        Returns (user_entry_id, assistant_entry_id).
        """
        uid = self.remember(
            content=f"User said: {user_message}",
            source="user",
            importance=importance,
            tags=tags,
            entry_type=ENTRY_TYPE_CONVERSATION,
            emotion_valence=emotion_valence,
            emotion_arousal=emotion_arousal,
            user_id=user_id,
        )
        aid = self.remember(
            content=f"Assistant said: {assistant_response}",
            source="assistant",
            importance=importance,
            tags=tags,
            entry_type=ENTRY_TYPE_CONVERSATION,
            emotion_valence=emotion_valence,
            emotion_arousal=emotion_arousal,
        )
        return uid, aid

    def remember_fact(
        self,
        content: str,
        importance: float = 0.7,
        tags: Optional[List[str]] = None,
        source: str = "system",
        **kwargs: Any,
    ) -> str:
        """Store an extracted fact (preference, goal, boundary, etc.)."""
        return self.remember(
            content=content,
            source=source,
            importance=importance,
            tags=tags,
            memory_type="long_term",
            entry_type=ENTRY_TYPE_FACT,
            **kwargs,
        )

    def remember_observation(
        self,
        content: str,
        importance: float = 0.4,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        """Store an observation (vision event, environment change, etc.)."""
        return self.remember(
            content=content,
            source="observation",
            importance=importance,
            tags=tags,
            entry_type=ENTRY_TYPE_OBSERVATION,
            **kwargs,
        )

    # ── Recall ───────────────────────────────────────────────

    def recall(
        self,
        query: str,
        top_k: int = 5,
        include_short_term: bool = True,
        include_long_term: bool = True,
        min_similarity: float = 0.1,
        tags: Optional[List[str]] = None,
        *,
        emotion_valence: Optional[float] = None,
        emotion_arousal: Optional[float] = None,
        entry_types: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """
        Retrieve the most relevant memories for *query*.

        Searches both memory tiers and merges results, ranking by
        a combined score of cosine similarity x decayed importance
        x recency, with optional emotion-salience bias.
        """
        query_emb = embed_text(query)

        tag_set = set(tags) if tags else None
        type_set = set(entry_types) if entry_types else None

        def _filter(entry):
            meta = entry.metadata
            # Tag filter
            if tag_set is not None:
                entry_tags = set(meta.get("tags", []))
                if not (tag_set & entry_tags):
                    return False
            # Entry type filter
            if type_set is not None:
                if meta.get("entry_type", ENTRY_TYPE_EVENT) not in type_set:
                    return False
            # TTL filter
            if is_expired(meta):
                return False
            return True

        results: List[SearchResult] = []

        if include_short_term:
            results.extend(
                self.short_term.search(
                    query_emb, top_k=top_k * 2,
                    threshold=min_similarity,
                    filter_fn=_filter,
                )
            )

        if include_long_term:
            results.extend(
                self.long_term.search(
                    query_emb, top_k=top_k * 2,
                    threshold=min_similarity,
                    filter_fn=_filter,
                )
            )

        now = _now()

        for r in results:
            meta = r.entry.metadata
            imp = meta.get("importance", 0.5)

            # Time-decayed importance
            eff_imp = effective_importance(
                base_importance=imp,
                created_at=meta.get("created_at", now),
                access_count=meta.get("access_count", 0),
                decay_rate=self.cfg.decay_rate_days,
                now=now,
            )

            # Recency boost: memories from last hour get extra weight
            age_hours = max(0.0, (now - meta.get("created_at", now)) / 3600.0)
            recency = self.cfg.recency_boost * math.exp(-age_hours / 24.0)

            # Emotion salience: bias toward memories with matching emotion
            emotion_bonus = 0.0
            if emotion_valence is not None and meta.get("emotion_valence") is not None:
                v_diff = abs(emotion_valence - meta["emotion_valence"])
                a_diff = abs((emotion_arousal or 0.5) - (meta.get("emotion_arousal") or 0.5))
                # Closer emotions = higher bonus
                emotion_bonus = self.cfg.emotion_retrieval_bias * max(0.0, 1.0 - (v_diff + a_diff) / 3.0)

            # Combined score
            r.score = r.score * (0.5 + 0.5 * eff_imp) + recency + emotion_bonus

            # Update access metadata
            meta["access_count"] = meta.get("access_count", 0) + 1
            meta["last_accessed"] = now

        # Deduplicate (same entry could appear in both stores)
        seen_ids = set()
        deduped: List[SearchResult] = []
        for r in results:
            if r.entry.id not in seen_ids:
                seen_ids.add(r.entry.id)
                deduped.append(r)

        deduped.sort(key=lambda r: r.score, reverse=True)
        return deduped[:top_k]

    # ── Promotion / demotion ─────────────────────────────────

    def promote(self, entry_id: str) -> bool:
        """Promote a short-term memory to long-term."""
        return self._promote(entry_id)

    def _promote(self, entry_id: str) -> bool:
        entry = self.short_term.get(entry_id)
        if entry is None:
            return False

        meta = dict(entry.metadata)
        meta["memory_type"] = "long_term"
        meta["promoted_at"] = _now()

        self._add_long_term(entry.content, entry.embedding, meta)
        return True

    # ── Long-term with consolidation ─────────────────────────

    def _add_long_term(
        self,
        content: str,
        embedding: List[float],
        metadata: Dict[str, Any],
    ) -> str:
        """
        Add to long-term memory with optional consolidation.

        If a very similar entry already exists, merge them instead
        of creating a duplicate.
        """
        # Check for near-duplicates
        similar = self.long_term.search(
            embedding, top_k=1,
            threshold=self.cfg.consolidation_threshold,
        )

        if similar:
            existing = similar[0].entry
            # Merge: keep the longer content, bump importance
            if len(content) > len(existing.content):
                merged_content = content
            else:
                merged_content = existing.content

            merged_imp = min(
                1.0,
                max(
                    metadata.get("importance", 0.5),
                    existing.metadata.get("importance", 0.5),
                ) + 0.1,
            )
            existing.metadata["importance"] = merged_imp
            existing.metadata["consolidated_at"] = _now()
            existing.metadata["consolidation_count"] = (
                existing.metadata.get("consolidation_count", 0) + 1
            )

            # Merge emotion: average if both present
            for dim in ("emotion_valence", "emotion_arousal"):
                old_val = existing.metadata.get(dim)
                new_val = metadata.get(dim)
                if old_val is not None and new_val is not None:
                    existing.metadata[dim] = (old_val + new_val) / 2.0
                elif new_val is not None:
                    existing.metadata[dim] = new_val

            # Merge tags
            old_tags = set(existing.metadata.get("tags", []))
            new_tags = set(metadata.get("tags", []))
            existing.metadata["tags"] = sorted(old_tags | new_tags)

            existing.content = merged_content

            # Re-embed if content changed
            if merged_content != existing.content:
                existing.embedding = embed_text(merged_content)

            self.long_term._save()
            return existing.id

        return self.long_term.add(content, embedding, metadata)

    # ── Consolidation pipeline ────────────────────────────────

    def consolidate(
        self,
        summarizer: Optional[Callable[[List[str]], str]] = None,
    ) -> Dict[str, Any]:
        """
        Run the memory consolidation pipeline.

        1. Prune expired (TTL) entries from both stores.
        2. Archive low-importance long-term entries (below ttl_floor
           after decay).
        3. If a ``summarizer`` callback is provided, summarize recent
           short-term entries (grouped by session) into a long-term
           summary entry.

        ``summarizer`` signature: ``fn(texts: List[str]) -> str``

        Returns a dict of stats about what was consolidated/pruned.
        """
        now = _now()
        stats: Dict[str, int] = {
            "pruned_expired": 0,
            "archived_decayed": 0,
            "summaries_created": 0,
        }

        # 1. Prune TTL-expired entries
        for store in (self.short_term, self.long_term):
            for entry in list(store.all_entries()):
                if is_expired(entry.metadata, now):
                    store.remove(entry.id)
                    stats["pruned_expired"] += 1

        # 2. Archive decayed long-term entries
        for entry in list(self.long_term.all_entries()):
            eff = effective_importance(
                base_importance=entry.metadata.get("importance", 0.5),
                created_at=entry.metadata.get("created_at", now),
                access_count=entry.metadata.get("access_count", 0),
                decay_rate=self.cfg.decay_rate_days,
                now=now,
            )
            if eff < self.cfg.ttl_floor:
                self.long_term.remove(entry.id)
                stats["archived_decayed"] += 1

        # 3. Summarize recent short-term into long-term abstractions
        if summarizer and self.short_term.count() >= 5:
            entries = self.short_term.all_entries()
            entries.sort(key=lambda e: e.metadata.get("created_at", 0))

            # Group by session
            sessions: Dict[str, list] = {}
            for e in entries:
                sid = e.metadata.get("session_id", "unknown")
                sessions.setdefault(sid, []).append(e)

            for sid, session_entries in sessions.items():
                if len(session_entries) < 3:
                    continue
                texts = [e.content for e in session_entries]
                try:
                    summary_text = summarizer(texts)
                    if summary_text:
                        # Average emotion from session entries
                        vals = [e.metadata.get("emotion_valence")
                                for e in session_entries
                                if e.metadata.get("emotion_valence") is not None]
                        aros = [e.metadata.get("emotion_arousal")
                                for e in session_entries
                                if e.metadata.get("emotion_arousal") is not None]
                        avg_val = sum(vals) / len(vals) if vals else None
                        avg_aro = sum(aros) / len(aros) if aros else None

                        # Collect all tags from session
                        all_tags: set = set()
                        for e in session_entries:
                            all_tags.update(e.metadata.get("tags", []))

                        self.remember(
                            content=f"[Session Summary] {summary_text}",
                            source="system",
                            importance=0.7,
                            tags=sorted(all_tags),
                            memory_type="long_term",
                            entry_type=ENTRY_TYPE_SUMMARY,
                            session_id=sid,
                            emotion_valence=avg_val,
                            emotion_arousal=avg_aro,
                        )
                        stats["summaries_created"] += 1
                except Exception:
                    pass

        return stats

    # ── Eviction ─────────────────────────────────────────────

    def _enforce_short_term_cap(self) -> None:
        """Evict oldest entries if short-term exceeds capacity."""
        entries = self.short_term.all_entries()
        if len(entries) <= self.cfg.max_short_term:
            return

        # Sort by creation time, evict oldest
        entries.sort(
            key=lambda e: e.metadata.get("created_at", 0),
        )
        to_remove = len(entries) - self.cfg.max_short_term
        for entry in entries[:to_remove]:
            # Auto-promote if important enough before evicting
            imp = entry.metadata.get("importance", 0.5)
            if imp >= self.cfg.auto_promote_threshold:
                self._promote(entry.id)
            self.short_term.remove(entry.id)

    # ── Bulk operations ──────────────────────────────────────

    def clear_short_term(self) -> None:
        """Clear all short-term memory (e.g. on session end)."""
        self.short_term.clear()

    def clear_all(self) -> None:
        """Clear both tiers (destructive)."""
        self.short_term.clear()
        self.long_term.clear()

    def stats(self) -> Dict[str, Any]:
        """Return memory statistics."""
        return {
            "profile": self.profile_name,
            "short_term_count": self.short_term.count(),
            "long_term_count": self.long_term.count(),
            "total": self.short_term.count() + self.long_term.count(),
            "session_id": self._session_id,
        }

    # ── Forget ───────────────────────────────────────────────

    def forget(self, entry_id: str) -> bool:
        """Remove a specific memory from either store."""
        return (
            self.short_term.remove(entry_id)
            or self.long_term.remove(entry_id)
        )

    def forget_about(self, topic: str, threshold: float = 0.5) -> int:
        """
        Remove all memories similar to *topic*.

        Returns the number of entries removed.
        """
        query_emb = embed_text(topic)
        removed = 0

        for store in (self.short_term, self.long_term):
            hits = store.search(query_emb, top_k=100, threshold=threshold)
            for hit in hits:
                store.remove(hit.entry.id)
                removed += 1

        return removed

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

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .embeddings import embed_text, cosine_similarity
from .vector_store import VectorStore, SearchResult

# Root directory for all memory data.
MEMORY_DATA_DIR = Path("memory_data")


# ------------------------------------------------------------------
# Memory entry metadata schema
# ------------------------------------------------------------------

def _now() -> float:
    return time.time()


def _make_metadata(
    memory_type: str,
    source: str = "user",
    importance: float = 0.5,
    tags: Optional[List[str]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "memory_type": memory_type,     # "short_term" | "long_term"
        "source": source,               # "user" | "assistant" | "system" | "observation"
        "importance": importance,        # 0.0 – 1.0
        "tags": tags or [],
        "created_at": _now(),
        "access_count": 0,
        "last_accessed": _now(),
    }
    if extra:
        meta.update(extra)
    return meta


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


class ProfileMemory:
    """
    Complete memory for a single AI profile.

    Manages two ``VectorStore`` instances (short-term, long-term) and
    handles promotion, eviction, and consolidation.
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

    # ── Add memories ─────────────────────────────────────────

    def remember(
        self,
        content: str,
        source: str = "user",
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        memory_type: str = "short_term",
        extra_metadata: Optional[Dict[str, Any]] = None,
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
        )
        aid = self.remember(
            content=f"Assistant said: {assistant_response}",
            source="assistant",
            importance=importance,
            tags=tags,
        )
        return uid, aid

    # ── Recall ───────────────────────────────────────────────

    def recall(
        self,
        query: str,
        top_k: int = 5,
        include_short_term: bool = True,
        include_long_term: bool = True,
        min_similarity: float = 0.1,
        tags: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """
        Retrieve the most relevant memories for *query*.

        Searches both memory tiers and merges results, ranking by
        a combined score of cosine similarity × importance.
        """
        query_emb = embed_text(query)

        tag_set = set(tags) if tags else None

        def _tag_filter(entry):
            if tag_set is None:
                return True
            entry_tags = set(entry.metadata.get("tags", []))
            return bool(tag_set & entry_tags)

        results: List[SearchResult] = []

        if include_short_term:
            results.extend(
                self.short_term.search(
                    query_emb, top_k=top_k * 2,
                    threshold=min_similarity,
                    filter_fn=_tag_filter,
                )
            )

        if include_long_term:
            results.extend(
                self.long_term.search(
                    query_emb, top_k=top_k * 2,
                    threshold=min_similarity,
                    filter_fn=_tag_filter,
                )
            )

        # Score = similarity * (0.5 + 0.5 * importance)
        # This gives importance a boost without drowning out similarity.
        for r in results:
            imp = r.entry.metadata.get("importance", 0.5)
            r.score = r.score * (0.5 + 0.5 * imp)
            # Update access metadata
            r.entry.metadata["access_count"] = r.entry.metadata.get("access_count", 0) + 1
            r.entry.metadata["last_accessed"] = _now()

        # Deduplicate (same entry could theoretically appear in both stores)
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
            existing.content = merged_content

            # Re-embed if content changed
            if merged_content != existing.content:
                existing.embedding = embed_text(merged_content)

            self.long_term._save()
            return existing.id

        return self.long_term.add(content, embedding, metadata)

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

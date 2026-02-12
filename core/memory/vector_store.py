"""
Pure-Python vector store with brute-force cosine-similarity search.

Designed as a lightweight, dependency-free alternative to ChromaDB
or FAISS.  Perfectly adequate for the memory sizes an AI assistant
will accumulate (thousands of entries, not millions).

Features
--------
* Add / remove / update entries by ID.
* Top-K nearest-neighbour search via cosine similarity.
* Filter by arbitrary metadata predicates.
* Persistence to / from a JSON file.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .embeddings import cosine_similarity


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class VectorEntry:
    """A single indexed item in the store."""
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "embedding": self.embedding,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VectorEntry":
        return cls(
            id=d["id"],
            content=d["content"],
            embedding=d["embedding"],
            metadata=d.get("metadata", {}),
        )


@dataclass
class SearchResult:
    """A single search hit."""
    entry: VectorEntry
    score: float  # cosine similarity


# ------------------------------------------------------------------
# Vector store
# ------------------------------------------------------------------

class VectorStore:
    """
    In-memory vector index backed by a JSON file.

    Usage::

        store = VectorStore(path="memories.json")
        store.add("greeting", "Hello, my name is Astra.", embedding, metadata={...})
        results = store.search(query_embedding, top_k=5)
    """

    def __init__(self, path: Optional[Path] = None):
        self._path = Path(path) if path else None
        self._entries: Dict[str, VectorEntry] = {}
        if self._path:
            self._load()

    # ── CRUD ─────────────────────────────────────────────────

    def add(
        self,
        content: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        entry_id: Optional[str] = None,
    ) -> str:
        """
        Add an entry.  Returns its ID.

        If *entry_id* is given and already exists, the entry is updated.
        """
        eid = entry_id or uuid.uuid4().hex[:12]
        self._entries[eid] = VectorEntry(
            id=eid,
            content=content,
            embedding=embedding,
            metadata=metadata or {},
        )
        self._save()
        return eid

    def remove(self, entry_id: str) -> bool:
        """Remove by ID.  Returns True if it existed."""
        if entry_id in self._entries:
            del self._entries[entry_id]
            self._save()
            return True
        return False

    def get(self, entry_id: str) -> Optional[VectorEntry]:
        return self._entries.get(entry_id)

    def count(self) -> int:
        return len(self._entries)

    def all_entries(self) -> List[VectorEntry]:
        return list(self._entries.values())

    def clear(self) -> None:
        self._entries.clear()
        self._save()

    # ── Search ───────────────────────────────────────────────

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        threshold: float = 0.0,
        filter_fn: Optional[Callable[[VectorEntry], bool]] = None,
    ) -> List[SearchResult]:
        """
        Find the *top_k* most similar entries.

        Parameters
        ----------
        query_embedding : list[float]
            The query vector (should be L2-normalised).
        top_k : int
            Max results to return.
        threshold : float
            Minimum cosine similarity to include.
        filter_fn : callable | None
            Optional predicate to pre-filter entries before scoring.
        """
        scored: List[SearchResult] = []

        for entry in self._entries.values():
            if filter_fn and not filter_fn(entry):
                continue

            score = cosine_similarity(query_embedding, entry.embedding)
            if score >= threshold:
                scored.append(SearchResult(entry=entry, score=score))

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    # ── Persistence ──────────────────────────────────────────

    def _save(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = [e.to_dict() for e in self._entries.values()]
            self._path.write_text(
                json.dumps(data, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            for d in raw:
                entry = VectorEntry.from_dict(d)
                self._entries[entry.id] = entry
        except (json.JSONDecodeError, KeyError, OSError):
            pass

    def save_to(self, path: Path) -> None:
        """Save to a specific path (for migration / export)."""
        old_path = self._path
        self._path = path
        self._save()
        self._path = old_path

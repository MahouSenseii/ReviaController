"""
RAG (Retrieval-Augmented Generation) engine — profile-aware memory
orchestrator for the Revia Controller.

This is the main entry point for the memory system.  It:

1. Manages a **ProfileMemory** instance per AI profile.
2. Automatically **switches memory context** when the user selects
   a different profile (via ``profile_selected`` event).
3. Listens for **chat messages** (``chat_message`` event) and stores
   them as short-term memories.
4. Provides ``get_rag_context(query)`` — retrieves relevant memories
   from the active profile and formats them as an LLM prompt block.
5. Publishes ``memory_updated`` events so the UI can react.

Profile isolation
-----------------
Each profile's memories are stored in separate directories::

    memory_data/
        astra/
            short_term.json
            long_term.json
        debugger/
            short_term.json
            long_term.json

"Astra" knows nothing about "Debugger"'s memories and vice versa.

LLM integration
---------------
Call ``get_rag_context(query)`` before building the LLM prompt.  It
returns a dict with:

* ``relevant_memories`` — list of memory dicts with content + score
* ``prompt_injection`` — ready-to-use natural-language block
* ``profile`` — which profile the memories belong to
* ``stats`` — memory statistics

The ``prompt_injection`` can be appended to the system prompt so the
LLM has context from past conversations and stored knowledge.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .memory_store import ProfileMemory, ProfileMemoryConfig, MEMORY_DATA_DIR

# EventBus imported at runtime to avoid circular import when used
# outside a Qt context.
try:
    from ..events import EventBus
except Exception:
    EventBus = None  # type: ignore[misc, assignment]


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

@dataclass
class RAGConfig:
    """Tunable parameters for the RAG engine."""
    memory_config: ProfileMemoryConfig = None  # type: ignore[assignment]
    retrieval_top_k: int = 7             # memories to retrieve per query
    min_similarity: float = 0.1          # below this, don't include
    max_context_tokens: int = 2000       # rough token limit for injection
    base_dir: Path = MEMORY_DATA_DIR     # root memory storage dir
    auto_store_chat: bool = True         # store chat turns automatically
    short_term_importance: float = 0.4   # default importance for chat
    long_term_importance: float = 0.7    # default for explicit "remember"

    def __post_init__(self):
        if self.memory_config is None:
            self.memory_config = ProfileMemoryConfig()


# ------------------------------------------------------------------
# RAG Engine
# ------------------------------------------------------------------

class RAGEngine:
    """
    Profile-aware RAG orchestrator.

    Usage::

        engine = RAGEngine(event_bus)
        # Memories auto-switch when profile changes
        engine.store("User prefers dark mode", importance=0.8)
        ctx = engine.get_rag_context("What theme does the user like?")
        system_prompt += ctx["prompt_injection"]
    """

    def __init__(
        self,
        event_bus: Optional[Any] = None,
        config: Optional[RAGConfig] = None,
    ):
        self.bus = event_bus
        self.cfg = config or RAGConfig()

        # Profile memory instances (keyed by profile name)
        self._profiles: Dict[str, ProfileMemory] = {}

        # Currently active profile
        self._active_profile: Optional[str] = None
        self._active_memory: Optional[ProfileMemory] = None

        # Subscribe to events
        if self.bus is not None:
            self.bus.subscribe("profile_selected", self._on_profile_selected)
            self.bus.subscribe("chat_message", self._on_chat_message)
            self.bus.subscribe("memory_command", self._on_memory_command)

    # ── Profile management ───────────────────────────────────

    def switch_profile(self, profile_name: str) -> None:
        """
        Switch the active memory context to *profile_name*.

        Creates the profile's memory store on first access.
        """
        if profile_name == self._active_profile:
            return

        self._active_profile = profile_name

        if profile_name not in self._profiles:
            self._profiles[profile_name] = ProfileMemory(
                profile_name=profile_name,
                config=self.cfg.memory_config,
                base_dir=self.cfg.base_dir,
            )

        self._active_memory = self._profiles[profile_name]

        if self.bus:
            self.bus.publish("memory_context_switched", {
                "profile": profile_name,
                "stats": self._active_memory.stats(),
            })

    @property
    def active_profile(self) -> Optional[str]:
        return self._active_profile

    @property
    def memory(self) -> Optional[ProfileMemory]:
        """The active profile's memory store."""
        return self._active_memory

    # ── Store ────────────────────────────────────────────────

    def store(
        self,
        content: str,
        source: str = "user",
        importance: float = 0.5,
        memory_type: str = "short_term",
        tags: Optional[List[str]] = None,
        profile: Optional[str] = None,
        *,
        entry_type: str = "event",
        emotion_valence: Optional[float] = None,
        emotion_arousal: Optional[float] = None,
        user_id: Optional[str] = None,
        ttl_days: Optional[float] = None,
    ) -> Optional[str]:
        """
        Store a memory in the specified (or active) profile.

        Returns the entry ID, or None if no profile is active.
        """
        mem = self._get_memory(profile)
        if mem is None:
            return None

        entry_id = mem.remember(
            content=content,
            source=source,
            importance=importance,
            tags=tags,
            memory_type=memory_type,
            entry_type=entry_type,
            emotion_valence=emotion_valence,
            emotion_arousal=emotion_arousal,
            user_id=user_id,
            ttl_days=ttl_days,
        )

        self._publish_update(mem)
        return entry_id

    def store_long_term(
        self,
        content: str,
        source: str = "system",
        importance: float = 0.7,
        tags: Optional[List[str]] = None,
        profile: Optional[str] = None,
    ) -> Optional[str]:
        """Convenience: store directly into long-term memory."""
        return self.store(
            content=content,
            source=source,
            importance=importance,
            memory_type="long_term",
            tags=tags,
            profile=profile,
        )

    def store_conversation(
        self,
        user_message: str,
        assistant_response: str,
        importance: float = 0.4,
        tags: Optional[List[str]] = None,
        profile: Optional[str] = None,
        *,
        emotion_valence: Optional[float] = None,
        emotion_arousal: Optional[float] = None,
        user_id: Optional[str] = None,
    ) -> Optional[tuple[str, str]]:
        """Store both sides of a conversation turn."""
        mem = self._get_memory(profile)
        if mem is None:
            return None

        result = mem.remember_conversation(
            user_message=user_message,
            assistant_response=assistant_response,
            importance=importance,
            tags=tags,
            emotion_valence=emotion_valence,
            emotion_arousal=emotion_arousal,
            user_id=user_id,
        )

        self._publish_update(mem)
        return result

    # ── Recall / RAG ─────────────────────────────────────────

    def recall(
        self,
        query: str,
        top_k: Optional[int] = None,
        include_short_term: bool = True,
        include_long_term: bool = True,
        tags: Optional[List[str]] = None,
        profile: Optional[str] = None,
        *,
        emotion_valence: Optional[float] = None,
        emotion_arousal: Optional[float] = None,
        entry_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories for *query*.

        Returns a list of dicts with ``content``, ``score``,
        ``source``, ``memory_type``, ``importance``, ``tags``,
        ``entry_type``, ``emotion_valence``, ``emotion_arousal``.
        """
        mem = self._get_memory(profile)
        if mem is None:
            return []

        results = mem.recall(
            query=query,
            top_k=top_k or self.cfg.retrieval_top_k,
            include_short_term=include_short_term,
            include_long_term=include_long_term,
            min_similarity=self.cfg.min_similarity,
            tags=tags,
            emotion_valence=emotion_valence,
            emotion_arousal=emotion_arousal,
            entry_types=entry_types,
        )

        return [
            {
                "id": r.entry.id,
                "content": r.entry.content,
                "score": round(r.score, 4),
                "source": r.entry.metadata.get("source", "unknown"),
                "memory_type": r.entry.metadata.get("memory_type", "unknown"),
                "importance": r.entry.metadata.get("importance", 0.5),
                "tags": r.entry.metadata.get("tags", []),
                "entry_type": r.entry.metadata.get("entry_type", "event"),
                "emotion_valence": r.entry.metadata.get("emotion_valence"),
                "emotion_arousal": r.entry.metadata.get("emotion_arousal"),
                "session_id": r.entry.metadata.get("session_id", ""),
            }
            for r in results
        ]

    def get_rag_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        profile: Optional[str] = None,
        *,
        emotion_valence: Optional[float] = None,
        emotion_arousal: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Build a structured RAG context dict for LLM prompt injection.

        The ``prompt_injection`` field is a ready-to-use text block
        that gives the LLM awareness of relevant past memories.
        """
        memories = self.recall(
            query=query,
            top_k=top_k,
            profile=profile,
            emotion_valence=emotion_valence,
            emotion_arousal=emotion_arousal,
        )

        mem = self._get_memory(profile)
        stats = mem.stats() if mem else {}

        if not memories:
            prompt_injection = (
                "[Memory Context]\n"
                "No relevant memories found for this conversation."
            )
        else:
            # Format memories as a numbered list, respecting token budget
            lines: List[str] = []
            total_chars = 0
            char_limit = self.cfg.max_context_tokens * 4  # rough chars-to-tokens

            for i, m in enumerate(memories, 1):
                tier = "LT" if m["memory_type"] == "long_term" else "ST"
                line = f"{i}. [{tier}] {m['content']}"

                if total_chars + len(line) > char_limit:
                    break

                lines.append(line)
                total_chars += len(line)

            memory_block = "\n".join(lines)

            st_count = sum(1 for m in memories if m["memory_type"] == "short_term")
            lt_count = sum(1 for m in memories if m["memory_type"] == "long_term")

            prompt_injection = (
                f"[Memory Context — Profile: {self._active_profile or 'Unknown'}]\n"
                f"Retrieved {len(lines)} relevant memories "
                f"({st_count} recent, {lt_count} long-term):\n"
                f"{memory_block}\n"
                "Use these memories to inform your response where relevant. "
                "Do not fabricate memories that are not listed above."
            )

        return {
            "relevant_memories": memories,
            "prompt_injection": prompt_injection,
            "profile": self._active_profile,
            "stats": stats,
        }

    # ── Forget ───────────────────────────────────────────────

    def forget(self, entry_id: str, profile: Optional[str] = None) -> bool:
        """Remove a specific memory by ID."""
        mem = self._get_memory(profile)
        if mem is None:
            return False
        result = mem.forget(entry_id)
        if result:
            self._publish_update(mem)
        return result

    def forget_about(
        self,
        topic: str,
        threshold: float = 0.5,
        profile: Optional[str] = None,
    ) -> int:
        """Remove all memories related to *topic*."""
        mem = self._get_memory(profile)
        if mem is None:
            return 0
        count = mem.forget_about(topic, threshold)
        if count > 0:
            self._publish_update(mem)
        return count

    # ── Consolidation ─────────────────────────────────────────

    def consolidate(
        self,
        summarizer=None,
        profile: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the memory consolidation pipeline on the active/specified profile."""
        mem = self._get_memory(profile)
        if mem is None:
            return {"error": "no active profile"}
        result = mem.consolidate(summarizer=summarizer)
        self._publish_update(mem)
        if self.bus:
            self.bus.publish("memory_consolidated", {
                "profile": mem.profile_name,
                "result": result,
            })
        return result

    # ── Session management ───────────────────────────────────

    def new_session(self, profile: Optional[str] = None) -> Optional[str]:
        """Start a new session for the active/specified profile."""
        mem = self._get_memory(profile)
        if mem is None:
            return None
        return mem.new_session()

    def end_session(self, profile: Optional[str] = None) -> None:
        """
        End the current session — clear short-term memory.

        Important memories should have already been promoted to
        long-term during the session.
        """
        mem = self._get_memory(profile)
        if mem is None:
            return
        mem.clear_short_term()
        self._publish_update(mem)

    # ── Stats ────────────────────────────────────────────────

    def stats(self, profile: Optional[str] = None) -> Dict[str, Any]:
        """Return memory statistics for the specified or active profile."""
        mem = self._get_memory(profile)
        if mem is None:
            return {"error": "no active profile"}
        return mem.stats()

    def all_profile_stats(self) -> List[Dict[str, Any]]:
        """Return stats for all loaded profiles."""
        return [mem.stats() for mem in self._profiles.values()]

    # ── Event handlers ───────────────────────────────────────

    def _on_profile_selected(self, data: Dict[str, Any]) -> None:
        name = data.get("value")
        if name:
            self.switch_profile(name)

    def _on_chat_message(self, data: Dict[str, Any]) -> None:
        """
        Auto-store chat messages.

        Expected data::

            {
                "role": "user" | "assistant",
                "content": "the message text",
                "importance": 0.5,      # optional
                "tags": ["topic"],       # optional
                "emotion_valence": 0.3,  # optional, from EmotionEngine
                "emotion_arousal": 0.5,  # optional
                "user_id": "...",        # optional
            }
        """
        if not self.cfg.auto_store_chat:
            return
        if self._active_memory is None:
            return

        role = data.get("role", "user")
        content = data.get("content", "")
        if not content:
            return

        importance = data.get("importance", self.cfg.short_term_importance)
        tags = data.get("tags")

        source = "user" if role == "user" else "assistant"
        prefix = "User said" if role == "user" else "Assistant said"

        mem = self._active_memory
        mem.remember(
            content=f"{prefix}: {content}",
            source=source,
            importance=importance,
            tags=tags,
            entry_type="conversation",
            emotion_valence=data.get("emotion_valence"),
            emotion_arousal=data.get("emotion_arousal"),
            user_id=data.get("user_id"),
        )
        self._publish_update(mem)

    def _on_memory_command(self, data: Dict[str, Any]) -> None:
        """
        Handle explicit memory commands from the UI or plugins.

        Expected data::

            {"action": "store_long_term", "content": "...", ...}
            {"action": "forget", "entry_id": "..."}
            {"action": "forget_about", "topic": "..."}
            {"action": "end_session"}
            {"action": "promote", "entry_id": "..."}
        """
        action = data.get("action")
        if action == "store_long_term":
            self.store_long_term(
                content=data.get("content", ""),
                source=data.get("source", "system"),
                importance=data.get("importance", self.cfg.long_term_importance),
                tags=data.get("tags"),
            )
        elif action == "forget":
            self.forget(data.get("entry_id", ""))
        elif action == "forget_about":
            self.forget_about(
                topic=data.get("topic", ""),
                threshold=data.get("threshold", 0.5),
            )
        elif action == "end_session":
            self.end_session()
        elif action == "promote":
            mem = self._active_memory
            if mem:
                mem.promote(data.get("entry_id", ""))
                self._publish_update(mem)
        elif action == "consolidate":
            self.consolidate()
        elif action == "new_session":
            self.new_session()

    # ── Internal helpers ─────────────────────────────────────

    def _get_memory(self, profile: Optional[str] = None) -> Optional[ProfileMemory]:
        """Get the memory store for *profile* or the active profile."""
        if profile:
            if profile not in self._profiles:
                self._profiles[profile] = ProfileMemory(
                    profile_name=profile,
                    config=self.cfg.memory_config,
                    base_dir=self.cfg.base_dir,
                )
            return self._profiles[profile]
        return self._active_memory

    def _publish_update(self, mem: ProfileMemory) -> None:
        if self.bus:
            self.bus.publish("memory_updated", {
                "profile": mem.profile_name,
                "stats": mem.stats(),
            })

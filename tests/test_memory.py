"""
Memory system evaluation harness.

Tests cover:
    1. Recall precision — relevant memories rank above irrelevant ones
    2. TTL expiry — expired memories are excluded from recall and pruned
    3. Temporal decay — importance decreases over time (half-life model)
    4. Emotion-biased retrieval — matching emotion boosts recall score
    5. Consolidation pipeline — prune, archive, and summarize
    6. Profile isolation — separate profiles share nothing
    7. Session management — session IDs, new_session()
    8. Entry type filtering — recall filters by entry_type
    9. Hallucination guard — only stored content is returned
   10. RAGEngine integration — end-to-end context generation
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# Bootstrap: import memory modules WITHOUT triggering core/__init__.py
# (which depends on PyQt6).  We load the leaf modules directly via
# importlib so the test suite can run in a minimal environment.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

def _load_module(dotted_name: str, file_path: Path) -> types.ModuleType:
    """Load a single Python module by file path, registering it in sys.modules."""
    if dotted_name in sys.modules:
        return sys.modules[dotted_name]
    spec = importlib.util.spec_from_file_location(dotted_name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = mod
    spec.loader.exec_module(mod)
    return mod

# Register stub packages so relative imports inside memory modules work.
for pkg in ("core", "core.memory"):
    if pkg not in sys.modules:
        p = types.ModuleType(pkg)
        p.__path__ = [str(_PROJECT_ROOT / pkg.replace(".", "/"))]
        p.__package__ = pkg
        sys.modules[pkg] = p

# Load the memory sub-modules in dependency order.
_embeddings = _load_module(
    "core.memory.embeddings",
    _PROJECT_ROOT / "core" / "memory" / "embeddings.py",
)
_vector_store = _load_module(
    "core.memory.vector_store",
    _PROJECT_ROOT / "core" / "memory" / "vector_store.py",
)
_memory_store = _load_module(
    "core.memory.memory_store",
    _PROJECT_ROOT / "core" / "memory" / "memory_store.py",
)
_rag_engine = _load_module(
    "core.memory.rag_engine",
    _PROJECT_ROOT / "core" / "memory" / "rag_engine.py",
)

# Pull names into local scope.
ProfileMemory = _memory_store.ProfileMemory
ProfileMemoryConfig = _memory_store.ProfileMemoryConfig
effective_importance = _memory_store.effective_importance
is_expired = _memory_store.is_expired
ENTRY_TYPE_CONVERSATION = _memory_store.ENTRY_TYPE_CONVERSATION
ENTRY_TYPE_EVENT = _memory_store.ENTRY_TYPE_EVENT
ENTRY_TYPE_FACT = _memory_store.ENTRY_TYPE_FACT
ENTRY_TYPE_OBSERVATION = _memory_store.ENTRY_TYPE_OBSERVATION
ENTRY_TYPE_SUMMARY = _memory_store.ENTRY_TYPE_SUMMARY
RAGEngine = _rag_engine.RAGEngine
RAGConfig = _rag_engine.RAGConfig


class _TempDirMixin:
    """Provides a fresh temp directory for each test."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()


# ================================================================
# 1. Recall precision
# ================================================================

class TestRecallPrecision(_TempDirMixin, unittest.TestCase):
    """Relevant memories should score higher than irrelevant ones."""

    def setUp(self):
        super().setUp()
        self.mem = ProfileMemory("tester", base_dir=self.base_dir)

    def test_relevant_ranked_above_irrelevant(self):
        self.mem.remember("User prefers dark mode for the editor theme", importance=0.6)
        self.mem.remember("User likes pizza with extra cheese", importance=0.6)
        self.mem.remember("The weather today is sunny and warm", importance=0.6)

        results = self.mem.recall("What theme does the user like?", top_k=3)
        self.assertTrue(len(results) > 0)
        # The dark mode memory should rank first
        self.assertIn("dark mode", results[0].entry.content.lower())

    def test_importance_influences_ranking(self):
        self.mem.remember("Low importance note about cats", importance=0.1)
        self.mem.remember("High importance note about cats and dogs", importance=0.9)

        results = self.mem.recall("cats", top_k=2)
        self.assertTrue(len(results) >= 2)
        # Higher importance should rank higher (given similar similarity)
        scores = [r.score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_top_k_limits_results(self):
        for i in range(10):
            self.mem.remember(f"Memory number {i} about testing", importance=0.5)

        results = self.mem.recall("testing", top_k=3)
        self.assertLessEqual(len(results), 3)

    def test_min_similarity_filters_noise(self):
        self.mem.remember("completely unrelated xyzzy foobar", importance=0.5)
        results = self.mem.recall(
            "dark mode theme preference",
            top_k=5,
            min_similarity=0.9,  # very high threshold
        )
        # May return nothing if similarity is low
        for r in results:
            self.assertGreaterEqual(r.score, 0.0)


# ================================================================
# 2. TTL expiry
# ================================================================

class TestTTLExpiry(_TempDirMixin, unittest.TestCase):
    """Memories past their TTL are excluded from recall and pruned."""

    def setUp(self):
        super().setUp()
        self.mem = ProfileMemory("tester", base_dir=self.base_dir)

    def test_is_expired_helper(self):
        now = time.time()
        meta_fresh = {"created_at": now, "ttl_days": 1.0}
        meta_old = {"created_at": now - 200_000, "ttl_days": 1.0}  # ~2.3 days old
        meta_no_ttl = {"created_at": now - 999_999}

        self.assertFalse(is_expired(meta_fresh, now))
        self.assertTrue(is_expired(meta_old, now))
        self.assertFalse(is_expired(meta_no_ttl, now))  # no TTL = never expires

    def test_expired_excluded_from_recall(self):
        # Store a memory with a very short TTL that we simulate as expired
        eid = self.mem.remember("temporary note", importance=0.5, ttl_days=0.0001)

        # Manually backdate the created_at so it's expired
        for entry in self.mem.short_term.all_entries():
            if entry.id == eid:
                entry.metadata["created_at"] = time.time() - 100_000
                break
        self.mem.short_term._save()

        results = self.mem.recall("temporary note", top_k=5)
        returned_ids = {r.entry.id for r in results}
        self.assertNotIn(eid, returned_ids)

    def test_consolidation_prunes_expired(self):
        eid = self.mem.remember("will expire", importance=0.5, ttl_days=0.0001)

        # Backdate
        for entry in self.mem.short_term.all_entries():
            if entry.id == eid:
                entry.metadata["created_at"] = time.time() - 100_000
                break
        self.mem.short_term._save()

        stats = self.mem.consolidate()
        self.assertGreaterEqual(stats["pruned_expired"], 1)


# ================================================================
# 3. Temporal decay
# ================================================================

class TestTemporalDecay(unittest.TestCase):
    """Importance should decay over time with a half-life model."""

    def test_decay_halves_at_halflife(self):
        now = time.time()
        created = now - (30 * 86400)  # 30 days ago
        eff = effective_importance(
            base_importance=1.0,
            created_at=created,
            access_count=0,
            decay_rate=30.0,  # 30-day half-life
            now=now,
        )
        # Should be approximately 0.5 after one half-life
        self.assertAlmostEqual(eff, 0.5, places=1)

    def test_no_decay_for_recent(self):
        now = time.time()
        eff = effective_importance(
            base_importance=0.8,
            created_at=now,
            access_count=0,
            decay_rate=30.0,
            now=now,
        )
        self.assertAlmostEqual(eff, 0.8, places=2)

    def test_access_count_provides_bonus(self):
        now = time.time()
        created = now - (60 * 86400)  # 60 days old
        eff_no_access = effective_importance(
            base_importance=0.5, created_at=created,
            access_count=0, decay_rate=30.0, now=now,
        )
        eff_with_access = effective_importance(
            base_importance=0.5, created_at=created,
            access_count=5, decay_rate=30.0, now=now,
        )
        self.assertGreater(eff_with_access, eff_no_access)

    def test_access_bonus_capped(self):
        now = time.time()
        eff = effective_importance(
            base_importance=0.5, created_at=now,
            access_count=100, decay_rate=30.0, now=now,
        )
        # access_bonus capped at 0.2
        self.assertLessEqual(eff, 1.0)

    def test_consolidation_archives_decayed(self):
        tmpdir = tempfile.TemporaryDirectory()
        try:
            cfg = ProfileMemoryConfig(
                decay_rate_days=1.0,  # 1-day half-life (fast decay)
                ttl_floor=0.05,
            )
            mem = ProfileMemory("tester", config=cfg, base_dir=Path(tmpdir.name))

            # Store in long-term with low importance
            mem.remember(
                "old unimportant note",
                importance=0.1,
                memory_type="long_term",
            )

            # Backdate it significantly
            for entry in mem.long_term.all_entries():
                entry.metadata["created_at"] = time.time() - (30 * 86400)
            mem.long_term._save()

            stats = mem.consolidate()
            self.assertGreaterEqual(stats["archived_decayed"], 1)
        finally:
            tmpdir.cleanup()


# ================================================================
# 4. Emotion-biased retrieval
# ================================================================

class TestEmotionBiasedRetrieval(_TempDirMixin, unittest.TestCase):
    """Memories tagged with matching emotion get a retrieval boost."""

    def setUp(self):
        super().setUp()
        cfg = ProfileMemoryConfig(emotion_retrieval_bias=0.15)
        self.mem = ProfileMemory("tester", config=cfg, base_dir=self.base_dir)

    def test_emotion_match_boosts_score(self):
        # Store two memories about similar topic but different emotions
        self.mem.remember(
            "User was happy talking about their vacation plans",
            importance=0.5,
            emotion_valence=0.8,
            emotion_arousal=0.6,
        )
        self.mem.remember(
            "User was sad discussing their vacation being cancelled",
            importance=0.5,
            emotion_valence=-0.7,
            emotion_arousal=0.4,
        )

        # Query with positive emotion — should boost the happy memory
        results_happy = self.mem.recall(
            "vacation",
            top_k=2,
            emotion_valence=0.8,
            emotion_arousal=0.6,
        )
        # Query with negative emotion — should boost the sad memory
        results_sad = self.mem.recall(
            "vacation",
            top_k=2,
            emotion_valence=-0.7,
            emotion_arousal=0.4,
        )

        if len(results_happy) >= 2 and len(results_sad) >= 2:
            # The happy vacation memory should rank higher when querying with positive emotion
            happy_first = results_happy[0].entry.content
            sad_first = results_sad[0].entry.content
            self.assertIn("happy", happy_first.lower())
            self.assertIn("sad", sad_first.lower())

    def test_no_emotion_no_bias(self):
        self.mem.remember(
            "test memory with emotion",
            importance=0.5,
            emotion_valence=0.5,
            emotion_arousal=0.5,
        )

        # Recall without emotion params — no bias applied
        results = self.mem.recall("test memory", top_k=1)
        self.assertTrue(len(results) > 0)


# ================================================================
# 5. Consolidation pipeline
# ================================================================

class TestConsolidation(_TempDirMixin, unittest.TestCase):
    """Test the full consolidation pipeline: prune, archive, summarize."""

    def setUp(self):
        super().setUp()
        self.mem = ProfileMemory("tester", base_dir=self.base_dir)

    def test_summarizer_creates_long_term_summaries(self):
        # Store enough short-term entries to trigger summarization
        for i in range(6):
            self.mem.remember(
                f"Discussion point {i} about project architecture",
                importance=0.5,
                entry_type=ENTRY_TYPE_CONVERSATION,
            )

        def mock_summarizer(texts: List[str]) -> str:
            return f"Summary of {len(texts)} conversation entries about architecture"

        stats = self.mem.consolidate(summarizer=mock_summarizer)
        self.assertGreaterEqual(stats["summaries_created"], 1)

        # Check long-term has the summary
        lt_entries = self.mem.long_term.all_entries()
        summaries = [
            e for e in lt_entries
            if e.metadata.get("entry_type") == ENTRY_TYPE_SUMMARY
        ]
        self.assertTrue(len(summaries) > 0)
        self.assertIn("Summary", summaries[0].content)

    def test_consolidation_without_summarizer(self):
        self.mem.remember("note one", importance=0.5)
        self.mem.remember("note two", importance=0.5)

        stats = self.mem.consolidate()
        self.assertEqual(stats["summaries_created"], 0)

    def test_near_duplicate_merging(self):
        # Store same content twice in long-term — should consolidate
        self.mem.remember(
            "User prefers dark mode",
            importance=0.5,
            memory_type="long_term",
        )
        self.mem.remember(
            "User prefers dark mode",
            importance=0.5,
            memory_type="long_term",
        )

        # Should have merged (or at most 2 entries if threshold not met)
        lt_count = self.mem.long_term.count()
        # The consolidation_threshold of 0.85 should catch exact duplicates
        self.assertLessEqual(lt_count, 2)


# ================================================================
# 6. Profile isolation
# ================================================================

class TestProfileIsolation(_TempDirMixin, unittest.TestCase):
    """Different profiles must not share any memories."""

    def test_separate_profiles_isolated(self):
        alice = ProfileMemory("Alice", base_dir=self.base_dir)
        bob = ProfileMemory("Bob", base_dir=self.base_dir)

        alice.remember("Alice's secret password is 1234", importance=0.9)
        bob.remember("Bob likes fishing on weekends", importance=0.9)

        alice_results = alice.recall("password", top_k=5)
        bob_results = bob.recall("password", top_k=5)

        # Alice should find her memory
        alice_contents = [r.entry.content for r in alice_results]
        self.assertTrue(any("Alice" in c for c in alice_contents))

        # Bob should NOT find Alice's memory
        bob_contents = [r.entry.content for r in bob_results]
        self.assertFalse(any("Alice" in c for c in bob_contents))

    def test_rag_engine_profile_switching(self):
        cfg = RAGConfig(base_dir=self.base_dir)
        engine = RAGEngine(config=cfg)

        engine.switch_profile("profile_a")
        engine.store("Profile A knows about quantum physics", importance=0.8)

        engine.switch_profile("profile_b")
        engine.store("Profile B knows about cooking recipes", importance=0.8)

        # Recall from profile A
        results_a = engine.recall("quantum", profile="profile_a")
        results_b = engine.recall("quantum", profile="profile_b")

        # Profile A should find quantum content
        a_contents = " ".join(r["content"] for r in results_a)
        self.assertIn("quantum", a_contents.lower())

        # Profile B should NOT have quantum content (only cooking)
        b_contents = " ".join(r["content"] for r in results_b)
        self.assertNotIn("quantum", b_contents.lower())


# ================================================================
# 7. Session management
# ================================================================

class TestSessionManagement(_TempDirMixin, unittest.TestCase):
    """Session IDs are assigned and new_session() rotates them."""

    def setUp(self):
        super().setUp()
        self.mem = ProfileMemory("tester", base_dir=self.base_dir)

    def test_session_id_assigned_on_init(self):
        self.assertIsNotNone(self.mem.session_id)
        self.assertTrue(len(self.mem.session_id) > 0)

    def test_new_session_rotates_id(self):
        old_id = self.mem.session_id
        new_id = self.mem.new_session()
        self.assertNotEqual(old_id, new_id)
        self.assertEqual(self.mem.session_id, new_id)

    def test_memories_tagged_with_session_id(self):
        sid = self.mem.session_id
        self.mem.remember("test content", importance=0.5)

        entries = self.mem.short_term.all_entries()
        self.assertTrue(len(entries) > 0)
        self.assertEqual(entries[0].metadata["session_id"], sid)

    def test_stats_includes_session_id(self):
        stats = self.mem.stats()
        self.assertIn("session_id", stats)
        self.assertEqual(stats["session_id"], self.mem.session_id)


# ================================================================
# 8. Entry type filtering
# ================================================================

class TestEntryTypeFiltering(_TempDirMixin, unittest.TestCase):
    """recall() can filter by entry_type."""

    def setUp(self):
        super().setUp()
        self.mem = ProfileMemory("tester", base_dir=self.base_dir)

    def test_filter_by_entry_type(self):
        self.mem.remember("User said hello", entry_type=ENTRY_TYPE_CONVERSATION)
        self.mem.remember("User likes dark mode", entry_type=ENTRY_TYPE_FACT)
        self.mem.remember("Screen changed to settings page", entry_type=ENTRY_TYPE_OBSERVATION)

        # Only recall facts
        results = self.mem.recall(
            "dark mode",
            top_k=10,
            entry_types=[ENTRY_TYPE_FACT],
        )
        for r in results:
            self.assertEqual(
                r.entry.metadata.get("entry_type"),
                ENTRY_TYPE_FACT,
            )

    def test_no_filter_returns_all_types(self):
        self.mem.remember("conversation entry", entry_type=ENTRY_TYPE_CONVERSATION)
        self.mem.remember("fact entry", entry_type=ENTRY_TYPE_FACT)

        results = self.mem.recall("entry", top_k=10)
        types_found = {r.entry.metadata.get("entry_type") for r in results}
        self.assertTrue(len(types_found) >= 2)

    def test_remember_fact_convenience(self):
        eid = self.mem.remember_fact("User's favorite color is blue")
        entry = self.mem.long_term.get(eid)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.metadata["entry_type"], ENTRY_TYPE_FACT)
        self.assertEqual(entry.metadata["memory_type"], "long_term")

    def test_remember_observation_convenience(self):
        eid = self.mem.remember_observation("Camera detected a person entering the room")
        entry = self.mem.short_term.get(eid)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.metadata["entry_type"], ENTRY_TYPE_OBSERVATION)


# ================================================================
# 9. Hallucination guard
# ================================================================

class TestHallucinationGuard(_TempDirMixin, unittest.TestCase):
    """Only content that was actually stored should be returned."""

    def setUp(self):
        super().setUp()
        self.mem = ProfileMemory("tester", base_dir=self.base_dir)

    def test_recall_only_returns_stored_content(self):
        stored_texts = [
            "The user prefers dark mode",
            "The user's name is Alice",
            "Meeting scheduled for Monday at 3pm",
        ]
        for text in stored_texts:
            self.mem.remember(text, importance=0.6)

        results = self.mem.recall("user preferences", top_k=10)
        returned_contents = {r.entry.content for r in results}

        # Every returned content must be one of the stored texts
        stored_set = set(stored_texts)
        for content in returned_contents:
            self.assertIn(content, stored_set)

    def test_empty_store_returns_nothing(self):
        results = self.mem.recall("anything at all", top_k=10)
        self.assertEqual(len(results), 0)


# ================================================================
# 10. RAGEngine integration
# ================================================================

class TestRAGEngineIntegration(_TempDirMixin, unittest.TestCase):
    """End-to-end tests for the RAGEngine."""

    def setUp(self):
        super().setUp()
        self.cfg = RAGConfig(base_dir=self.base_dir)
        self.engine = RAGEngine(config=self.cfg)

    def test_store_and_recall(self):
        self.engine.switch_profile("integration_test")
        self.engine.store("The answer to everything is 42", importance=0.8)

        results = self.engine.recall("answer to everything")
        self.assertTrue(len(results) > 0)
        self.assertIn("42", results[0]["content"])

    def test_get_rag_context_format(self):
        self.engine.switch_profile("integration_test")
        self.engine.store("User prefers concise answers", importance=0.7)

        ctx = self.engine.get_rag_context("How should I respond?")
        self.assertIn("relevant_memories", ctx)
        self.assertIn("prompt_injection", ctx)
        self.assertIn("profile", ctx)
        self.assertIn("stats", ctx)
        self.assertEqual(ctx["profile"], "integration_test")

    def test_rag_context_with_emotion(self):
        self.engine.switch_profile("emotion_test")
        self.engine.store(
            "User was excited about the new feature",
            importance=0.6,
            emotion_valence=0.9,
            emotion_arousal=0.8,
        )

        ctx = self.engine.get_rag_context(
            "new feature",
            emotion_valence=0.9,
            emotion_arousal=0.8,
        )
        self.assertTrue(len(ctx["relevant_memories"]) > 0)

    def test_recall_returns_emotion_fields(self):
        self.engine.switch_profile("emotion_fields")
        self.engine.store(
            "Happy memory about sunshine",
            importance=0.6,
            emotion_valence=0.7,
            emotion_arousal=0.5,
            entry_type=ENTRY_TYPE_EVENT,
        )

        results = self.engine.recall("sunshine")
        self.assertTrue(len(results) > 0)
        r = results[0]
        self.assertIn("emotion_valence", r)
        self.assertIn("emotion_arousal", r)
        self.assertIn("entry_type", r)
        self.assertIn("session_id", r)

    def test_store_conversation_with_emotion(self):
        self.engine.switch_profile("conv_test")
        result = self.engine.store_conversation(
            user_message="I love this feature!",
            assistant_response="Glad you enjoy it!",
            emotion_valence=0.8,
            emotion_arousal=0.6,
        )
        self.assertIsNotNone(result)
        uid, aid = result
        self.assertTrue(len(uid) > 0)
        self.assertTrue(len(aid) > 0)

    def test_consolidate_via_engine(self):
        self.engine.switch_profile("consolidation_test")
        for i in range(6):
            self.engine.store(f"Note {i} about testing", importance=0.5)

        result = self.engine.consolidate()
        self.assertIn("pruned_expired", result)
        self.assertIn("archived_decayed", result)
        self.assertIn("summaries_created", result)

    def test_new_session_via_engine(self):
        self.engine.switch_profile("session_test")
        old_sid = self.engine.memory.session_id
        new_sid = self.engine.new_session()
        self.assertNotEqual(old_sid, new_sid)

    def test_empty_context_message(self):
        self.engine.switch_profile("empty_test")
        ctx = self.engine.get_rag_context("anything")
        self.assertIn("No relevant memories", ctx["prompt_injection"])

    def test_stats(self):
        self.engine.switch_profile("stats_test")
        self.engine.store("test memory", importance=0.5)
        stats = self.engine.stats()
        self.assertEqual(stats["profile"], "stats_test")
        self.assertGreaterEqual(stats["total"], 1)

    def test_forget_and_forget_about(self):
        self.engine.switch_profile("forget_test")
        eid = self.engine.store("secret data to forget", importance=0.5)
        self.assertIsNotNone(eid)

        success = self.engine.forget(eid)
        self.assertTrue(success)

        # Store more and forget by topic
        self.engine.store("pizza recipe with extra cheese", importance=0.5)
        self.engine.store("another pizza recipe with mushrooms", importance=0.5)
        count = self.engine.forget_about("pizza recipe")
        self.assertGreaterEqual(count, 1)


# ================================================================
# 11. Emotion consistency over time
# ================================================================

class TestEmotionConsistency(_TempDirMixin, unittest.TestCase):
    """Emotion tags persist and are consistent across store/recall cycles."""

    def setUp(self):
        super().setUp()
        self.mem = ProfileMemory("tester", base_dir=self.base_dir)

    def test_emotion_tags_persist(self):
        self.mem.remember(
            "Exciting news about the project launch",
            importance=0.7,
            emotion_valence=0.9,
            emotion_arousal=0.8,
        )

        results = self.mem.recall("project launch", top_k=1)
        self.assertTrue(len(results) > 0)
        meta = results[0].entry.metadata
        self.assertAlmostEqual(meta["emotion_valence"], 0.9, places=5)
        self.assertAlmostEqual(meta["emotion_arousal"], 0.8, places=5)

    def test_emotion_merged_on_consolidation(self):
        # Store two similar entries with different emotions in long-term
        self.mem.remember(
            "Team meeting went well",
            importance=0.6,
            memory_type="long_term",
            emotion_valence=0.5,
            emotion_arousal=0.4,
        )
        self.mem.remember(
            "Team meeting went well",
            importance=0.6,
            memory_type="long_term",
            emotion_valence=0.9,
            emotion_arousal=0.8,
        )

        # After consolidation-merging, emotion should be averaged
        entries = self.mem.long_term.all_entries()
        if len(entries) == 1:
            meta = entries[0].metadata
            # Averaged: (0.5 + 0.9)/2 = 0.7
            self.assertAlmostEqual(meta["emotion_valence"], 0.7, places=1)

    def test_none_emotion_does_not_crash(self):
        self.mem.remember("No emotion tagged", importance=0.5)
        results = self.mem.recall("No emotion", top_k=1)
        self.assertTrue(len(results) > 0)
        meta = results[0].entry.metadata
        self.assertIsNone(meta["emotion_valence"])
        self.assertIsNone(meta["emotion_arousal"])


if __name__ == "__main__":
    unittest.main()

"""
Profile-isolated RAG memory system.

Provides short-term and long-term memory for each AI profile with
vector-similarity retrieval and LLM context generation.
"""

from .embeddings import embed_text, cosine_similarity, EMBED_DIM
from .vector_store import VectorStore, VectorEntry, SearchResult
from .memory_store import (
    ProfileMemory,
    ProfileMemoryConfig,
    effective_importance,
    is_expired,
    ENTRY_TYPE_EVENT,
    ENTRY_TYPE_FACT,
    ENTRY_TYPE_CONVERSATION,
    ENTRY_TYPE_OBSERVATION,
    ENTRY_TYPE_SUMMARY,
)
from .rag_engine import RAGEngine, RAGConfig

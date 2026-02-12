"""
Profile-isolated RAG memory system.

Provides short-term and long-term memory for each AI profile with
vector-similarity retrieval and LLM context generation.
"""

from .embeddings import embed_text, cosine_similarity, EMBED_DIM
from .vector_store import VectorStore, VectorEntry, SearchResult
from .memory_store import ProfileMemory, ProfileMemoryConfig
from .rag_engine import RAGEngine, RAGConfig

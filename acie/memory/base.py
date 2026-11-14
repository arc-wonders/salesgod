"""Semantic long-term memory protocols (Layer 8).

A MemoryStore keeps short texts (call summaries) per customer and returns the most similar ones for a query.
An Embedder turns texts into fixed-size vectors. Infra's PgVectorMemoryStore implements MemoryStore.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class MemoryStore(Protocol):
    async def add(self, customer_id: str, text: str, meta: dict) -> None: ...

    async def search(self, customer_id: str, query: str, k: int = 3) -> list[dict]:
        """Best first; each item is {"text": str, "meta": dict, "score": float}."""
        ...


def memory_enabled() -> bool:
    return os.getenv("ACIE_MEMORY", "0") == "1"


def make_embedder() -> Embedder:
    """ACIE_EMBED=st picks sentence-transformers (falls back to hashing if it cannot load)."""
    from .hashing import HashingEmbedder
    if os.getenv("ACIE_EMBED", "hashing") == "st":
        try:
            from .st import SentenceTransformerEmbedder
            return SentenceTransformerEmbedder()
        except Exception:
            pass
    return HashingEmbedder()


def make_store(embedder: Embedder | None = None) -> MemoryStore:
    """ACIE_MEMORY_STORE=memory (default) | pgvector (infra lane's acie.store.pgvector_memory, imported lazily)."""
    emb = embedder or make_embedder()
    if os.getenv("ACIE_MEMORY_STORE", "memory") == "pgvector":
        try:
            from acie.store.pgvector_memory import PgVectorMemoryStore  # type: ignore
            return PgVectorMemoryStore(embedder=emb)
        except Exception:
            pass  # fall back to in-process memory: never break a call over memory
    from .inmemory import InMemoryStore
    return InMemoryStore(emb)

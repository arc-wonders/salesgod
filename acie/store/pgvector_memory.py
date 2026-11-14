"""Semantic memory in Postgres + pgvector. Implements the feedback lane's MemoryStore protocol
(acie/memory/base.py); a local copy of the protocol shapes is kept here so this file has no cross-lane import."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Protocol


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class MemoryStore(Protocol):
    async def add(self, customer_id: str, text: str, meta: dict) -> None: ...
    async def search(self, customer_id: str, query: str, k: int = 3) -> list[dict]: ...


def _vec(v) -> str:
    return "[" + ",".join(f"{float(x):.7g}" for x in v) + "]"


class PgVectorMemoryStore:
    def __init__(self, dsn: str, embedder: Embedder, dim: int, pool=None, index: str = "hnsw") -> None:
        self.dsn, self.embedder, self.dim, self.index, self._pool = dsn, embedder, dim, index, pool
        self._lock = asyncio.Lock()

    async def _get_pool(self):
        if self._pool is None:
            async with self._lock:
                if self._pool is None:
                    import asyncpg  # lazy
                    pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=5)
                    async with pool.acquire() as c:
                        for stmt in self.schema():
                            await c.execute(stmt)
                    self._pool = pool
        return self._pool

    def schema(self) -> list[str]:
        ops = ("USING hnsw (embedding vector_cosine_ops)" if self.index == "hnsw"
               else "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)")
        return [
            "CREATE EXTENSION IF NOT EXISTS vector",
            f"""CREATE TABLE IF NOT EXISTS memories (
                id text PRIMARY KEY, customer_id text NOT NULL, text text NOT NULL, meta jsonb NOT NULL DEFAULT '{{}}',
                embedding vector({self.dim}) NOT NULL, created_at timestamptz NOT NULL DEFAULT now())""",
            "CREATE INDEX IF NOT EXISTS memories_customer_idx ON memories (customer_id)",
            f"CREATE INDEX IF NOT EXISTS memories_embedding_idx ON memories {ops}",
        ]

    async def add(self, customer_id: str, text: str, meta: dict) -> None:
        (emb,) = await asyncio.to_thread(self.embedder.embed, [text])
        if len(emb) != self.dim:
            raise ValueError(f"embedding has {len(emb)} dims, store expects {self.dim}")
        pool = await self._get_pool()
        async with pool.acquire() as c:
            await c.execute("INSERT INTO memories(id, customer_id, text, meta, embedding) VALUES($1,$2,$3,$4::jsonb,$5::vector)",
                            uuid.uuid4().hex, customer_id, text, json.dumps(meta or {}), _vec(emb))

    async def search(self, customer_id: str, query: str, k: int = 3) -> list[dict]:
        (emb,) = await asyncio.to_thread(self.embedder.embed, [query])
        pool = await self._get_pool()
        async with pool.acquire() as c:
            rows = await c.fetch(
                "SELECT text, meta::text AS meta, 1 - (embedding <=> $2::vector) AS score FROM memories "
                "WHERE customer_id=$1 ORDER BY embedding <=> $2::vector LIMIT $3", customer_id, _vec(emb), k)
        return [{"text": r["text"], "meta": json.loads(r["meta"]), "score": float(r["score"])} for r in rows]

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()

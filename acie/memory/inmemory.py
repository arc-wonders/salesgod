"""In-process MemoryStore with brute-force cosine search (fine for dev and per-customer call counts)."""

from __future__ import annotations

import math

from .base import Embedder
from .hashing import HashingEmbedder


def cosine(a: list[float], b: list[float]) -> float:
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


class InMemoryStore:
    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder or HashingEmbedder()
        self._rows: dict[str, list[tuple[str, dict, list[float]]]] = {}

    async def add(self, customer_id: str, text: str, meta: dict) -> None:
        vec = self.embedder.embed([text])[0]
        self._rows.setdefault(customer_id, []).append((text, dict(meta), vec))

    async def search(self, customer_id: str, query: str, k: int = 3) -> list[dict]:
        rows = self._rows.get(customer_id, [])
        if not rows:
            return []
        if not query.strip():  # nothing to match on: the most recent memories
            return [{"text": t, "meta": m, "score": 0.0} for t, m, _ in reversed(rows[-k:])]
        q = self.embedder.embed([query])[0]
        scored = sorted(((cosine(q, v), i) for i, (_, _, v) in enumerate(rows)), reverse=True)[:k]
        return [{"text": rows[i][0], "meta": rows[i][1], "score": round(s, 4)} for s, i in scored]

"""Optional sentence-transformers embedder (multilingual). Heavy: imported lazily, model from ACIE_EMBED_MODEL."""

from __future__ import annotations

import os


class SentenceTransformerEmbedder:
    def __init__(self, model: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer  # lazy: optional dependency
        self.name = model or os.getenv("ACIE_EMBED_MODEL", "intfloat/multilingual-e5-small")
        self._m = SentenceTransformer(self.name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        # e5 models expect a "query: " / "passage: " prefix; one symmetric prefix works for similarity
        pre = "query: " if "e5" in self.name else ""
        return [list(map(float, v)) for v in self._m.encode([pre + t for t in texts], normalize_embeddings=True)]

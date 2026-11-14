"""Dependency-free embedder: hashed bag of words + char n-grams, 256-d, L2-normalised.

Tokenising splits on whitespace/punctuation (not on \\w, which would cut Devanagari at every matra), so
Devanagari and Roman Hindi both work. Char 3/4-grams give some robustness to spelling variants.
"""

from __future__ import annotations

import math
import re
import zlib

_SPLIT = re.compile(r"[\s\.,;:!?\"'()\[\]{}|/\\\-_।॥“”‘’…]+")


def tokens(text: str) -> list[str]:
    return [t for t in _SPLIT.split(text.lower()) if t]


class HashingEmbedder:
    def __init__(self, dim: int = 256, ngram_weight: float = 0.5) -> None:
        self.dim, self.ngram_weight = dim, ngram_weight

    def _add(self, v: list[float], feat: str, w: float) -> None:
        h = zlib.crc32(feat.encode("utf-8"))
        v[h % self.dim] += w if (h >> 16) & 1 else -w

    def _one(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in tokens(text):
            self._add(v, "w:" + tok, 1.0)
            padded = f"<{tok}>"
            for n in (3, 4):
                for i in range(max(0, len(padded) - n + 1)):
                    self._add(v, "c:" + padded[i:i + n], self.ngram_weight)
        norm = math.sqrt(sum(x * x for x in v))
        return [x / norm for x in v] if norm else v

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

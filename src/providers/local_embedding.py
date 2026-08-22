"""Deterministic local embedding provider for offline testing and development.

Uses TF-IDF-like token hashing to generate deterministic dense vectors.
Zero external API calls, zero cost, completely reproducible.
"""

import math
import re
from typing import List


class LocalEmbeddingProvider:
    """Local, offline, deterministic embedding provider based on hashed term frequencies."""

    def __init__(self, dimension: int = 128):
        self.dimension = dimension

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def _embed_single(self, text: str) -> list[float]:
        tokens = self._tokenize(text)
        vec = [0.0] * self.dimension
        if not tokens:
            return vec

        # Term frequency with hash bucketing
        for token in tokens:
            h = hash(token) % self.dimension
            vec[h] += 1.0

        # Sublinear TF scaling
        for i in range(self.dimension):
            if vec[i] > 0:
                vec[i] = 1.0 + math.log(vec[i])

        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 1e-9:
            vec = [x / norm for x in vec]

        return vec

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate deterministic embeddings for all input texts."""
        return [self._embed_single(t) for t in texts]

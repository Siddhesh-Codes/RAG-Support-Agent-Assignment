"""Embedding index: build, cache, and query a local vector index.

Uses numpy for cosine similarity — appropriate for the ~50-80 chunk corpus.
Embedding computation is provider-abstracted and cached to disk.
"""

import json
import hashlib
import numpy as np
from pathlib import Path
from typing import Protocol

from src.rag.ingest import Chunk


class EmbeddingProvider(Protocol):
    """Abstract interface for embedding providers.
    Implementations must provide embed_texts().
    """
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts. Returns list of vectors."""
        ...


class VectorIndex:
    """Local numpy-based vector index with disk caching."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings: np.ndarray | None = None
        self.chunks: list[Chunk] = []

    def _cache_key(self, chunks: list[Chunk]) -> str:
        """Generate a deterministic cache key from chunk contents."""
        content = "".join(c.text for c in chunks)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _cache_paths(self, key: str) -> tuple[Path, Path]:
        """Return paths for embeddings and metadata cache files."""
        return (
            self.cache_dir / f"embeddings_{key}.npy",
            self.cache_dir / f"chunks_{key}.json",
        )

    def build(self, chunks: list[Chunk], provider: EmbeddingProvider) -> None:
        """Build the index from chunks, using cached embeddings if available."""
        self.chunks = chunks
        key = self._cache_key(chunks)
        emb_path, meta_path = self._cache_paths(key)

        if emb_path.exists() and meta_path.exists():
            # Load from cache
            self.embeddings = np.load(str(emb_path))
            return

        # Compute embeddings
        texts = [c.text for c in chunks]
        vectors = provider.embed_texts(texts)
        self.embeddings = np.array(vectors, dtype=np.float32)

        # Cache to disk
        np.save(str(emb_path), self.embeddings)
        # Save chunk metadata for verification
        meta = [{"source": c.source_file, "heading": c.heading, "idx": c.chunk_index}
                for c in chunks]
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def query(self, query_embedding: list[float], top_k: int = 8) -> list[tuple[Chunk, float]]:
        """Find top-k most similar chunks by cosine similarity.

        Returns list of (chunk, similarity_score) tuples, sorted by score descending.
        """
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        q = np.array(query_embedding, dtype=np.float32)

        # Cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(q)
        # Avoid division by zero
        norms = np.maximum(norms, 1e-10)
        similarities = self.embeddings @ q / norms

        # Get top-k indices
        k = min(top_k, len(similarities))
        top_indices = np.argsort(similarities)[-k:][::-1]

        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(similarities[idx])))

        return results

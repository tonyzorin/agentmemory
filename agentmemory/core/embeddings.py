"""
Embedding service using sentence-transformers (BAAI/bge-base-en-v1.5, 768-dim).

Features:
- Local CPU inference (no external API calls)
- Redis-backed embedding cache with TTL
- Batch encoding for efficiency
- Unit-normalized output for cosine similarity
- BGE query prefix for optimal retrieval quality

The model is loaded lazily on first use to avoid startup overhead when
the embeddings module is imported but not used (e.g., in CLI commands
that don't need embeddings).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from agentmemory.db.redis_client import MemoryRedisClient

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_DIM = 768

# BGE models require a query prefix for retrieval tasks to achieve optimal quality.
# Documents are stored as-is; only queries use this prefix.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbeddingService:
    """
    Sentence-transformers wrapper with Redis caching.

    Usage:
        svc = EmbeddingService()
        emb = svc.encode("Anton prefers Claude for coding")
        batch = svc.encode_batch(["text1", "text2"])
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        redis_url: str = "redis://localhost:6380/0",
        key_prefix: str = "",
        cache_ttl: int = 86400,
        embedding_dim: int = DEFAULT_DIM,
    ):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self._model: SentenceTransformer | None = None
        # Apply BGE query prefix only for BGE models
        self._use_query_prefix = "bge" in model_name.lower()

        self.redis_client = MemoryRedisClient(
            redis_url=redis_url,
            key_prefix=key_prefix,
            embedding_dim=embedding_dim,
            embedding_cache_ttl=cache_ttl,
        )

    @property
    def model(self) -> "SentenceTransformer":
        """Lazy-load the model on first access."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            logger.info("Model loaded successfully")
        return self._model

    def encode(self, text: str, use_cache: bool = True, is_query: bool = False) -> list[float]:
        """
        Encode a single text string to a normalized embedding vector.

        Args:
            text: The text to encode.
            use_cache: If True, check Redis cache first and store result.
            is_query: If True and using a BGE model, prepend the query prefix.

        Returns:
            List of floats (unit-normalized).
        """
        encode_text = (BGE_QUERY_PREFIX + text) if (is_query and self._use_query_prefix) else text

        if use_cache:
            cached = self.redis_client.get_cached_embedding(encode_text)
            if cached is not None:
                return cached

        embedding = self._encode_raw(encode_text)

        if use_cache:
            self.redis_client.cache_embedding(encode_text, embedding)

        return embedding

    def encode_batch(
        self, texts: list[str], use_cache: bool = True, is_query: bool = False
    ) -> list[list[float]]:
        """
        Encode multiple texts, using cache where available.

        Args:
            texts: List of texts to encode.
            use_cache: If True, use Redis cache for hits/misses.
            is_query: If True and using a BGE model, prepend the query prefix.

        Returns:
            List of embeddings in the same order as input texts.
        """
        if not texts:
            return []

        encode_texts = [
            (BGE_QUERY_PREFIX + t) if (is_query and self._use_query_prefix) else t
            for t in texts
        ]

        results: list[list[float] | None] = [None] * len(encode_texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        if use_cache:
            for i, text in enumerate(encode_texts):
                cached = self.redis_client.get_cached_embedding(text)
                if cached is not None:
                    results[i] = cached
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)
        else:
            uncached_indices = list(range(len(encode_texts)))
            uncached_texts = encode_texts

        if uncached_texts:
            new_embeddings = self._encode_batch_raw(uncached_texts)
            for idx, emb in zip(uncached_indices, new_embeddings):
                results[idx] = emb
                if use_cache:
                    self.redis_client.cache_embedding(encode_texts[idx], emb)

        return [r for r in results if r is not None]

    def _encode_raw(self, text: str) -> list[float]:
        """Run model inference for a single text."""
        import numpy as np
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if hasattr(embedding, "tolist"):
            return [float(x) for x in embedding.tolist()]
        return [float(x) for x in embedding]

    def _encode_batch_raw(self, texts: list[str]) -> list[list[float]]:
        """Run model inference for a batch of texts."""
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        result = []
        for emb in embeddings:
            if hasattr(emb, "tolist"):
                result.append([float(x) for x in emb.tolist()])
            else:
                result.append([float(x) for x in emb])
        return result

    def close(self) -> None:
        """Release resources."""
        self.redis_client.close()
        self._model = None

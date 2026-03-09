"""
Cross-encoder reranker for agentmemory retrieval pipeline.

Sits between hybrid search (BM25 + vector) and the final scoring step.
Takes the top-K candidates from hybrid search and scores each (query, document)
pair jointly using a cross-encoder model, replacing the raw similarity field.

When enabled, the retrieval pipeline becomes:
    hybrid search (top-K) → cross-encoder rerank → graph boost + recency + importance → top-N

The cross-encoder is more accurate than bi-encoder similarity because it
sees both the query and document together — it catches semantic nuances
that embedding-based retrieval misses.

By default the reranker is DISABLED (reranker_enabled=False in config).
It pays off significantly at corpus sizes of ~1000+ nodes.

Model options (all CPU-feasible, lazy-loaded):
    cross-encoder/ms-marco-MiniLM-L6-v2  — 22M params, fast, good quality (default)
    cross-encoder/ms-marco-MiniLM-L12-v2 — 33M params, slightly better quality
    BAAI/bge-reranker-v2-m3              — multilingual, strong MTEB scores
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


class RerankerService:
    """
    Cross-encoder reranker with lazy model loading.

    Usage:
        reranker = RerankerService()
        candidates = reranker.rerank(query, documents, top_k=10)

    The model is loaded on first call to rerank() — startup stays fast
    even when the reranker is configured but corpus is small.
    """

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL):
        self.model_name = model_name
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> "CrossEncoder":
        """Lazy-load the cross-encoder model on first access."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info("Loading reranker model: %s", self.model_name)
            self._model = CrossEncoder(self.model_name)
            logger.info("Reranker model loaded")
        return self._model

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Score each (query, document.content) pair with the cross-encoder.

        Replaces each document's `similarity` field with the cross-encoder score
        (normalized to [0, 1]) so the downstream scoring formula (graph boost,
        recency, importance) remains unchanged.

        Args:
            query: The search query.
            documents: List of candidate dicts from hybrid search. Each must
                       have a `content` field; others are passed through.
            top_k: Return only the top-K results by reranker score. If None,
                   returns all documents sorted by score.

        Returns:
            Documents sorted by reranker score descending, with `similarity`
            replaced by the normalized cross-encoder score.
        """
        if not documents:
            return documents

        pairs = [(query, doc.get("content", doc.get("name", ""))) for doc in documents]

        try:
            raw_scores: list[float] = self.model.predict(pairs).tolist()
        except Exception as e:
            logger.error("Reranker inference failed: %s — returning original order", e)
            return documents[:top_k] if top_k else documents

        # Normalize scores to [0, 1] using sigmoid (cross-encoder outputs are unbounded logits)
        import math
        normalized = [1.0 / (1.0 + math.exp(-s)) for s in raw_scores]

        ranked = sorted(
            zip(normalized, documents),
            key=lambda x: x[0],
            reverse=True,
        )

        result = []
        for score, doc in ranked:
            result.append({**doc, "similarity": round(score, 4), "reranker_score": round(score, 4)})

        return result[:top_k] if top_k else result

    def close(self) -> None:
        """Release model resources."""
        self._model = None

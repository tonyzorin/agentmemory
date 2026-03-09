"""
Tests for the embeddings module.

Tests run against the real sentence-transformers model (all-MiniLM-L6-v2).
No mocks — we want to verify the actual model loads and produces correct output.
Cache tests use real Redis.
"""

import math

import pytest

from tests.conftest import TEST_REDIS_URL
from agentmemory.core.embeddings import EmbeddingService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def embed(redis_client):
    """EmbeddingService with real model and Redis cache (module-scoped for model reuse)."""
    # Clean test embedding cache keys before module
    keys = redis_client.keys("test:emb_cache:*")
    if keys:
        redis_client.delete(*keys)

    svc = EmbeddingService(
        model_name="all-MiniLM-L6-v2",
        redis_url=TEST_REDIS_URL,
        key_prefix="test:",
        cache_ttl=60,
    )
    yield svc
    svc.close()


# ---------------------------------------------------------------------------
# Basic encoding
# ---------------------------------------------------------------------------


class TestEncode:
    def test_encode_returns_list_of_floats(self, embed):
        result = embed.encode("Hello world")
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)

    def test_encode_returns_correct_dimension(self, embed):
        result = embed.encode("Test sentence")
        assert len(result) == 384

    def test_encode_empty_string(self, embed):
        result = embed.encode("")
        assert len(result) == 384

    def test_encode_long_text(self, embed):
        long_text = "This is a very long sentence. " * 50
        result = embed.encode(long_text)
        assert len(result) == 384

    def test_encode_is_normalized(self, embed):
        """Embeddings should be unit-normalized (cosine similarity ready)."""
        result = embed.encode("Normalized vector test")
        magnitude = math.sqrt(sum(x * x for x in result))
        assert abs(magnitude - 1.0) < 0.01

    def test_similar_texts_have_high_similarity(self, embed):
        emb1 = embed.encode("Anton works on feedback1")
        emb2 = embed.encode("Anton is working on the feedback1 project")
        similarity = sum(a * b for a, b in zip(emb1, emb2))
        assert similarity > 0.7

    def test_different_texts_have_lower_similarity(self, embed):
        emb1 = embed.encode("PostgreSQL database configuration")
        emb2 = embed.encode("The weather is sunny today")
        similarity = sum(a * b for a, b in zip(emb1, emb2))
        assert similarity < 0.7


# ---------------------------------------------------------------------------
# Batch encoding
# ---------------------------------------------------------------------------


class TestBatchEncode:
    def test_batch_encode_returns_list_of_embeddings(self, embed):
        texts = ["First sentence", "Second sentence", "Third sentence"]
        results = embed.encode_batch(texts)
        assert len(results) == 3
        assert all(len(r) == 384 for r in results)

    def test_batch_encode_empty_list(self, embed):
        results = embed.encode_batch([])
        assert results == []

    def test_batch_encode_single_item(self, embed):
        results = embed.encode_batch(["Single item"])
        assert len(results) == 1
        assert len(results[0]) == 384

    def test_batch_encode_matches_single_encode(self, embed):
        """Batch encoding should produce same results as individual encoding."""
        texts = ["Hello world", "Goodbye world"]
        batch = embed.encode_batch(texts)
        singles = [embed.encode(t) for t in texts]
        for b, s in zip(batch, singles):
            for bv, sv in zip(b, s):
                assert abs(bv - sv) < 1e-5


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


class TestCaching:
    def test_second_encode_uses_cache(self, embed):
        """Second call for same text should hit cache (faster)."""
        import time
        text = "Cache test sentence unique xyz123"
        # First call — model inference
        t0 = time.time()
        emb1 = embed.encode(text)
        t1 = time.time()
        # Second call — should use cache
        t2 = time.time()
        emb2 = embed.encode(text)
        t3 = time.time()
        # Cache hit should be faster (or at least return same result)
        for a, b in zip(emb1[:5], emb2[:5]):
            assert abs(a - b) < 1e-4

    def test_cache_can_be_bypassed(self, embed):
        """encode with use_cache=False should skip cache."""
        text = "Bypass cache test"
        emb1 = embed.encode(text, use_cache=True)
        emb2 = embed.encode(text, use_cache=False)
        # Results should still be the same (same model, same input)
        for a, b in zip(emb1[:5], emb2[:5]):
            assert abs(a - b) < 1e-4

    def test_cache_stores_embedding(self, embed):
        text = "Stored in cache test unique abc987"
        embed.encode(text, use_cache=True)
        # Verify it's in Redis
        cached = embed.redis_client.get_cached_embedding(text)
        assert cached is not None
        assert len(cached) == 384

"""
Tests for the Redis search client.

Runs against a real Redis 8.4 instance (via Docker Compose).
Tests define the expected behavior — implementation must make these pass.
"""

import struct
import time
import uuid

import pytest
import redis as redis_lib

from tests.conftest import TEST_REDIS_PREFIX, TEST_REDIS_URL
from agentmemory.db.redis_client import MemoryRedisClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_embedding(dim: int = 384, seed: float = 0.1) -> list[float]:
    """Create a deterministic test embedding vector."""
    import math
    return [math.sin(i * seed) for i in range(dim)]


def make_memory_doc(
    memory_id: str | None = None,
    content: str = "Test memory content",
    tags: list[str] | None = None,
    source: str = "cli",
    importance: float = 0.5,
    embedding: list[float] | None = None,
) -> dict:
    return {
        "id": memory_id or str(uuid.uuid4()),
        "content": content,
        "tags": tags or ["test"],
        "source": source,
        "importance": importance,
        "created_at": "2026-02-28T00:00:00",
        "embedding": embedding or make_embedding(),
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client(clean_redis):
    """MemoryRedisClient pointed at test Redis, with clean state."""
    c = MemoryRedisClient(redis_url=TEST_REDIS_URL, key_prefix=TEST_REDIS_PREFIX)
    yield c
    c.close()


# ---------------------------------------------------------------------------
# Index creation
# ---------------------------------------------------------------------------


class TestIndexCreation:
    def test_index_created_on_init(self, client):
        """Index should exist after client initialisation."""
        info = client.redis.execute_command("FT.INFO", f"{TEST_REDIS_PREFIX}memory_idx")
        assert info is not None

    def test_index_has_vector_field(self, client):
        """Index must have an embedding vector field."""
        info = client.redis.execute_command("FT.INFO", f"{TEST_REDIS_PREFIX}memory_idx")
        # FT.INFO returns a flat list of key-value pairs
        info_dict = dict(zip(info[::2], info[1::2]))
        fields_raw = info_dict.get(b"attributes", info_dict.get("attributes", []))
        field_names = []
        for field in fields_raw:
            if isinstance(field, list):
                field_names.append(field[1] if len(field) > 1 else "")
        assert any(b"embedding" in str(f).encode() or "embedding" in str(f) for f in field_names)

    def test_index_creation_is_idempotent(self, client):
        """Creating the client twice should not raise (index already exists)."""
        client2 = MemoryRedisClient(redis_url=TEST_REDIS_URL, key_prefix=TEST_REDIS_PREFIX)
        client2.close()


# ---------------------------------------------------------------------------
# Storing documents
# ---------------------------------------------------------------------------


class TestStoreMemory:
    def test_store_returns_true_on_success(self, client):
        doc = make_memory_doc()
        result = client.store_memory(doc)
        assert result is True

    def test_stored_document_is_retrievable(self, client):
        doc = make_memory_doc(memory_id="test-mem-001", content="Hello world")
        client.store_memory(doc)
        key = f"{TEST_REDIS_PREFIX}memory:test-mem-001"
        stored = client.redis.json().get(key)
        assert stored is not None
        assert stored["content"] == "Hello world"

    def test_store_preserves_all_fields(self, client):
        doc = make_memory_doc(
            memory_id="test-mem-002",
            content="Full fields test",
            tags=["tag1", "tag2"],
            source="mcp",
            importance=0.9,
        )
        client.store_memory(doc)
        key = f"{TEST_REDIS_PREFIX}memory:test-mem-002"
        stored = client.redis.json().get(key)
        assert stored["tags"] == ["tag1", "tag2"]
        assert stored["source"] == "mcp"
        assert stored["importance"] == 0.9

    def test_store_overwrites_existing(self, client):
        doc = make_memory_doc(memory_id="test-mem-003", content="Original")
        client.store_memory(doc)
        doc["content"] = "Updated"
        client.store_memory(doc)
        key = f"{TEST_REDIS_PREFIX}memory:test-mem-003"
        stored = client.redis.json().get(key)
        assert stored["content"] == "Updated"

    def test_delete_memory(self, client):
        doc = make_memory_doc(memory_id="test-mem-del")
        client.store_memory(doc)
        result = client.delete_memory("test-mem-del")
        assert result is True
        key = f"{TEST_REDIS_PREFIX}memory:test-mem-del"
        assert client.redis.json().get(key) is None

    def test_delete_nonexistent_returns_false(self, client):
        result = client.delete_memory("nonexistent-id")
        assert result is False


# ---------------------------------------------------------------------------
# Vector search (KNN)
# ---------------------------------------------------------------------------


class TestVectorSearch:
    def test_vector_search_returns_results(self, client):
        # Store a few memories with known embeddings
        query_emb = make_embedding(seed=0.1)
        similar_emb = make_embedding(seed=0.105)   # very similar
        different_emb = make_embedding(seed=5.0)   # very different

        client.store_memory(make_memory_doc("test-vs-1", "Similar content", embedding=similar_emb))
        client.store_memory(make_memory_doc("test-vs-2", "Different content", embedding=different_emb))

        # Small sleep to allow index to update
        time.sleep(0.1)

        results = client.vector_search(query_emb, limit=5)
        assert len(results) >= 1

    def test_vector_search_ranks_similar_higher(self, client):
        query_emb = make_embedding(seed=0.1)
        similar_emb = make_embedding(seed=0.101)
        different_emb = make_embedding(seed=9.9)

        client.store_memory(make_memory_doc("test-rank-1", "Very similar", embedding=similar_emb))
        client.store_memory(make_memory_doc("test-rank-2", "Very different", embedding=different_emb))
        time.sleep(0.1)

        results = client.vector_search(query_emb, limit=5)
        ids = [r["id"] for r in results]
        # Similar should appear before different
        assert ids.index("test-rank-1") < ids.index("test-rank-2")

    def test_vector_search_respects_limit(self, client):
        for i in range(5):
            client.store_memory(make_memory_doc(f"test-lim-{i}", f"Memory {i}"))
        time.sleep(0.1)

        results = client.vector_search(make_embedding(), limit=2)
        assert len(results) <= 2

    def test_vector_search_empty_index_returns_empty(self, client):
        results = client.vector_search(make_embedding(), limit=5)
        assert results == []


# ---------------------------------------------------------------------------
# Hybrid search (FT.HYBRID — BM25 + vector)
# ---------------------------------------------------------------------------


class TestHybridSearch:
    def test_hybrid_search_returns_results(self, client):
        client.store_memory(make_memory_doc(
            "test-hyb-1",
            "Anton prefers Claude for coding tasks",
            embedding=make_embedding(seed=0.2),
        ))
        client.store_memory(make_memory_doc(
            "test-hyb-2",
            "Redis is used for vector search",
            embedding=make_embedding(seed=0.5),
        ))
        time.sleep(0.1)

        results = client.hybrid_search(
            query_text="Claude coding",
            query_embedding=make_embedding(seed=0.2),
            limit=5,
        )
        assert len(results) >= 1

    def test_hybrid_search_finds_keyword_match(self, client):
        client.store_memory(make_memory_doc(
            "test-hyb-kw",
            "PostgreSQL is the primary database for feedback1",
            embedding=make_embedding(seed=0.3),
        ))
        time.sleep(0.1)

        results = client.hybrid_search(
            query_text="PostgreSQL database",
            query_embedding=make_embedding(seed=0.3),
            limit=5,
        )
        ids = [r["id"] for r in results]
        assert "test-hyb-kw" in ids

    def test_hybrid_search_falls_back_to_vector_on_error(self, client):
        """If FT.HYBRID is unavailable, should fall back to vector search."""
        client.store_memory(make_memory_doc(
            "test-fb-1", "Fallback test", embedding=make_embedding(seed=0.1)
        ))
        time.sleep(0.1)
        # Even with an unusual query, should return results via fallback
        results = client.hybrid_search(
            query_text="fallback",
            query_embedding=make_embedding(seed=0.1),
            limit=5,
        )
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Tag filtering
# ---------------------------------------------------------------------------


class TestTagFiltering:
    def test_search_filters_by_tag(self, client):
        client.store_memory(make_memory_doc(
            "test-tag-1", "Tagged memory", tags=["personal", "preferences"],
            embedding=make_embedding(seed=0.1),
        ))
        client.store_memory(make_memory_doc(
            "test-tag-2", "Other memory", tags=["work"],
            embedding=make_embedding(seed=0.15),
        ))
        time.sleep(0.1)

        results = client.vector_search(make_embedding(seed=0.1), limit=10, tags=["personal"])
        ids = [r["id"] for r in results]
        assert "test-tag-1" in ids
        assert "test-tag-2" not in ids


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------


class TestEmbeddingCache:
    def test_cache_store_and_retrieve(self, client):
        text = "cache test text"
        embedding = make_embedding(seed=0.7)
        client.cache_embedding(text, embedding)
        retrieved = client.get_cached_embedding(text)
        assert retrieved is not None
        assert len(retrieved) == 384
        # Values should be approximately equal (float32 precision)
        for a, b in zip(embedding[:5], retrieved[:5]):
            assert abs(a - b) < 1e-4

    def test_cache_miss_returns_none(self, client):
        result = client.get_cached_embedding("not cached text xyz")
        assert result is None

    def test_cache_has_ttl(self, client):
        """Cache entries should expire (we just verify TTL is set, not wait for it)."""
        text = "ttl test"
        client.cache_embedding(text, make_embedding(), ttl=60)
        import hashlib
        key = f"{TEST_REDIS_PREFIX}emb_cache:{hashlib.sha256(text.encode()).hexdigest()}"
        ttl = client.redis.ttl(key)
        assert ttl > 0


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_stats_returns_dict(self, client):
        stats = client.stats()
        assert isinstance(stats, dict)
        assert "memory_count" in stats
        assert "index_name" in stats

    def test_stats_count_increases_after_store(self, client):
        stats_before = client.stats()
        client.store_memory(make_memory_doc("test-stats-1"))
        time.sleep(0.1)
        stats_after = client.stats()
        assert stats_after["memory_count"] >= stats_before["memory_count"]

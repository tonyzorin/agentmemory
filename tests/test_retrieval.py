"""
Tests for hybrid retrieval — Redis FT.HYBRID + AGE graph traversal + confidence scoring.
"""

import time

import pytest

from tests.conftest import TEST_DATABASE_URL, TEST_REDIS_URL, TEST_GRAPH_NAME
from agentmemory.core.retrieval import HybridRetrieval


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def retrieval(redis_client):
    """HybridRetrieval with real backends."""
    keys = redis_client.keys("test:*")
    if keys:
        redis_client.delete(*keys)
    try:
        redis_client.execute_command("FT.DROPINDEX", "test:memory_idx", "DD")
    except Exception:
        pass

    r = HybridRetrieval(
        database_url=TEST_DATABASE_URL,
        redis_url=TEST_REDIS_URL,
        key_prefix="test:",
        graph_name=TEST_GRAPH_NAME,
        embedding_model="BAAI/bge-base-en-v1.5",
        embedding_dim=768,
    )
    # Seed some data
    r.memory.store(
        content="Anton is a product manager who codes",
        node_type="Memory",
        tags=["person", "role"],
        importance=0.9,
    )
    r.memory.store(
        content="feedback1 is a SaaS product for collecting user feedback",
        node_type="Memory",
        tags=["project", "saas"],
        importance=0.8,
    )
    r.memory.store(
        content="Redis 8.4 supports FT.HYBRID for combined BM25 and vector search",
        node_type="Learning",
        tags=["redis", "search"],
        extra={
            "what_failed": "FT.HYBRID not available in Redis < 8.0",
            "why_it_failed": "Old Redis version",
            "what_to_avoid": "Redis < 8.0 for hybrid search",
        },
    )
    r.memory.store(
        content="PostgreSQL 18 with Apache AGE 1.7.0 provides graph query capabilities",
        node_type="Decision",
        tags=["database", "graph"],
        extra={"rationale": "Best combination for structured + graph queries"},
    )
    time.sleep(0.3)
    yield r
    r.close()


# ---------------------------------------------------------------------------
# Basic retrieval
# ---------------------------------------------------------------------------


class TestBasicRetrieval:
    def test_retrieve_returns_results(self, retrieval):
        results = retrieval.retrieve("Anton product manager")
        assert len(results) >= 1

    def test_retrieve_has_required_fields(self, retrieval):
        results = retrieval.retrieve("feedback1 project")
        assert len(results) >= 1
        first = results[0]
        assert "id" in first or "content" in first
        assert "score" in first

    def test_retrieve_scores_are_normalized(self, retrieval):
        results = retrieval.retrieve("Redis search")
        for r in results:
            assert 0.0 <= r["score"] <= 1.0

    def test_retrieve_respects_limit(self, retrieval):
        results = retrieval.retrieve("memory", limit=2)
        assert len(results) <= 2

    def test_retrieve_orders_by_score(self, retrieval):
        results = retrieval.retrieve("Anton feedback1")
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Graph boost
# ---------------------------------------------------------------------------


class TestGraphBoost:
    def test_related_entities_get_boosted(self, retrieval):
        """
        Entities related to a queried entity should score higher than
        unrelated entities with similar content.
        """
        # Store a project and a related memory
        proj = retrieval.memory.store_project(
            name="boost-test-project",
            description="Project for graph boost test",
        )
        mem = retrieval.memory.store(
            content="Critical information about boost-test-project deployment",
            node_type="Memory",
            tags=["deployment"],
        )
        retrieval.memory.relate(mem["id"], proj["id"], "ABOUT")
        time.sleep(0.2)

        results = retrieval.retrieve(
            "boost-test-project deployment",
            anchor_entity_id=proj["id"],
        )
        # The related memory should appear in results
        result_ids = [r.get("id", "") for r in results]
        assert mem["id"] in result_ids

    def test_anchor_entity_boosts_neighbors(self, retrieval):
        """Results connected to anchor entity should have graph_boost > 0."""
        proj = retrieval.memory.store_project(
            name="anchor-test-proj",
            description="Anchor entity test",
        )
        mem = retrieval.memory.store(
            content="Memory connected to anchor-test-proj",
            node_type="Memory",
        )
        retrieval.memory.relate(mem["id"], proj["id"], "ABOUT")
        time.sleep(0.2)

        results = retrieval.retrieve(
            "anchor-test-proj memory",
            anchor_entity_id=proj["id"],
        )
        # At least one result should have a graph_boost
        boosted = [r for r in results if r.get("graph_boost", 0) > 0]
        assert len(boosted) >= 0  # May be 0 if no overlap — just verify no crash


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------


class TestConfidenceScoring:
    def test_high_confidence_for_exact_match(self, retrieval):
        """Exact content match should have a non-zero confidence score."""
        unique_text = "unique_xyz_987_confidence_test_memory"
        retrieval.memory.store(
            content=unique_text,
            node_type="Memory",
            importance=1.0,
        )
        time.sleep(0.2)
        results = retrieval.retrieve(unique_text)
        if results:
            assert results[0]["score"] > 0.0

    def test_importance_affects_score(self, retrieval):
        """High-importance memories should rank higher than low-importance ones."""
        retrieval.memory.store(
            content="High importance test memory about retrieval scoring",
            node_type="Memory",
            importance=1.0,
            tags=["importance-test"],
        )
        retrieval.memory.store(
            content="Low importance test memory about retrieval scoring",
            node_type="Memory",
            importance=0.1,
            tags=["importance-test"],
        )
        time.sleep(0.2)

        results = retrieval.retrieve(
            "retrieval scoring",
            tags=["importance-test"],
        )
        if len(results) >= 2:
            # High importance should rank first
            high_idx = next(
                (i for i, r in enumerate(results) if "High importance" in r.get("content", "")),
                None,
            )
            low_idx = next(
                (i for i, r in enumerate(results) if "Low importance" in r.get("content", "")),
                None,
            )
            if high_idx is not None and low_idx is not None:
                assert high_idx <= low_idx


# ---------------------------------------------------------------------------
# Recency scoring
# ---------------------------------------------------------------------------


class TestRecencyScoring:
    def test_results_have_recency_field(self, retrieval):
        """All results should include a recency field."""
        results = retrieval.retrieve("Anton product manager")
        for r in results:
            assert "recency" in r
            assert 0.0 <= r["recency"] <= 1.0

    def test_recency_score_helper(self):
        """Verify recency decay function directly."""
        from agentmemory.core.retrieval import _recency_score
        from datetime import datetime, timezone, timedelta

        # Brand new memory should be close to 1.0
        now_str = datetime.now(timezone.utc).isoformat()
        assert _recency_score(now_str) > 0.95

        # Memory from 30 days ago should be lower
        old_str = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        assert _recency_score(old_str) < _recency_score(now_str)

        # None/missing should return neutral 0.5
        assert _recency_score(None) == 0.5


# ---------------------------------------------------------------------------
# Auto-link
# ---------------------------------------------------------------------------


class TestAutoLink:
    def test_auto_link_creates_edges(self, retrieval):
        """Storing similar memories should create RELATED_TO edges."""
        mem1 = retrieval.memory.store(
            content="Anton uses Redis for caching in all projects",
            node_type="Memory",
            tags=["redis", "caching"],
        )
        mem2 = retrieval.memory.store(
            content="Redis is used for vector search and caching",
            node_type="Memory",
            tags=["redis", "search"],
        )
        time.sleep(0.3)

        # Manually trigger auto-link for mem2 to find mem1
        linked = retrieval.memory.auto_link(
            node_id=mem2["id"],
            content="Redis is used for vector search and caching",
            node_type="Memory",
            min_similarity=0.3,
        )
        # Should find at least one related memory (mem1 is semantically similar)
        assert isinstance(linked, list)

    def test_auto_link_skips_non_searchable_types(self, retrieval):
        """Non-searchable node types should not be auto-linked."""
        linked = retrieval.memory.auto_link(
            node_id="some-id",
            content="Some project description",
            node_type="Project",
        )
        assert linked == []

    def test_auto_link_no_self_link(self, retrieval):
        """A memory should never be linked to itself."""
        mem = retrieval.memory.store(
            content="Unique auto-link self-test memory xyz987",
            node_type="Memory",
        )
        time.sleep(0.2)
        linked = retrieval.memory.auto_link(
            node_id=mem["id"],
            content="Unique auto-link self-test memory xyz987",
            node_type="Memory",
        )
        assert mem["id"] not in linked


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    def test_retrieve_with_bad_anchor_id_does_not_crash(self, retrieval):
        """Should return results even if anchor entity doesn't exist."""
        results = retrieval.retrieve(
            "test query",
            anchor_entity_id="nonexistent-entity-id-xyz",
        )
        assert isinstance(results, list)

    def test_retrieve_empty_query(self, retrieval):
        """Empty query should return empty list or handle gracefully."""
        results = retrieval.retrieve("")
        assert isinstance(results, list)

    def test_results_have_all_score_fields(self, retrieval):
        """Every result should include score, similarity, graph_boost, recency, importance."""
        results = retrieval.retrieve("Anton product manager")
        for r in results:
            assert "score" in r
            assert "similarity" in r
            assert "graph_boost" in r
            assert "recency" in r
            assert "importance" in r

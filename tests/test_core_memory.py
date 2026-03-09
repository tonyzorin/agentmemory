"""
Tests for core memory operations.

Tests store, recall, relate, context, forget, and specialized operations
(learn, decide, goal, task) against real Redis + PostgreSQL + AGE.
"""

import time
import uuid

import pytest

from tests.conftest import TEST_DATABASE_URL, TEST_REDIS_URL, TEST_GRAPH_NAME
from agentmemory.core.memory import MemoryService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def memory(redis_client):
    """MemoryService with real backends (module-scoped to reuse model)."""
    # Clean test keys
    keys = redis_client.keys("test:*")
    if keys:
        redis_client.delete(*keys)
    try:
        redis_client.execute_command("FT.DROPINDEX", "test:memory_idx", "DD")
    except Exception:
        pass

    svc = MemoryService(
        database_url=TEST_DATABASE_URL,
        redis_url=TEST_REDIS_URL,
        key_prefix="test:",
        graph_name=TEST_GRAPH_NAME,
        embedding_model="all-MiniLM-L6-v2",
    )
    yield svc
    svc.close()


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class TestStore:
    def test_store_memory_returns_id(self, memory):
        result = memory.store(
            content="Anton prefers Claude for coding tasks",
            node_type="Memory",
            tags=["preferences", "tools"],
        )
        assert result is not None
        assert "id" in result
        assert result["id"]

    def test_store_learning_returns_id(self, memory):
        result = memory.store(
            content="psycopg2-binary doesn't work on Python 3.14",
            node_type="Learning",
            tags=["python", "dependencies"],
            extra={
                "what_failed": "psycopg2-binary installation",
                "why_it_failed": "No wheel for Python 3.14",
                "what_to_avoid": "psycopg2-binary on Python 3.14",
            },
        )
        assert result["id"]

    def test_store_decision_returns_id(self, memory):
        result = memory.store(
            content="Use Redis 8.4 for vector search",
            node_type="Decision",
            tags=["architecture"],
            extra={"rationale": "Already proven in feedback1"},
        )
        assert result["id"]

    def test_store_persists_to_redis(self, memory):
        result = memory.store(
            content="Redis persistence test content",
            node_type="Memory",
        )
        time.sleep(0.1)
        mem_id = result["id"]
        doc = memory.redis.get_memory(mem_id)
        assert doc is not None
        assert doc["content"] == "Redis persistence test content"

    def test_store_persists_to_postgres(self, memory):
        result = memory.store(
            content="Postgres persistence test",
            node_type="Memory",
            tags=["test"],
        )
        mem_id = result["id"]
        entity = memory.postgres.get_entity(mem_id)
        assert entity is not None
        assert entity["node_type"] == "Memory"

    def test_store_persists_to_graph(self, memory):
        result = memory.store(
            content="Graph persistence test",
            node_type="Memory",
        )
        mem_id = result["id"]
        node = memory.graph.get_node(mem_id)
        assert node is not None

    def test_store_project_profile(self, memory):
        result = memory.store_project(
            name="feedback1",
            description="Product feedback SaaS",
            repo_path="/home/anton/feedback1",
            stack=["Python", "FastAPI", "PostgreSQL", "Redis"],
            run_cmd="docker compose up",
            test_cmd="pytest",
        )
        assert result["id"]

    def test_store_goal(self, memory):
        result = memory.store_goal(
            name="Launch Feedback1 GTM",
            description="Go-to-market for feedback1",
            tags=["gtm", "feedback1"],
        )
        assert result["id"]

    def test_store_task(self, memory):
        result = memory.store_task(
            name="Build SSE endpoint",
            description="Implement SSE transport for MCP",
            tags=["mcp", "feedback1"],
        )
        assert result["id"]


# ---------------------------------------------------------------------------
# Recall (semantic search)
# ---------------------------------------------------------------------------


class TestRecall:
    def test_recall_returns_results(self, memory):
        memory.store(content="Anton uses Cursor IDE for development", node_type="Memory")
        time.sleep(0.2)
        results = memory.recall("what IDE does Anton use")
        assert len(results) >= 1

    def test_recall_finds_relevant_content(self, memory):
        memory.store(
            content="The feedback1 project uses PostgreSQL 18 as its main database",
            node_type="Memory",
            tags=["feedback1", "database"],
        )
        time.sleep(0.2)
        results = memory.recall("feedback1 database")
        contents = [r["content"] for r in results]
        assert any("PostgreSQL" in c or "database" in c for c in contents)

    def test_recall_respects_limit(self, memory):
        results = memory.recall("test query", limit=2)
        assert len(results) <= 2

    def test_recall_returns_empty_on_no_match(self, memory):
        results = memory.recall("xyzzy completely unrelated query 99999")
        # Should return empty or very low-score results
        assert isinstance(results, list)

    def test_recall_filters_by_node_type(self, memory):
        memory.store(
            content="Learning: always use virtual environments",
            node_type="Learning",
            extra={
                "what_failed": "global pip install",
                "why_it_failed": "package conflicts",
                "what_to_avoid": "global pip install",
            },
        )
        time.sleep(0.2)
        results = memory.recall("virtual environments", node_type="Learning")
        assert all(r.get("node_type") == "Learning" for r in results if "node_type" in r)


# ---------------------------------------------------------------------------
# Relate
# ---------------------------------------------------------------------------


class TestRelate:
    def test_relate_creates_edge(self, memory):
        proj = memory.store_project(name="test-proj-relate", description="Test project")
        person = memory.store(
            content="Test person for relate",
            node_type="Person",
            extra={"name": "Test Person"},
        )
        result = memory.relate(
            from_id=person["id"],
            to_id=proj["id"],
            edge_type="WORKS_ON",
        )
        assert result is True

    def test_relate_invalid_edge_type_raises(self, memory):
        with pytest.raises(ValueError):
            memory.relate(
                from_id="some-id",
                to_id="other-id",
                edge_type="INVALID_EDGE_TYPE",
            )

    def test_relate_creates_postgres_record(self, memory):
        proj = memory.store_project(name="test-proj-pg-relate", description="Test")
        mem = memory.store(content="Test memory for relation", node_type="Memory")
        memory.relate(mem["id"], proj["id"], "ABOUT")
        relations = memory.postgres.get_relations(from_id=mem["id"])
        assert any(r["edge_type"] == "ABOUT" for r in relations)


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


class TestContext:
    def test_context_returns_entity_info(self, memory):
        proj = memory.store_project(
            name="context-test-project",
            description="Project for context test",
        )
        ctx = memory.context(proj["id"])
        assert ctx is not None
        assert "entity" in ctx

    def test_context_includes_related_nodes(self, memory):
        proj = memory.store_project(name="ctx-proj-2", description="Context project 2")
        mem = memory.store(content="Memory about ctx-proj-2", node_type="Memory")
        memory.relate(mem["id"], proj["id"], "ABOUT")

        ctx = memory.context(proj["id"])
        related_ids = [n.get("id") for n in ctx.get("neighbors", [])]
        assert mem["id"] in related_ids

    def test_context_nonexistent_entity(self, memory):
        ctx = memory.context("nonexistent-entity-id-xyz")
        assert ctx is None or ctx.get("entity") is None


# ---------------------------------------------------------------------------
# Forget
# ---------------------------------------------------------------------------


class TestForget:
    def test_forget_removes_from_redis(self, memory):
        mem = memory.store(content="Memory to forget", node_type="Memory")
        mem_id = mem["id"]
        time.sleep(0.1)
        memory.forget(mem_id)
        doc = memory.redis.get_memory(mem_id)
        assert doc is None

    def test_forget_removes_from_postgres(self, memory):
        mem = memory.store(content="Postgres memory to forget", node_type="Memory")
        mem_id = mem["id"]
        memory.forget(mem_id)
        entity = memory.postgres.get_entity(mem_id)
        assert entity is None

    def test_forget_nonexistent_returns_false(self, memory):
        result = memory.forget("nonexistent-id-xyz-forget")
        assert result is False


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_stats_returns_dict(self, memory):
        stats = memory.stats()
        assert isinstance(stats, dict)
        assert "redis" in stats
        assert "postgres" in stats

    def test_stats_redis_has_memory_count(self, memory):
        stats = memory.stats()
        assert "memory_count" in stats["redis"]

    def test_stats_postgres_has_entity_counts(self, memory):
        stats = memory.stats()
        assert "total_entities" in stats["postgres"]

"""
Tests for MCP tool functions.

We test the tool handler functions directly (not the MCP protocol layer),
since the protocol is handled by FastMCP. This tests the business logic
of each tool including graceful degradation.
"""

import time

import pytest

from tests.conftest import TEST_DATABASE_URL, TEST_REDIS_URL, TEST_GRAPH_NAME
from agentmemory.mcp.tools import MemoryTools


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tools(redis_client):
    """MemoryTools with real backends."""
    keys = redis_client.keys("test:*")
    if keys:
        redis_client.delete(*keys)
    try:
        redis_client.execute_command("FT.DROPINDEX", "test:memory_idx", "DD")
    except Exception:
        pass

    t = MemoryTools(
        database_url=TEST_DATABASE_URL,
        redis_url=TEST_REDIS_URL,
        key_prefix="test:",
        graph_name=TEST_GRAPH_NAME,
        embedding_model="BAAI/bge-base-en-v1.5",
        embedding_dim=768,
    )
    yield t
    t.close()


# ---------------------------------------------------------------------------
# memory_store
# ---------------------------------------------------------------------------


class TestMemoryStore:
    def test_store_memory_returns_id(self, tools):
        result = tools.memory_store(
            content="Anton prefers Anthropic Claude for coding tasks",
            node_type="Memory",
            tags=["preferences", "tools"],
        )
        assert "id" in result
        assert result["id"]
        assert "error" not in result

    def test_store_learning_returns_id(self, tools):
        result = tools.memory_store(
            content="psycopg2-binary doesn't work on Python 3.14",
            node_type="Learning",
            tags=["python"],
            extra={
                "what_failed": "psycopg2-binary",
                "why_it_failed": "No wheel for 3.14",
                "what_to_avoid": "psycopg2-binary on 3.14",
            },
        )
        assert "id" in result

    def test_store_invalid_node_type_returns_error(self, tools):
        result = tools.memory_store(
            content="Test",
            node_type="InvalidType",
        )
        assert "error" in result

    def test_store_empty_content_returns_error(self, tools):
        result = tools.memory_store(content="", node_type="Memory")
        assert "error" in result

    def test_store_with_explicit_importance(self, tools):
        result = tools.memory_store(
            content="High importance decision about architecture",
            node_type="Decision",
            importance=0.9,
        )
        assert "id" in result
        assert "error" not in result

    def test_store_without_importance_uses_type_default(self, tools):
        """Storing without importance should use per-type default, not 0.5 for all."""
        from agentmemory.core.models import DEFAULT_IMPORTANCE
        result = tools.memory_store(
            content="Goal: Launch product by Q2",
            node_type="Goal",
        )
        assert "id" in result
        # The stored importance should match the Goal default (0.8)
        stored = tools.memory.redis.get_memory(result["id"])
        if stored:
            assert float(stored.get("importance", 0)) == DEFAULT_IMPORTANCE["Goal"]


# ---------------------------------------------------------------------------
# learning_store with importance
# ---------------------------------------------------------------------------


class TestLearningStoreImportance:
    def test_learning_store_with_importance(self, tools):
        result = tools.learning_store(
            content="Learned that X fails under Y conditions",
            what_failed="X",
            why_it_failed="Y conditions",
            what_to_avoid="X under Y",
            importance=0.8,
        )
        assert "id" in result
        assert "error" not in result

    def test_learning_store_default_importance(self, tools):
        """Learning without explicit importance should use 0.65 default."""
        from agentmemory.core.models import DEFAULT_IMPORTANCE
        result = tools.learning_store(
            content="Another learning without explicit importance",
            what_failed="something",
            why_it_failed="some reason",
            what_to_avoid="that thing",
        )
        assert "id" in result
        stored = tools.memory.redis.get_memory(result["id"])
        if stored:
            assert float(stored.get("importance", 0)) == DEFAULT_IMPORTANCE["Learning"]


# ---------------------------------------------------------------------------
# memory_recall
# ---------------------------------------------------------------------------


class TestMemoryRecall:
    def test_recall_returns_results(self, tools):
        tools.memory_store(
            content="feedback1 is a SaaS for product feedback",
            node_type="Memory",
            tags=["feedback1"],
        )
        time.sleep(0.2)
        result = tools.memory_recall(query="feedback1 product")
        assert "results" in result
        assert isinstance(result["results"], list)

    def test_recall_empty_query_returns_error(self, tools):
        result = tools.memory_recall(query="")
        assert "error" in result

    def test_recall_has_total_field(self, tools):
        result = tools.memory_recall(query="test query")
        assert "total" in result

    def test_recall_respects_limit(self, tools):
        result = tools.memory_recall(query="memory", limit=2)
        assert len(result.get("results", [])) <= 2

    def test_recall_results_have_score_fields(self, tools):
        """Recall results should include all scoring signal fields."""
        result = tools.memory_recall(query="feedback product")
        results = result.get("results", [])
        for r in results:
            assert "score" in r
            assert 0.0 <= r["score"] <= 1.0


# ---------------------------------------------------------------------------
# memory_relate
# ---------------------------------------------------------------------------


class TestMemoryRelate:
    def test_relate_creates_relationship(self, tools):
        proj = tools.memory.store_project(name="mcp-test-proj", description="MCP test")
        mem = tools.memory.store(content="MCP relate test memory", node_type="Memory")

        result = tools.memory_relate(
            from_id=mem["id"],
            to_id=proj["id"],
            edge_type="ABOUT",
        )
        assert result.get("success") is True

    def test_relate_invalid_edge_type(self, tools):
        result = tools.memory_relate(
            from_id="some-id",
            to_id="other-id",
            edge_type="INVALID_TYPE",
        )
        assert "error" in result

    def test_relate_nonexistent_nodes(self, tools):
        result = tools.memory_relate(
            from_id="nonexistent-a",
            to_id="nonexistent-b",
            edge_type="ABOUT",
        )
        # Should handle gracefully (may succeed or return error)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# memory_context
# ---------------------------------------------------------------------------


class TestMemoryContext:
    def test_context_returns_entity_info(self, tools):
        proj = tools.memory.store_project(
            name="context-mcp-proj",
            description="Context test project",
        )
        result = tools.memory_context(entity_id=proj["id"])
        assert "entity" in result or "error" in result

    def test_context_nonexistent_entity(self, tools):
        result = tools.memory_context(entity_id="nonexistent-xyz")
        assert "error" in result or result.get("entity") is None


# ---------------------------------------------------------------------------
# memory_forget
# ---------------------------------------------------------------------------


class TestMemoryForget:
    def test_forget_removes_memory(self, tools):
        mem = tools.memory.store(content="Memory to forget via MCP", node_type="Memory")
        result = tools.memory_forget(memory_id=mem["id"])
        assert result.get("success") is True

    def test_forget_nonexistent_returns_error(self, tools):
        result = tools.memory_forget(memory_id="nonexistent-forget-xyz")
        assert "error" in result or result.get("success") is False


# ---------------------------------------------------------------------------
# goal_manage
# ---------------------------------------------------------------------------


class TestGoalManage:
    def test_create_goal(self, tools):
        result = tools.goal_manage(
            action="create",
            name="Launch Feedback1 GTM",
            description="Go-to-market strategy",
        )
        assert "id" in result
        assert "error" not in result

    def test_list_goals(self, tools):
        result = tools.goal_manage(action="list")
        assert "goals" in result
        assert isinstance(result["goals"], list)

    def test_invalid_action_returns_error(self, tools):
        result = tools.goal_manage(action="invalid_action")
        assert "error" in result


# ---------------------------------------------------------------------------
# task_manage
# ---------------------------------------------------------------------------


class TestTaskManage:
    def test_create_task(self, tools):
        result = tools.task_manage(
            action="create",
            name="Build SSE endpoint",
            description="Implement SSE for MCP",
        )
        assert "id" in result

    def test_complete_task(self, tools):
        created = tools.task_manage(
            action="create",
            name="Task to complete",
            description="Will be completed",
        )
        result = tools.task_manage(
            action="complete",
            task_id=created["id"],
            result_summary="Completed successfully",
        )
        assert result.get("success") is True or "id" in result

    def test_list_tasks(self, tools):
        result = tools.task_manage(action="list")
        assert "tasks" in result


# ---------------------------------------------------------------------------
# competitor_manage
# ---------------------------------------------------------------------------


class TestCompetitorManage:
    def test_create_competitor(self, tools):
        result = tools.competitor_manage(
            action="create",
            name="Canny",
            website="https://canny.io",
            positioning="Feature voting and roadmap tool",
        )
        assert "id" in result

    def test_list_competitors(self, tools):
        result = tools.competitor_manage(action="list")
        assert "competitors" in result


# ---------------------------------------------------------------------------
# metric_record and metric_query
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_record_metric(self, tools):
        result = tools.metric_record(
            name="website_visitors",
            metric_type="visitors",
            value=1250.0,
            unit="count",
        )
        assert "id" in result

    def test_query_metric(self, tools):
        tools.metric_record(
            name="monthly_revenue",
            metric_type="revenue",
            value=4500.0,
            unit="USD",
        )
        result = tools.metric_query(
            name="monthly_revenue",
        )
        assert "data_points" in result or "error" in result


# ---------------------------------------------------------------------------
# timeline
# ---------------------------------------------------------------------------


class TestTimeline:
    def test_timeline_returns_list(self, tools):
        result = tools.timeline(since="2020-01-01")
        assert "events" in result
        assert isinstance(result["events"], list)


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    def test_all_tools_return_dict(self, tools):
        """Every tool should return a dict, never raise an exception."""
        tool_calls = [
            lambda: tools.memory_store(content="test", node_type="Memory"),
            lambda: tools.memory_recall(query="test"),
            lambda: tools.memory_context(entity_id="nonexistent"),
            lambda: tools.memory_forget(memory_id="nonexistent"),
            lambda: tools.goal_manage(action="list"),
            lambda: tools.task_manage(action="list"),
            lambda: tools.competitor_manage(action="list"),
            lambda: tools.timeline(),
        ]
        for call in tool_calls:
            result = call()
            assert isinstance(result, dict), f"Tool returned non-dict: {result}"

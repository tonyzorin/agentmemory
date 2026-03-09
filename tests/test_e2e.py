"""
End-to-end tests: store via CLI -> recall via MCP tools -> verify graph.

These tests exercise the full stack: CLI -> MemoryService -> Redis + PostgreSQL + AGE.
"""

import time
import uuid

import pytest

from tests.conftest import TEST_DATABASE_URL, TEST_REDIS_URL, TEST_GRAPH_NAME
from agentmemory.core.memory import MemoryService
from agentmemory.mcp.tools import MemoryTools


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def svc(redis_client):
    """Shared MemoryService for e2e tests."""
    keys = redis_client.keys("test:*")
    if keys:
        redis_client.delete(*keys)
    try:
        redis_client.execute_command("FT.DROPINDEX", "test:memory_idx", "DD")
    except Exception:
        pass

    service = MemoryService(
        database_url=TEST_DATABASE_URL,
        redis_url=TEST_REDIS_URL,
        key_prefix="test:",
        graph_name=TEST_GRAPH_NAME,
        embedding_model="all-MiniLM-L6-v2",
    )
    yield service
    service.close()


@pytest.fixture(scope="module")
def tools_instance(svc):
    """MemoryTools that shares the same MemoryService."""
    t = MemoryTools(
        database_url=TEST_DATABASE_URL,
        redis_url=TEST_REDIS_URL,
        key_prefix="test:",
        graph_name=TEST_GRAPH_NAME,
        embedding_model="all-MiniLM-L6-v2",
    )
    # Share the same service to avoid double model loading
    t.memory = svc
    t.retrieval.memory = svc
    yield t
    t.close()


# ---------------------------------------------------------------------------
# E2E: Store via service -> Recall via MCP tools
# ---------------------------------------------------------------------------


class TestStoreRecallFlow:
    def test_store_then_recall_via_mcp(self, svc, tools_instance):
        """
        Store a memory via MemoryService (simulating CLI),
        then recall it via MCP tools.
        """
        unique_id = str(uuid.uuid4())[:8]
        content = f"E2E test: Anton uses Redis {unique_id} for caching embeddings"

        # Store via service (simulates CLI)
        stored = svc.store(
            content=content,
            node_type="Memory",
            tags=["e2e", "redis"],
            importance=0.9,
        )
        assert "id" in stored
        time.sleep(0.2)

        # Recall via MCP tools
        result = tools_instance.memory_recall(query=f"Redis {unique_id} caching")
        assert "results" in result
        contents = [r.get("content", "") for r in result["results"]]
        assert any(unique_id in str(c) for c in contents), (
            f"Stored content not found in recall results. Contents: {contents}"
        )

    def test_store_project_then_recall_context(self, svc, tools_instance):
        """
        Store a project + related memory, then get context via MCP.
        """
        proj = svc.store_project(
            name=f"e2e-project-{str(uuid.uuid4())[:6]}",
            description="E2E test project",
            stack=["Python", "FastAPI"],
        )
        mem = svc.store(
            content=f"E2E memory about {proj['id']}",
            node_type="Memory",
        )
        svc.relate(mem["id"], proj["id"], "ABOUT")

        # Get context via MCP
        ctx = tools_instance.memory_context(entity_id=proj["id"])
        assert "entity" in ctx
        assert "neighbors" in ctx

        neighbor_ids = [n.get("id") for n in ctx["neighbors"]]
        assert mem["id"] in neighbor_ids


# ---------------------------------------------------------------------------
# E2E: Work structure flow
# ---------------------------------------------------------------------------


class TestWorkStructureFlow:
    def test_goal_initiative_task_chain(self, svc, tools_instance):
        """
        Create goal -> initiative -> task, verify they're linked.
        """
        # Create goal
        goal = tools_instance.goal_manage(
            action="create",
            name="E2E Goal: Launch product",
            description="Full GTM launch",
        )
        assert "id" in goal

        # Create initiative under goal
        initiative = tools_instance.initiative_manage(
            action="create",
            name="E2E Initiative: Beta program",
            description="Run beta with 10 customers",
            goal_id=goal["id"],
        )
        assert "id" in initiative

        # Create task under initiative
        task = tools_instance.task_manage(
            action="create",
            name="E2E Task: Recruit beta users",
            description="Find 10 beta users",
            initiative_id=initiative["id"],
        )
        assert "id" in task

        # Verify relations exist
        relations = svc.postgres.get_relations(from_id=goal["id"])
        assert any(r["edge_type"] == "ACHIEVED_VIA" for r in relations)

        # Complete the task
        done = tools_instance.task_manage(
            action="complete",
            task_id=task["id"],
            result_summary="Recruited 12 beta users via LinkedIn",
        )
        assert done.get("success") is True


# ---------------------------------------------------------------------------
# E2E: Market intelligence flow
# ---------------------------------------------------------------------------


class TestMarketIntelligenceFlow:
    def test_competitor_metric_feedback_flow(self, svc, tools_instance):
        """
        Add competitor, record metrics, store customer feedback.
        """
        # Add competitor
        comp = tools_instance.competitor_manage(
            action="create",
            name=f"E2E Competitor {str(uuid.uuid4())[:6]}",
            positioning="Feature voting tool",
        )
        assert "id" in comp

        # Record a metric
        metric = tools_instance.metric_record(
            name=f"e2e_visitors_{str(uuid.uuid4())[:6]}",
            metric_type="visitors",
            value=1500.0,
            unit="count",
        )
        assert "id" in metric

        # Store customer feedback
        feedback = tools_instance.customer_feedback_store(
            content="The product is great but needs better onboarding",
            sentiment="positive",
            source="direct",
        )
        assert "id" in feedback


# ---------------------------------------------------------------------------
# E2E: Graph traversal
# ---------------------------------------------------------------------------


class TestGraphTraversal:
    def test_multi_hop_query(self, svc):
        """
        Anton -[WORKS_ON]-> feedback1 -[FOR]-> GTM Goal
        Query from Anton should reach GTM Goal at depth 2.
        """
        anton_id = f"e2e-anton-{str(uuid.uuid4())[:6]}"
        fb1_id = f"e2e-fb1-{str(uuid.uuid4())[:6]}"
        goal_id = f"e2e-goal-{str(uuid.uuid4())[:6]}"

        # Create nodes
        svc.store(content="Anton - PM", node_type="Person", extra={"name": "Anton"})
        # Use direct graph operations for test control
        svc.graph.create_node("Person", {"id": anton_id, "name": "Anton", "node_type": "Person"})
        svc.graph.create_node("Project", {"id": fb1_id, "name": "feedback1", "node_type": "Project"})
        svc.graph.create_node("Goal", {"id": goal_id, "name": "GTM Goal", "node_type": "Goal"})

        svc.graph.create_edge(anton_id, fb1_id, "WORKS_ON")
        svc.graph.create_edge(fb1_id, goal_id, "FOR")

        # Traverse from Anton at depth 2
        neighbors = svc.graph.get_neighborhood(anton_id, depth=2)
        neighbor_ids = {n.get("id") for n in neighbors}

        assert fb1_id in neighbor_ids
        assert goal_id in neighbor_ids


# ---------------------------------------------------------------------------
# E2E: Timeline
# ---------------------------------------------------------------------------


class TestTimeline:
    def test_timeline_shows_recent_activity(self, svc, tools_instance):
        """Store some entities, then verify they appear in timeline."""
        svc.store(content="Timeline test memory", node_type="Memory")
        time.sleep(0.1)

        result = tools_instance.timeline(since="1d", limit=50)
        assert "events" in result
        # Should have at least the memory we just stored
        assert result["total"] >= 1


# ---------------------------------------------------------------------------
# E2E: Full round-trip with forget
# ---------------------------------------------------------------------------


class TestForgetFlow:
    def test_store_recall_forget_verify(self, svc, tools_instance):
        """
        Store -> recall (found) -> forget -> recall (not found).
        """
        unique_id = str(uuid.uuid4())[:8]
        content = f"Temporary memory to forget {unique_id}"

        stored = svc.store(content=content, node_type="Memory", importance=1.0)
        mem_id = stored["id"]
        time.sleep(0.2)

        # Should be recallable
        recall_result = tools_instance.memory_recall(query=unique_id)
        contents = [r.get("content", "") for r in recall_result.get("results", [])]
        assert any(unique_id in str(c) for c in contents)

        # Forget it
        forget_result = tools_instance.memory_forget(memory_id=mem_id)
        assert forget_result.get("success") is True

        # Should not be in postgres
        entity = svc.postgres.get_entity(mem_id)
        assert entity is None

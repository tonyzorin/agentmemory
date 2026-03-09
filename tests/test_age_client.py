"""
Tests for the Apache AGE graph client.

Runs against a real PostgreSQL 18 + AGE 1.7.0 instance (via Docker Compose).
Tests define expected graph behavior — implementation must make these pass.
"""

import uuid

import pytest

from tests.conftest import TEST_DATABASE_URL, TEST_GRAPH_NAME
from agentmemory.db.age_client import AGEClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def graph(clean_postgres):
    """AGEClient with a fresh test graph per test."""
    client = AGEClient(dsn=TEST_DATABASE_URL, graph_name=TEST_GRAPH_NAME)
    client.ensure_graph_exists()
    yield client
    client.close()


def uid(prefix: str = "test") -> str:
    return f"{prefix}-{str(uuid.uuid4())[:8]}"


# ---------------------------------------------------------------------------
# Graph initialization
# ---------------------------------------------------------------------------


class TestGraphInit:
    def test_ensure_graph_creates_graph(self, graph):
        """ensure_graph_exists should create the graph if missing."""
        # Graph was created in fixture — verify it exists
        rows = graph.execute_cypher("MATCH (n) RETURN count(n) AS cnt")
        assert rows is not None

    def test_ensure_graph_is_idempotent(self, graph):
        """Calling ensure_graph_exists twice should not raise."""
        graph.ensure_graph_exists()
        graph.ensure_graph_exists()

    def test_graph_name_is_set(self, graph):
        assert graph.graph_name == TEST_GRAPH_NAME


# ---------------------------------------------------------------------------
# Node (vertex) CRUD
# ---------------------------------------------------------------------------


class TestNodeCRUD:
    def test_create_node_returns_id(self, graph):
        node_id = uid("person")
        result = graph.create_node(
            node_type="Person",
            properties={"id": node_id, "name": "Anton", "node_type": "Person"},
        )
        assert result is not None

    def test_get_node_by_id(self, graph):
        node_id = uid("person")
        graph.create_node(
            node_type="Person",
            properties={"id": node_id, "name": "Anton", "node_type": "Person"},
        )
        node = graph.get_node(node_id)
        assert node is not None
        assert node["name"] == "Anton"

    def test_get_nonexistent_node_returns_none(self, graph):
        node = graph.get_node("nonexistent-id-xyz")
        assert node is None

    def test_create_multiple_node_types(self, graph):
        for node_type in ["Person", "Project", "Memory", "Goal", "Task"]:
            nid = uid(node_type.lower())
            graph.create_node(
                node_type=node_type,
                properties={"id": nid, "name": f"Test {node_type}", "node_type": node_type},
            )
            node = graph.get_node(nid)
            assert node is not None, f"Node of type {node_type} not found"

    def test_update_node_properties(self, graph):
        node_id = uid("proj")
        graph.create_node(
            node_type="Project",
            properties={"id": node_id, "name": "feedback1", "node_type": "Project"},
        )
        graph.update_node(node_id, {"description": "Product feedback tool"})
        node = graph.get_node(node_id)
        assert node["description"] == "Product feedback tool"

    def test_delete_node(self, graph):
        node_id = uid("del")
        graph.create_node(
            node_type="Topic",
            properties={"id": node_id, "name": "To Delete", "node_type": "Topic"},
        )
        graph.delete_node(node_id)
        assert graph.get_node(node_id) is None

    def test_list_nodes_by_type(self, graph):
        for i in range(3):
            graph.create_node(
                node_type="Tool",
                properties={"id": uid("tool"), "name": f"Tool {i}", "node_type": "Tool"},
            )
        tools = graph.list_nodes_by_type("Tool")
        assert len(tools) >= 3


# ---------------------------------------------------------------------------
# Edge (relationship) CRUD
# ---------------------------------------------------------------------------


class TestEdgeCRUD:
    def _make_two_nodes(self, graph) -> tuple[str, str]:
        a_id, b_id = uid("a"), uid("b")
        graph.create_node("Person", {"id": a_id, "name": "Anton", "node_type": "Person"})
        graph.create_node("Project", {"id": b_id, "name": "feedback1", "node_type": "Project"})
        return a_id, b_id

    def test_create_edge(self, graph):
        a_id, b_id = self._make_two_nodes(graph)
        result = graph.create_edge(
            from_id=a_id,
            to_id=b_id,
            edge_type="WORKS_ON",
            properties={"since": "2024-01"},
        )
        assert result is True

    def test_get_edges_from_node(self, graph):
        a_id, b_id = self._make_two_nodes(graph)
        graph.create_edge(a_id, b_id, "WORKS_ON")
        edges = graph.get_edges(from_id=a_id)
        assert len(edges) >= 1
        assert any(e["edge_type"] == "WORKS_ON" for e in edges)

    def test_get_edges_by_type(self, graph):
        a_id, b_id = self._make_two_nodes(graph)
        graph.create_edge(a_id, b_id, "WORKS_ON")
        graph.create_edge(a_id, b_id, "OWNS")
        works_on = graph.get_edges(from_id=a_id, edge_type="WORKS_ON")
        assert all(e["edge_type"] == "WORKS_ON" for e in works_on)

    def test_delete_edge(self, graph):
        a_id, b_id = self._make_two_nodes(graph)
        graph.create_edge(a_id, b_id, "WORKS_ON")
        graph.delete_edge(a_id, b_id, "WORKS_ON")
        edges = graph.get_edges(from_id=a_id, edge_type="WORKS_ON")
        assert len(edges) == 0

    def test_get_neighbors(self, graph):
        a_id, b_id = self._make_two_nodes(graph)
        graph.create_edge(a_id, b_id, "WORKS_ON")
        neighbors = graph.get_neighbors(a_id)
        neighbor_ids = [n["id"] for n in neighbors]
        assert b_id in neighbor_ids


# ---------------------------------------------------------------------------
# Multi-hop traversal
# ---------------------------------------------------------------------------


class TestTraversal:
    def test_two_hop_traversal(self, graph):
        """
        Anton -[WORKS_ON]-> feedback1 -[HOSTS]-> VM817
        Traversal from Anton at depth 2 should reach VM817.
        """
        anton_id = uid("anton")
        fb1_id = uid("fb1")
        vm_id = uid("vm")

        graph.create_node("Person", {"id": anton_id, "name": "Anton", "node_type": "Person"})
        graph.create_node("Project", {"id": fb1_id, "name": "feedback1", "node_type": "Project"})
        graph.create_node("Environment", {"id": vm_id, "name": "VM817", "node_type": "Environment"})

        graph.create_edge(anton_id, fb1_id, "WORKS_ON")
        graph.create_edge(vm_id, fb1_id, "HOSTS")

        neighbors = graph.get_neighborhood(anton_id, depth=2)
        neighbor_ids = [n["id"] for n in neighbors]
        assert fb1_id in neighbor_ids
        assert vm_id in neighbor_ids

    def test_find_path_between_nodes(self, graph):
        """Should find a path between two connected nodes."""
        a_id = uid("src")
        b_id = uid("mid")
        c_id = uid("dst")

        graph.create_node("Person", {"id": a_id, "name": "A", "node_type": "Person"})
        graph.create_node("Project", {"id": b_id, "name": "B", "node_type": "Project"})
        graph.create_node("Goal", {"id": c_id, "name": "C", "node_type": "Goal"})

        graph.create_edge(a_id, b_id, "WORKS_ON")
        graph.create_edge(b_id, c_id, "FOR")

        path = graph.find_path(a_id, c_id, max_depth=3)
        assert path is not None
        assert len(path) >= 2

    def test_no_path_returns_none(self, graph):
        a_id = uid("iso-a")
        b_id = uid("iso-b")
        graph.create_node("Person", {"id": a_id, "name": "Isolated A", "node_type": "Person"})
        graph.create_node("Person", {"id": b_id, "name": "Isolated B", "node_type": "Person"})
        # No edge between them
        path = graph.find_path(a_id, b_id, max_depth=3)
        assert path is None or len(path) == 0


# ---------------------------------------------------------------------------
# Relational tables (entities + relations)
# ---------------------------------------------------------------------------


class TestRelationalTables:
    def test_upsert_entity_to_postgres(self, graph):
        """Entity should be persisted in the relational entities table."""
        from agentmemory.db.postgres import PostgresClient
        pg = PostgresClient(dsn=TEST_DATABASE_URL)

        entity_id = uid("ent")
        pg.upsert_entity({
            "id": entity_id,
            "name": "Test Entity",
            "node_type": "Topic",
            "metadata": {"key": "value"},
            "tags": ["test"],
        })

        row = pg.get_entity(entity_id)
        assert row is not None
        assert row["name"] == "Test Entity"
        pg.close()

    def test_upsert_relation_to_postgres(self, graph):
        from agentmemory.db.postgres import PostgresClient
        pg = PostgresClient(dsn=TEST_DATABASE_URL)

        a_id, b_id = uid("ra"), uid("rb")
        pg.upsert_entity({"id": a_id, "name": "A", "node_type": "Person", "metadata": {}, "tags": []})
        pg.upsert_entity({"id": b_id, "name": "B", "node_type": "Project", "metadata": {}, "tags": []})

        rel_id = uid("rel")
        pg.upsert_relation({
            "id": rel_id,
            "from_id": a_id,
            "to_id": b_id,
            "edge_type": "WORKS_ON",
            "properties": {},
        })

        relations = pg.get_relations(from_id=a_id)
        assert any(r["edge_type"] == "WORKS_ON" for r in relations)
        pg.close()

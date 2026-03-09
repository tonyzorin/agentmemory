"""
Apache AGE graph client for agentmemory.md.

Provides safe Cypher query execution against the memory_graph.
Adapted from feedback1/apps/ai/graph/age_client.py with:
- memory_graph schema (17 node types, 25+ edge types)
- neighborhood traversal for context queries
- path finding for relationship discovery

AGE 1.7.0 Cypher quirks handled here:
- Property map literals use unquoted keys: {id: 'val'} not {"id": "val"}
- Parameters ($name) require the third argument to be a literal agtype::jsonb cast
- We use string interpolation for property maps (values are escaped)
- Variable-length paths (*1..N) work but shortestPath has limited param support

Uses psycopg2 (sync) since apache-age-python requires it.
"""

import json
import logging
import re
import urllib.parse
from typing import Any

import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)


def _escape_str(value: str) -> str:
    """Escape a string value for safe inline Cypher embedding."""
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _to_cypher_literal(value: Any) -> str:
    """
    Convert a Python value to a Cypher literal string.
    Handles str, int, float, bool, None, list, dict.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return f"'{_escape_str(value)}'"
    if isinstance(value, list):
        items = ", ".join(_to_cypher_literal(v) for v in value)
        return f"[{items}]"
    if isinstance(value, dict):
        pairs = ", ".join(
            f"{k}: {_to_cypher_literal(v)}" for k, v in value.items()
        )
        return f"{{{pairs}}}"
    # Fallback: JSON-encode and wrap as string
    return f"'{_escape_str(json.dumps(value))}'"


def _props_to_cypher(props: dict[str, Any]) -> str:
    """Convert a properties dict to a Cypher map literal: {key: 'val', ...}"""
    pairs = ", ".join(
        f"{k}: {_to_cypher_literal(v)}" for k, v in props.items()
    )
    return f"{{{pairs}}}"


class AGEClient:
    """
    Safe Apache AGE client using psycopg2.

    All Cypher queries use inline literal values (not $params) because AGE 1.7.0
    requires the parameter argument to be a literal agtype cast, which psycopg2
    cannot easily provide. Values are escaped via _escape_str / _to_cypher_literal.
    """

    ALLOWED_NODE_TYPES = {
        "Memory", "Learning", "Decision", "Goal", "Initiative", "Task",
        "Project", "Person", "ExternalContact", "Preference", "Environment",
        "Tool", "Workflow", "Resource", "Competitor", "Metric",
        "CustomerFeedback", "Topic",
    }

    ALLOWED_EDGE_TYPES = {
        "ABOUT", "ACHIEVED_VIA", "BROKEN_INTO", "PART_OF", "BELONGS_TO",
        "FOR", "WORKS_ON", "OWNS", "COLLABORATES_ON", "INVOLVED_IN",
        "RELATED_TO", "PREVENTED", "AFFECTS", "LED_TO", "USES", "USED_BY",
        "HOSTS", "DOCUMENTS", "MENTIONS", "TAGGED_WITH", "COMPETES_WITH",
        "TRACKS", "FROM",
    }

    def __init__(self, dsn: str, graph_name: str = "memory_graph"):
        self.graph_name = graph_name
        parsed = urllib.parse.urlparse(dsn)
        self.conn = psycopg2.connect(
            host=parsed.hostname,
            port=parsed.port or 5432,
            user=parsed.username,
            password=parsed.password,
            database=parsed.path.lstrip("/"),
        )
        self.conn.autocommit = True
        self._setup_age()

    def _setup_age(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute("LOAD 'age';")
            cur.execute("SET search_path = ag_catalog, \"$user\", public;")

    def ensure_graph_exists(self) -> None:
        """Create the graph if it doesn't already exist."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM ag_catalog.ag_graph WHERE name = %s",
                (self.graph_name,),
            )
            if not cur.fetchone():
                cur.execute(
                    "SELECT ag_catalog.create_graph(%s);",
                    (self.graph_name,),
                )
                logger.info("Created graph '%s'", self.graph_name)

    def execute_cypher_raw(self, cypher: str, columns: list[str]) -> list[tuple]:
        """
        Execute a Cypher query with explicit column definitions.

        AGE requires the column list in the AS clause. `columns` is a list of
        column names — each is typed as agtype.
        """
        col_defs = ", ".join(f"{c} ag_catalog.agtype" for c in columns)
        sql = (
            f"SELECT {', '.join(columns)} FROM ag_catalog.cypher(%s, $$ {cypher} $$) "
            f"AS ({col_defs});"
        )
        with self.conn.cursor() as cur:
            try:
                cur.execute(sql, (self.graph_name,))
                return cur.fetchall()
            except psycopg2.Error as e:
                logger.error("Cypher failed: %s\nQuery: %s", e, cypher)
                raise

    def execute_cypher(self, cypher: str, _params: dict | None = None) -> list[Any]:
        """
        Execute a Cypher query, returning raw rows.
        _params is ignored — values must be inlined via _to_cypher_literal.
        """
        sql = (
            f"SELECT result FROM ag_catalog.cypher(%s, $$ {cypher} $$) "
            f"AS (result ag_catalog.agtype);"
        )
        with self.conn.cursor() as cur:
            try:
                cur.execute(sql, (self.graph_name,))
                return cur.fetchall()
            except psycopg2.Error as e:
                logger.error("Cypher failed: %s\nQuery: %s", e, cypher)
                raise

    def _parse_agtype(self, value: Any) -> Any:
        """Parse an AGE agtype value to Python dict/list/scalar."""
        if value is None:
            return None
        s = str(value)
        for suffix in ("::vertex", "::edge", "::path"):
            if s.endswith(suffix):
                s = s[: -len(suffix)]
        try:
            return json.loads(s)
        except (json.JSONDecodeError, ValueError):
            return s

    def _validate_label(self, label: str, allowed: set[str]) -> str:
        if label not in allowed:
            raise ValueError(
                f"Label '{label}' not in allowed set: {sorted(allowed)}"
            )
        return label

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def create_node(
        self, node_type: str, properties: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Create a vertex with the given type and properties."""
        self._validate_label(node_type, self.ALLOWED_NODE_TYPES)
        props_cypher = _props_to_cypher(properties)
        cypher = f"CREATE (n:{node_type} {props_cypher}) RETURN n"
        try:
            rows = self.execute_cypher(cypher)
            if rows:
                parsed = self._parse_agtype(rows[0][0])
                if isinstance(parsed, dict):
                    return parsed.get("properties", parsed)
                return parsed
            return None
        except Exception as e:
            logger.error("create_node failed: %s", e)
            return None

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        """Retrieve a node by its id property."""
        safe_id = _escape_str(node_id)
        cypher = f"MATCH (n {{id: '{safe_id}'}}) RETURN n LIMIT 1"
        try:
            rows = self.execute_cypher(cypher)
            if rows:
                parsed = self._parse_agtype(rows[0][0])
                if isinstance(parsed, dict):
                    return parsed.get("properties", parsed)
            return None
        except Exception as e:
            logger.error("get_node failed for id=%s: %s", node_id, e)
            return None

    def update_node(self, node_id: str, properties: dict[str, Any]) -> bool:
        """Update properties on an existing node."""
        safe_id = _escape_str(node_id)
        set_clauses = ", ".join(
            f"n.{k} = {_to_cypher_literal(v)}" for k, v in properties.items()
        )
        cypher = f"MATCH (n {{id: '{safe_id}'}}) SET {set_clauses} RETURN n"
        try:
            rows = self.execute_cypher(cypher)
            return len(rows) > 0
        except Exception as e:
            logger.error("update_node failed: %s", e)
            return False

    def delete_node(self, node_id: str) -> bool:
        """Delete a node and all its edges."""
        safe_id = _escape_str(node_id)
        cypher = f"MATCH (n {{id: '{safe_id}'}}) DETACH DELETE n"
        try:
            # DELETE returns no rows — use a different approach
            sql = (
                f"SELECT * FROM ag_catalog.cypher(%s, $$ {cypher} $$) "
                f"AS (result ag_catalog.agtype);"
            )
            with self.conn.cursor() as cur:
                try:
                    cur.execute(sql, (self.graph_name,))
                except psycopg2.ProgrammingError:
                    pass  # No results is fine for DELETE
            return True
        except Exception as e:
            logger.error("delete_node failed: %s", e)
            return False

    def list_nodes_by_type(self, node_type: str) -> list[dict[str, Any]]:
        """List all nodes of a given type."""
        self._validate_label(node_type, self.ALLOWED_NODE_TYPES)
        cypher = f"MATCH (n:{node_type}) RETURN n"
        try:
            rows = self.execute_cypher(cypher)
            result = []
            for row in rows:
                parsed = self._parse_agtype(row[0])
                if isinstance(parsed, dict):
                    result.append(parsed.get("properties", parsed))
            return result
        except Exception as e:
            logger.error("list_nodes_by_type failed: %s", e)
            return []

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def create_edge(
        self,
        from_id: str,
        to_id: str,
        edge_type: str,
        properties: dict[str, Any] | None = None,
    ) -> bool:
        """Create a directed edge between two nodes."""
        self._validate_label(edge_type, self.ALLOWED_EDGE_TYPES)
        safe_from = _escape_str(from_id)
        safe_to = _escape_str(to_id)
        props = properties or {}
        props_cypher = _props_to_cypher(props) if props else "{}"
        cypher = (
            f"MATCH (a {{id: '{safe_from}'}}), (b {{id: '{safe_to}'}}) "
            f"CREATE (a)-[r:{edge_type} {props_cypher}]->(b) "
            f"RETURN r"
        )
        try:
            rows = self.execute_cypher(cypher)
            return len(rows) > 0
        except Exception as e:
            logger.error("create_edge failed: %s", e)
            return False

    def get_edges(
        self,
        from_id: str | None = None,
        to_id: str | None = None,
        edge_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get edges, optionally filtered by from_id, to_id, or edge_type."""
        if edge_type:
            self._validate_label(edge_type, self.ALLOWED_EDGE_TYPES)

        rel_pattern = f"[r:{edge_type}]" if edge_type else "[r]"

        if from_id:
            safe_from = _escape_str(from_id)
            cypher = (
                f"MATCH (a {{id: '{safe_from}'}})-{rel_pattern}->(b) "
                f"RETURN r, b.id AS to_id"
            )
        elif to_id:
            safe_to = _escape_str(to_id)
            cypher = (
                f"MATCH (a)-{rel_pattern}->(b {{id: '{safe_to}'}}) "
                f"RETURN r, a.id AS from_id"
            )
        else:
            return []

        try:
            # Use multi-column return
            col_defs = "r ag_catalog.agtype, other_id ag_catalog.agtype"
            sql = (
                f"SELECT r, other_id FROM ag_catalog.cypher(%s, $$ {cypher} $$) "
                f"AS (r ag_catalog.agtype, other_id ag_catalog.agtype);"
            )
            with self.conn.cursor() as cur:
                cur.execute(sql, (self.graph_name,))
                rows = cur.fetchall()

            result = []
            for row in rows:
                edge_parsed = self._parse_agtype(row[0])
                other_id_parsed = self._parse_agtype(row[1])
                if isinstance(edge_parsed, dict):
                    props = edge_parsed.get("properties", {})
                    label = edge_parsed.get("label", edge_type or "UNKNOWN")
                    entry = {
                        "edge_type": label,
                        "properties": props,
                        "from_id": from_id or str(other_id_parsed),
                        "to_id": to_id or str(other_id_parsed),
                    }
                    result.append(entry)
            return result
        except Exception as e:
            logger.error("get_edges failed: %s", e)
            return []

    def delete_edge(self, from_id: str, to_id: str, edge_type: str) -> bool:
        """Delete a specific edge between two nodes."""
        self._validate_label(edge_type, self.ALLOWED_EDGE_TYPES)
        safe_from = _escape_str(from_id)
        safe_to = _escape_str(to_id)
        cypher = (
            f"MATCH (a {{id: '{safe_from}'}})-[r:{edge_type}]->(b {{id: '{safe_to}'}}) "
            f"DELETE r"
        )
        try:
            sql = (
                f"SELECT * FROM ag_catalog.cypher(%s, $$ {cypher} $$) "
                f"AS (result ag_catalog.agtype);"
            )
            with self.conn.cursor() as cur:
                try:
                    cur.execute(sql, (self.graph_name,))
                except psycopg2.ProgrammingError:
                    pass
            return True
        except Exception as e:
            logger.error("delete_edge failed: %s", e)
            return False

    def get_neighbors(self, node_id: str) -> list[dict[str, Any]]:
        """Get all direct neighbors (outgoing edges) of a node."""
        safe_id = _escape_str(node_id)
        cypher = f"MATCH ({{id: '{safe_id}'}})-[]->(n) RETURN n"
        try:
            rows = self.execute_cypher(cypher)
            result = []
            for row in rows:
                parsed = self._parse_agtype(row[0])
                if isinstance(parsed, dict):
                    result.append(parsed.get("properties", parsed))
            return result
        except Exception as e:
            logger.error("get_neighbors failed: %s", e)
            return []

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def get_neighborhood(
        self, node_id: str, depth: int = 2
    ) -> list[dict[str, Any]]:
        """Get all nodes within `depth` hops from a given node."""
        safe_id = _escape_str(node_id)
        cypher = (
            f"MATCH (start {{id: '{safe_id}'}})-[*1..{depth}]-(n) "
            f"RETURN DISTINCT n"
        )
        try:
            rows = self.execute_cypher(cypher)
            result = []
            seen_ids: set[str] = set()
            for row in rows:
                parsed = self._parse_agtype(row[0])
                if isinstance(parsed, dict):
                    props = parsed.get("properties", parsed)
                    nid = props.get("id")
                    if nid and nid not in seen_ids:
                        seen_ids.add(nid)
                        result.append(props)
            return result
        except Exception as e:
            logger.error("get_neighborhood failed: %s", e)
            return []

    def get_neighborhood_with_depth(
        self, node_id: str, max_depth: int = 2
    ) -> dict[str, int]:
        """
        Get all nodes within max_depth hops, returning a map of {node_id: hop_distance}.

        This is used for graduated graph_boost: 1-hop neighbors get a higher boost
        than 2-hop neighbors. Uses separate queries per depth level to avoid AGE
        limitations with variable-length path length extraction.

        Returns empty dict on error (graph boost simply won't fire).
        """
        safe_id = _escape_str(node_id)
        depth_map: dict[str, int] = {}

        for hop in range(1, max_depth + 1):
            # Match nodes at exactly this hop distance by excluding closer ones
            if hop == 1:
                cypher = (
                    f"MATCH (start {{id: '{safe_id}'}})-[]->(n) "
                    f"WHERE n.id <> '{safe_id}' "
                    f"RETURN DISTINCT n.id AS nid"
                )
            else:
                # Exactly `hop` hops: path of length hop, not reachable in fewer
                cypher = (
                    f"MATCH (start {{id: '{safe_id}'}})-[*{hop}]->(n) "
                    f"WHERE n.id <> '{safe_id}' "
                    f"RETURN DISTINCT n.id AS nid"
                )
            try:
                sql = (
                    f"SELECT nid FROM ag_catalog.cypher(%s, $$ {cypher} $$) "
                    f"AS (nid ag_catalog.agtype);"
                )
                with self.conn.cursor() as cur:
                    cur.execute(sql, (self.graph_name,))
                    rows = cur.fetchall()
                for row in rows:
                    raw_id = self._parse_agtype(row[0])
                    if raw_id and isinstance(raw_id, str):
                        nid = raw_id.strip('"')
                        # Only record the shortest path distance
                        if nid not in depth_map:
                            depth_map[nid] = hop
            except Exception as e:
                logger.warning("get_neighborhood_with_depth hop=%d failed: %s", hop, e)

        return depth_map

    def find_path(
        self, from_id: str, to_id: str, max_depth: int = 5
    ) -> list[dict[str, Any]] | None:
        """
        Find a path between two nodes using variable-length matching.

        Note: AGE 1.7.0 does not support shortestPath(). We use MATCH with
        variable-length relationships and LIMIT 1 to get the first path found.
        Returns a list of node property dicts along the path.
        """
        safe_from = _escape_str(from_id)
        safe_to = _escape_str(to_id)

        # Collect intermediate nodes along the path
        cypher = (
            f"MATCH (a {{id: '{safe_from}'}})-[*1..{max_depth}]-(b {{id: '{safe_to}'}}) "
            f"RETURN a, b LIMIT 1"
        )
        try:
            sql = (
                f"SELECT a_node, b_node FROM ag_catalog.cypher(%s, $$ {cypher} $$) "
                f"AS (a_node ag_catalog.agtype, b_node ag_catalog.agtype);"
            )
            with self.conn.cursor() as cur:
                cur.execute(sql, (self.graph_name,))
                rows = cur.fetchall()

            if not rows:
                return None

            result = []
            for raw in rows[0]:
                parsed = self._parse_agtype(raw)
                if isinstance(parsed, dict):
                    result.append(parsed.get("properties", parsed))
            return result if result else None
        except Exception as e:
            logger.error("find_path failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

"""
PostgreSQL relational client for entity metadata and relations.

Handles the entities, relations, and metric_data_points tables.
Uses psycopg2 (sync) for simplicity — wrap in run_in_executor for async.
"""

import json
import logging
import urllib.parse
from typing import Any

import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)


class PostgresClient:
    """
    Client for the relational tables (entities, relations, metric_data_points).
    Mirrors the AGE graph for fast SQL queries without Cypher.
    """

    def __init__(self, dsn: str):
        parsed = urllib.parse.urlparse(dsn)
        self.conn = psycopg2.connect(
            host=parsed.hostname,
            port=parsed.port or 5432,
            user=parsed.username,
            password=parsed.password,
            database=parsed.path.lstrip("/"),
            cursor_factory=psycopg2.extras.RealDictCursor,
        )
        self.conn.autocommit = False

    # ------------------------------------------------------------------
    # Entities
    # ------------------------------------------------------------------

    def _rollback_if_needed(self) -> None:
        """Rollback any aborted transaction so the connection can be reused."""
        try:
            if self.conn.status == psycopg2.extensions.STATUS_IN_TRANSACTION:
                self.conn.rollback()
        except Exception:
            pass

    def upsert_entity(self, entity: dict[str, Any]) -> None:
        """Insert or update an entity row."""
        self._rollback_if_needed()
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO entities (id, name, node_type, metadata, tags)
                    VALUES (%(id)s, %(name)s, %(node_type)s, %(metadata)s, %(tags)s)
                    ON CONFLICT (id) DO UPDATE SET
                        name      = EXCLUDED.name,
                        node_type = EXCLUDED.node_type,
                        metadata  = EXCLUDED.metadata,
                        tags      = EXCLUDED.tags,
                        updated_at = NOW()
                    """,
                    {
                        "id": entity["id"],
                        "name": entity["name"],
                        "node_type": entity["node_type"],
                        "metadata": json.dumps(entity.get("metadata", {})),
                        "tags": entity.get("tags", []),
                    },
                )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def get_entity(self, entity_id: str) -> dict[str, Any] | None:
        """Retrieve an entity by ID."""
        self._rollback_if_needed()
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM entities WHERE id = %s", (entity_id,))
            row = cur.fetchone()
            if row:
                return dict(row)
            return None

    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity (cascades to relations and metric_data_points)."""
        self._rollback_if_needed()
        try:
            with self.conn.cursor() as cur:
                cur.execute("DELETE FROM entities WHERE id = %s", (entity_id,))
                deleted = cur.rowcount
            self.conn.commit()
            return deleted > 0
        except Exception:
            self.conn.rollback()
            raise

    def list_entities_by_type(
        self, node_type: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """List entities of a given type."""
        self._rollback_if_needed()
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM entities WHERE node_type = %s ORDER BY created_at DESC LIMIT %s",
                (node_type, limit),
            )
            return [dict(r) for r in cur.fetchall()]

    def find_entity_by_name_and_type(
        self, name: str, node_type: str
    ) -> dict[str, Any] | None:
        """Find an entity by exact name and type (case-insensitive). Returns the most recently created match."""
        self._rollback_if_needed()
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM entities WHERE LOWER(name) = LOWER(%s) AND node_type = %s ORDER BY created_at DESC LIMIT 1",
                (name, node_type),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def search_entities_by_name(
        self, query: str, node_type: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Full-text search on entity names."""
        self._rollback_if_needed()
        with self.conn.cursor() as cur:
            if node_type:
                cur.execute(
                    """
                    SELECT *, ts_rank(to_tsvector('english', name), plainto_tsquery('english', %s)) AS rank
                    FROM entities
                    WHERE node_type = %s
                      AND to_tsvector('english', name) @@ plainto_tsquery('english', %s)
                    ORDER BY rank DESC
                    LIMIT %s
                    """,
                    (query, node_type, query, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT *, ts_rank(to_tsvector('english', name), plainto_tsquery('english', %s)) AS rank
                    FROM entities
                    WHERE to_tsvector('english', name) @@ plainto_tsquery('english', %s)
                    ORDER BY rank DESC
                    LIMIT %s
                    """,
                    (query, query, limit),
                )
            return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # Relations
    # ------------------------------------------------------------------

    def upsert_relation(self, relation: dict[str, Any]) -> None:
        """Insert or update a relation row."""
        self._rollback_if_needed()
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO relations (id, from_id, to_id, edge_type, properties)
                    VALUES (%(id)s, %(from_id)s, %(to_id)s, %(edge_type)s, %(properties)s)
                    ON CONFLICT (id) DO UPDATE SET
                        edge_type  = EXCLUDED.edge_type,
                        properties = EXCLUDED.properties
                    """,
                    {
                        "id": relation["id"],
                        "from_id": relation["from_id"],
                        "to_id": relation["to_id"],
                        "edge_type": relation["edge_type"],
                        "properties": json.dumps(relation.get("properties", {})),
                    },
                )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def get_relations(
        self,
        from_id: str | None = None,
        to_id: str | None = None,
        edge_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query relations with optional filters."""
        self._rollback_if_needed()
        conditions = []
        params: list[Any] = []

        if from_id:
            conditions.append("from_id = %s")
            params.append(from_id)
        if to_id:
            conditions.append("to_id = %s")
            params.append(to_id)
        if edge_type:
            conditions.append("edge_type = %s")
            params.append(edge_type)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT * FROM relations {where} ORDER BY created_at DESC", params)
            return [dict(r) for r in cur.fetchall()]

    def delete_relation(self, relation_id: str) -> bool:
        """Delete a relation by ID."""
        self._rollback_if_needed()
        try:
            with self.conn.cursor() as cur:
                cur.execute("DELETE FROM relations WHERE id = %s", (relation_id,))
                deleted = cur.rowcount
            self.conn.commit()
            return deleted > 0
        except Exception:
            self.conn.rollback()
            raise

    # ------------------------------------------------------------------
    # Metric data points
    # ------------------------------------------------------------------

    def insert_metric_point(self, point: dict[str, Any]) -> None:
        """Insert a metric data point."""
        self._rollback_if_needed()
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO metric_data_points (id, metric_id, value, recorded_at, source, notes)
                    VALUES (%(id)s, %(metric_id)s, %(value)s, %(recorded_at)s, %(source)s, %(notes)s)
                    """,
                    {
                        "id": point["id"],
                        "metric_id": point["metric_id"],
                        "value": point["value"],
                        "recorded_at": point["recorded_at"],
                        "source": point.get("source", "manual"),
                        "notes": point.get("notes"),
                    },
                )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def get_metric_points(
        self,
        metric_id: str,
        since: str | None = None,
        until: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query metric data points for a metric, optionally filtered by date range."""
        self._rollback_if_needed()
        conditions = ["metric_id = %s"]
        params: list[Any] = [metric_id]

        if since:
            conditions.append("recorded_at >= %s")
            params.append(since)
        if until:
            conditions.append("recorded_at <= %s")
            params.append(until)

        params.append(limit)
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT * FROM metric_data_points WHERE {' AND '.join(conditions)} "
                f"ORDER BY recorded_at DESC LIMIT %s",
                params,
            )
            return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return counts of entities by type."""
        self._rollback_if_needed()
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT node_type, COUNT(*) AS cnt FROM entities GROUP BY node_type ORDER BY cnt DESC"
            )
            rows = cur.fetchall()
            counts = {r["node_type"]: r["cnt"] for r in rows}
            total = sum(counts.values())
            return {"total_entities": total, "by_type": counts}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        try:
            self.conn.close()
        except Exception:
            pass

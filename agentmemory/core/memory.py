"""
Core memory service — the main entry point for all memory operations.

Orchestrates:
- Embedding generation (sentence-transformers)
- Redis storage + search (vector + hybrid)
- PostgreSQL entity metadata
- Apache AGE knowledge graph

This is the layer that CLI and MCP tools call into.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any

from agentmemory.core.embeddings import EmbeddingService
from agentmemory.core.models import DEFAULT_IMPORTANCE, DEFAULT_TTL_DAYS
from agentmemory.db.age_client import AGEClient
from agentmemory.db.postgres import PostgresClient
from agentmemory.db.redis_client import MemoryRedisClient

logger = logging.getLogger(__name__)

# Node types that get embeddings stored in Redis for semantic search
SEARCHABLE_NODE_TYPES = {
    "Memory", "Learning", "Decision", "Preference", "Workflow", "CustomerFeedback"
}


class MemoryService:
    """
    Unified interface for all memory operations.

    Stores memories across three backends:
    1. Redis — fast vector + hybrid search
    2. PostgreSQL — relational entity metadata + relations
    3. Apache AGE — knowledge graph for multi-hop queries
    """

    def __init__(
        self,
        database_url: str = "postgresql://openclaw:openclaw@localhost:5433/openclaw_memory",
        redis_url: str = "redis://localhost:6380/0",
        key_prefix: str = "",
        graph_name: str = "memory_graph",
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        embedding_dim: int = 768,
    ):
        self.redis = MemoryRedisClient(
            redis_url=redis_url,
            key_prefix=key_prefix,
            embedding_dim=embedding_dim,
        )
        self.postgres = PostgresClient(dsn=database_url)
        self.graph = AGEClient(dsn=database_url, graph_name=graph_name)
        self.graph.ensure_graph_exists()
        self.embeddings = EmbeddingService(
            model_name=embedding_model,
            redis_url=redis_url,
            key_prefix=key_prefix,
            embedding_dim=embedding_dim,
        )

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def store(
        self,
        content: str,
        node_type: str = "Memory",
        name: str | None = None,
        tags: list[str] | None = None,
        source: str = "cli",
        importance: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Store a memory/learning/decision/preference.

        Persists to Redis (with embedding), PostgreSQL, and AGE graph.
        Returns a dict with at least {"id": str, "node_type": str}.

        If importance is not provided, a sensible default is chosen based on
        the node_type (e.g., Goals=0.8, Decisions=0.7, Learnings=0.65).
        """
        node_id = str(uuid.uuid4())
        tags = tags or []
        extra = extra or {}
        node_name = name or content[:80]
        now = datetime.utcnow().isoformat()

        # Apply per-type default importance when caller doesn't specify
        if importance is None:
            importance = DEFAULT_IMPORTANCE.get(node_type, 0.5)

        # Apply per-type default TTL when not already set in extra
        if "ttl_days" not in extra:
            default_ttl = DEFAULT_TTL_DAYS.get(node_type)
            if default_ttl is not None:
                extra["ttl_days"] = default_ttl

        # Generate embedding for searchable types
        embedding: list[float] | None = None
        if node_type in SEARCHABLE_NODE_TYPES:
            embedding = self.embeddings.encode(content)

        # Dedup: for searchable types, check vector similarity against existing nodes.
        # For non-searchable types (Goal, Task, Project, etc.), check by exact name.
        # Threshold 0.92 catches near-identical re-stores while allowing intentional updates.
        DEDUP_SIMILARITY_THRESHOLD = 0.92
        if embedding is not None:
            try:
                candidates = self.redis.vector_search(
                    query_embedding=embedding,
                    limit=1,
                    min_score=DEDUP_SIMILARITY_THRESHOLD,
                )
                if candidates:
                    existing_id = candidates[0].get("id") or candidates[0].get("$.id")
                    existing_content = candidates[0].get("content", "")
                    if existing_id and candidates[0].get("node_type", node_type) == node_type:
                        logger.info(
                            "Dedup: new content is %.0f%% similar to existing node %s — skipping store",
                            candidates[0].get("similarity", 0) * 100,
                            existing_id,
                        )
                        return {
                            "id": existing_id,
                            "node_type": node_type,
                            "content": existing_content,
                            "deduplicated": True,
                            "similarity": candidates[0].get("similarity"),
                        }
            except Exception as e:
                logger.debug("Dedup similarity check skipped: %s", e)
        elif name:
            # Non-searchable types: check by name match in PostgreSQL
            try:
                existing = self.postgres.find_entity_by_name_and_type(name, node_type)
                if existing:
                    logger.info(
                        "Dedup: node with name '%s' of type '%s' already exists (%s) — skipping store",
                        name, node_type, existing["id"],
                    )
                    return {
                        "id": existing["id"],
                        "node_type": node_type,
                        "content": existing.get("metadata", {}).get("content", name),
                        "deduplicated": True,
                    }
            except Exception as e:
                logger.debug("Dedup name check skipped: %s", e)

        # Build the base document
        doc: dict[str, Any] = {
            "id": node_id,
            "name": node_name,
            "content": content,
            "node_type": node_type,
            "source": source,
            "tags": tags,
            "importance": importance,
            "created_at": now,
            "updated_at": now,
            **extra,
        }

        # 1. Store in Redis (with embedding for search)
        if embedding is not None:
            redis_doc = {**doc, "embedding": embedding}
            self.redis.store_memory(redis_doc)

        # 2. Store in PostgreSQL entities table
        self.postgres.upsert_entity({
            "id": node_id,
            "name": node_name,
            "node_type": node_type,
            "metadata": {k: v for k, v in doc.items() if k not in ("id", "name", "node_type")},
            "tags": tags,
        })

        # 3. Store in AGE graph
        graph_props = {
            "id": node_id,
            "name": node_name,
            "node_type": node_type,
            "content": content,
            "created_at": now,
        }
        # Add extra fields that fit as simple types
        for k, v in extra.items():
            if isinstance(v, (str, int, float, bool)):
                graph_props[k] = v

        self.graph.create_node(node_type, graph_props)

        # Auto-link to similar existing memories (best-effort, non-blocking)
        if node_type in SEARCHABLE_NODE_TYPES:
            try:
                self.auto_link(node_id, content, node_type)
            except Exception as e:
                logger.debug("auto_link skipped: %s", e)

        return {"id": node_id, "node_type": node_type, "content": content}

    def store_project(
        self,
        name: str,
        description: str = "",
        repo_path: str | None = None,
        repo_url: str | None = None,
        stack: list[str] | None = None,
        run_cmd: str | None = None,
        test_cmd: str | None = None,
        deploy_cmd: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Store a Project node with profile metadata."""
        extra: dict[str, Any] = {}
        if repo_path:
            extra["repo_path"] = repo_path
        if repo_url:
            extra["repo_url"] = repo_url
        if stack:
            extra["stack"] = stack
        if run_cmd:
            extra["run_cmd"] = run_cmd
        if test_cmd:
            extra["test_cmd"] = test_cmd
        if deploy_cmd:
            extra["deploy_cmd"] = deploy_cmd

        return self.store(
            content=description or f"Project: {name}",
            node_type="Project",
            name=name,
            tags=tags or [],
            extra=extra,
        )

    def store_goal(
        self,
        name: str,
        description: str = "",
        project_id: str | None = None,
        owner_id: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Store a Goal/OKR node."""
        extra: dict[str, Any] = {}
        if project_id:
            extra["project_id"] = project_id
        if owner_id:
            extra["owner_id"] = owner_id

        result = self.store(
            content=description or f"Goal: {name}",
            node_type="Goal",
            name=name,
            tags=tags or [],
            extra=extra,
        )

        # Link to project if provided
        if project_id:
            try:
                self.relate(result["id"], project_id, "FOR")
            except Exception as e:
                logger.warning("Could not link goal to project: %s", e)

        return result

    def store_initiative(
        self,
        name: str,
        description: str = "",
        goal_id: str | None = None,
        project_id: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Store an Initiative/Campaign node."""
        extra: dict[str, Any] = {}
        if goal_id:
            extra["goal_id"] = goal_id
        if project_id:
            extra["project_id"] = project_id

        result = self.store(
            content=description or f"Initiative: {name}",
            node_type="Initiative",
            name=name,
            tags=tags or [],
            extra=extra,
        )

        if goal_id:
            try:
                self.relate(goal_id, result["id"], "ACHIEVED_VIA")
            except Exception as e:
                logger.warning("Could not link initiative to goal: %s", e)

        return result

    def store_task(
        self,
        name: str,
        description: str = "",
        initiative_id: str | None = None,
        project_id: str | None = None,
        assigned_to: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Store a Task/Todo node."""
        extra: dict[str, Any] = {"status": "todo"}
        if initiative_id:
            extra["initiative_id"] = initiative_id
        if project_id:
            extra["project_id"] = project_id
        if assigned_to:
            extra["assigned_to"] = assigned_to

        result = self.store(
            content=description or f"Task: {name}",
            node_type="Task",
            name=name,
            tags=tags or [],
            extra=extra,
        )

        if initiative_id:
            try:
                self.relate(initiative_id, result["id"], "BROKEN_INTO")
            except Exception as e:
                logger.warning("Could not link task to initiative: %s", e)

        return result

    def store_competitor(
        self,
        name: str,
        website: str | None = None,
        strengths: list[str] | None = None,
        weaknesses: list[str] | None = None,
        positioning: str | None = None,
        project_id: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Store a Competitor node."""
        extra: dict[str, Any] = {}
        if website:
            extra["website"] = website
        if positioning:
            extra["positioning"] = positioning

        result = self.store(
            content=f"Competitor: {name}. {positioning or ''}",
            node_type="Competitor",
            name=name,
            tags=tags or [],
            extra=extra,
        )

        if project_id:
            try:
                self.relate(result["id"], project_id, "COMPETES_WITH")
            except Exception as e:
                logger.warning("Could not link competitor to project: %s", e)

        return result

    def store_metric(
        self,
        name: str,
        metric_type: str,
        value: float,
        unit: str = "",
        project_id: str | None = None,
        source: str = "manual",
        notes: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Store a Metric node and record a data point."""
        # Check if metric already exists
        existing = self.postgres.list_entities_by_type("Metric")
        metric_entity = next(
            (e for e in existing if e["name"] == name), None
        )

        if not metric_entity:
            result = self.store(
                content=f"Metric: {name} ({metric_type})",
                node_type="Metric",
                name=name,
                tags=tags or [],
                extra={"metric_type": metric_type, "unit": unit},
            )
            metric_id = result["id"]
            if project_id:
                try:
                    self.relate(metric_id, project_id, "TRACKS")
                except Exception as e:
                    logger.warning("Could not link metric to project: %s", e)
        else:
            metric_id = metric_entity["id"]

        # Record the data point
        point_id = str(uuid.uuid4())
        self.postgres.insert_metric_point({
            "id": point_id,
            "metric_id": metric_id,
            "value": value,
            "recorded_at": datetime.utcnow().isoformat(),
            "source": source,
            "notes": notes,
        })

        return {"id": metric_id, "node_type": "Metric", "value": value}

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------

    def recall(
        self,
        query: str,
        limit: int = 10,
        node_type: str | None = None,
        tags: list[str] | None = None,
        min_score: float = 0.0,
        anchor_entity_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Semantic recall — find relevant memories using hybrid search with
        graph boost, recency decay, and importance weighting.

        Delegates to HybridRetrieval to ensure all scoring signals are applied.
        Increments access_count on returned nodes for dynamic importance tracking.
        Returns list of memory dicts sorted by final score.
        """
        # Import here to avoid circular import (retrieval imports memory)
        from agentmemory.core.retrieval import HybridRetrieval

        # Reuse an existing HybridRetrieval if attached, otherwise create inline
        if not hasattr(self, "_retrieval") or self._retrieval is None:
            self._retrieval = HybridRetrieval.__new__(HybridRetrieval)
            self._retrieval.memory = self

        results = self._retrieval.retrieve(
            query=query,
            limit=limit,
            tags=tags,
            node_type=node_type,
            anchor_entity_id=anchor_entity_id,
            min_score=min_score,
        )

        # Increment access_count for returned nodes (best-effort, non-blocking)
        for result in results:
            node_id = result.get("id")
            if node_id:
                try:
                    self._increment_access_count(node_id)
                except Exception as e:
                    logger.debug("access_count increment failed for %s: %s", node_id, e)

        return results

    def _increment_access_count(self, node_id: str) -> None:
        """
        Increment the access_count field in PostgreSQL metadata and Redis for a node.

        This is used by the dynamic importance formula: frequently-recalled
        nodes get a small importance boost (capped at 1.5×).
        Both stores are updated so retrieval can read access_count from Redis
        without an extra DB round-trip.
        """
        self.postgres._rollback_if_needed()
        new_count: int | None = None
        try:
            with self.postgres.conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE entities
                    SET metadata = jsonb_set(
                        metadata,
                        '{access_count}',
                        to_jsonb(COALESCE((metadata->>'access_count')::int, 0) + 1)
                    )
                    WHERE id = %s
                    RETURNING (metadata->>'access_count')::int AS new_count
                    """,
                    (node_id,),
                )
                row = cur.fetchone()
                if row:
                    new_count = row["new_count"]
            self.postgres.conn.commit()
        except Exception:
            self.postgres.conn.rollback()
            raise

        # Update Redis JSON (best-effort — only if the key exists)
        if new_count is not None:
            try:
                key = f"{self.redis.prefix}memory:{node_id}"
                self.redis.redis.json().set(key, "$.access_count", new_count)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Auto-link
    # ------------------------------------------------------------------

    def auto_link(
        self,
        node_id: str,
        content: str,
        node_type: str,
        top_k: int = 3,
        min_similarity: float = 0.5,
    ) -> list[str]:
        """
        Automatically create RELATED_TO edges from a newly stored node to
        the most semantically similar existing memories.

        Only links to other searchable node types (Memory, Learning, Decision,
        Preference, Workflow, CustomerFeedback). Skips self-links.

        Args:
            node_id: The ID of the newly stored node.
            content: The content to find similar nodes for.
            node_type: The type of the newly stored node.
            top_k: Maximum number of links to create.
            min_similarity: Minimum cosine similarity threshold (0–1).

        Returns:
            List of IDs that were linked.
        """
        if node_type not in SEARCHABLE_NODE_TYPES:
            return []

        try:
            query_embedding = self.embeddings.encode(content, is_query=True)
            candidates = self.redis.vector_search(
                query_embedding=query_embedding,
                limit=top_k + 1,  # +1 to account for self
                min_score=min_similarity,
            )
        except Exception as e:
            logger.warning("auto_link vector search failed: %s", e)
            return []

        linked = []
        for candidate in candidates:
            cand_id = candidate.get("id") or candidate.get("$.id", "")
            if not cand_id:
                key = candidate.get("_key", "")
                if ":" in key:
                    cand_id = key.split(":")[-1]

            if not cand_id or cand_id == node_id:
                continue

            try:
                self.relate(node_id, cand_id, "RELATED_TO")
                linked.append(cand_id)
                if len(linked) >= top_k:
                    break
            except Exception as e:
                logger.debug("auto_link: could not link %s -> %s: %s", node_id, cand_id, e)

        if linked:
            logger.debug("auto_link: linked %s to %d nodes", node_id, len(linked))

        return linked

    # ------------------------------------------------------------------
    # Relate
    # ------------------------------------------------------------------

    def relate(
        self,
        from_id: str,
        to_id: str,
        edge_type: str,
        properties: dict[str, Any] | None = None,
    ) -> bool:
        """
        Create a relationship between two entities.

        Persists to both AGE graph and PostgreSQL relations table.
        Raises ValueError for invalid edge types.
        """
        # Validate edge type (AGEClient will also validate, but we want clear error)
        if edge_type not in AGEClient.ALLOWED_EDGE_TYPES:
            raise ValueError(
                f"Invalid edge type '{edge_type}'. "
                f"Allowed: {sorted(AGEClient.ALLOWED_EDGE_TYPES)}"
            )

        # Create in graph
        graph_ok = self.graph.create_edge(from_id, to_id, edge_type, properties)

        # Create in PostgreSQL
        try:
            rel_id = str(uuid.uuid4())
            self.postgres.upsert_relation({
                "id": rel_id,
                "from_id": from_id,
                "to_id": to_id,
                "edge_type": edge_type,
                "properties": properties or {},
            })
        except Exception as e:
            logger.warning("Could not persist relation to PostgreSQL: %s", e)

        return graph_ok

    # ------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------

    def context(
        self, entity_id: str, depth: int = 2
    ) -> dict[str, Any] | None:
        """
        Get full context for an entity — all connected nodes + relations.

        Returns dict with entity info, neighbors, and relations.
        """
        entity = self.postgres.get_entity(entity_id)
        if not entity:
            # Try graph
            node = self.graph.get_node(entity_id)
            if not node:
                return None
            entity = node

        neighbors = self.graph.get_neighborhood(entity_id, depth=depth)
        relations = self.postgres.get_relations(from_id=entity_id)
        incoming = self.postgres.get_relations(to_id=entity_id)

        return {
            "entity": entity,
            "neighbors": neighbors,
            "outgoing_relations": relations,
            "incoming_relations": incoming,
        }

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        memory_id: str,
        content: str | None = None,
        name: str | None = None,
        tags: list[str] | None = None,
        importance: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """
        Update an existing memory's content, name, tags, or importance.

        Only the fields you pass are changed — omitted fields are left as-is.
        If content changes, a new embedding is generated and Redis is updated.
        Returns the updated entity dict, or None if not found.
        """
        entity = self.postgres.get_entity(memory_id)
        if not entity:
            return None

        metadata = entity.get("metadata") or {}
        node_type = entity.get("node_type", "Memory")
        now = datetime.utcnow().isoformat()

        # Merge updates
        new_content = content if content is not None else metadata.get("content", "")
        new_name = name if name is not None else entity.get("name", "")
        new_tags = tags if tags is not None else entity.get("tags", [])
        new_importance = importance if importance is not None else metadata.get("importance", 0.5)
        new_extra = {**metadata, **(extra or {})}

        # Update Redis — always re-embed to keep content/metadata in sync
        if node_type in SEARCHABLE_NODE_TYPES:
            embedding = self.embeddings.encode(new_content)
            redis_doc = {
                "id": memory_id,
                "content": new_content,
                "name": new_name,
                "node_type": node_type,
                "tags": new_tags,
                "importance": new_importance,
                "updated_at": now,
                "embedding": embedding,
            }
            self.redis.store_memory(redis_doc)

        # Update Postgres
        updated_metadata = {
            **new_extra,
            "content": new_content,
            "importance": new_importance,
            "updated_at": now,
        }
        self.postgres.upsert_entity({
            "id": memory_id,
            "name": new_name,
            "node_type": node_type,
            "metadata": updated_metadata,
            "tags": new_tags,
        })

        # Update graph node properties (only when name or content actually changed)
        graph_updates: dict[str, Any] = {}
        if name is not None:
            graph_updates["name"] = new_name
        if content is not None:
            graph_updates["content"] = new_content
        if graph_updates:
            self.graph.update_node(memory_id, graph_updates)

        return {
            "id": memory_id,
            "node_type": node_type,
            "name": new_name,
            "content": new_content,
            "tags": new_tags,
            "importance": new_importance,
            "updated_at": now,
        }

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------

    def split(
        self,
        memory_id: str,
        chunks: list[str],
        node_type: str | None = None,
        importance: float | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Split a long/mixed node into multiple focused atomic nodes.

        Each chunk becomes a new node inheriting the original's node_type,
        importance, and tags (unless overridden). The original node's outgoing
        graph edges are re-created from each new node. The original is deleted.

        Args:
            memory_id: ID of the node to split.
            chunks: List of focused content strings (one fact per string).
            node_type: Override node type for all new nodes (default: inherit).
            importance: Override importance for all new nodes (default: inherit).
            tags: Override tags for all new nodes (default: inherit).

        Returns:
            Dict with new_ids list and summary counts.
        """
        entity = self.postgres.get_entity(memory_id)
        if not entity:
            raise ValueError(f"Entity {memory_id} not found")

        meta = entity.get("metadata") or {}
        if isinstance(meta, str):
            import json as _json
            try:
                meta = _json.loads(meta)
            except Exception:
                meta = {}

        inherited_type = node_type or entity.get("node_type", "Memory")
        inherited_importance = importance if importance is not None else meta.get("importance", 0.5)
        inherited_tags = tags if tags is not None else entity.get("tags", [])
        inherited_source = meta.get("source", "mcp")

        # Collect existing outgoing relations to re-attach to new nodes
        existing_relations = self.postgres.get_relations(from_id=memory_id)

        new_ids: list[str] = []
        for chunk in chunks:
            if not chunk.strip():
                continue
            result = self.store(
                content=chunk.strip(),
                node_type=inherited_type,
                tags=inherited_tags,
                source=inherited_source,
                importance=inherited_importance,
            )
            new_id = result["id"]
            new_ids.append(new_id)

            # Re-attach the original's outgoing relations to each new node
            for rel in existing_relations:
                to_id = rel.get("to_id", "")
                edge_type = rel.get("edge_type", "")
                if to_id and edge_type and to_id != memory_id:
                    try:
                        self.relate(new_id, to_id, edge_type)
                    except Exception as e:
                        logger.debug("split: could not re-attach edge %s->%s: %s", new_id, to_id, e)

        # Delete the original node
        self.forget(memory_id)

        return {
            "original_id": memory_id,
            "new_ids": new_ids,
            "chunks_created": len(new_ids),
        }

    # ------------------------------------------------------------------
    # Batch update
    # ------------------------------------------------------------------

    def batch_update(
        self,
        node_type: str | None = None,
        ids: list[str] | None = None,
        importance: float | None = None,
    ) -> dict[str, Any]:
        """
        Bulk-update importance (and optionally other fields) across many nodes.

        Pass either node_type to target all nodes of that type, or ids to
        target specific nodes. At least one of importance must be provided.

        Returns a summary dict with updated/skipped/error counts.
        """
        if importance is None:
            raise ValueError("At least importance must be provided for batch_update")

        if ids:
            entities = [self.postgres.get_entity(eid) for eid in ids]
            entities = [e for e in entities if e]
        elif node_type:
            entities = self.postgres.list_entities_by_type(node_type)
        else:
            raise ValueError("Provide either node_type or ids")

        updated = 0
        skipped = 0
        errors = 0

        for entity in entities:
            entity_id = entity.get("id")
            if not entity_id:
                skipped += 1
                continue
            try:
                self.update(memory_id=entity_id, importance=importance)
                updated += 1
            except Exception as e:
                logger.warning("batch_update: failed to update %s: %s", entity_id, e)
                errors += 1

        return {"updated": updated, "skipped": skipped, "errors": errors, "total": len(entities)}

    # ------------------------------------------------------------------
    # Forget
    # ------------------------------------------------------------------

    def forget(self, memory_id: str) -> bool:
        """
        Remove a memory from all backends.

        Returns True if found and deleted, False if not found.
        """
        # Check if exists
        entity = self.postgres.get_entity(memory_id)
        if not entity:
            return False

        # Delete from Redis
        self.redis.delete_memory(memory_id)

        # Delete from PostgreSQL (cascades to relations)
        self.postgres.delete_entity(memory_id)

        # Delete from graph
        self.graph.delete_node(memory_id)

        return True

    # ------------------------------------------------------------------
    # Entity lookup (for auto-anchor detection)
    # ------------------------------------------------------------------

    def find_entity_by_name(self, name: str) -> dict[str, Any] | None:
        """
        Find the best-matching entity by name (case-insensitive).

        Tries exact match first, then prefix match, then full-text search.
        Returns the entity dict or None if not found.
        """
        self.postgres._rollback_if_needed()
        with self.postgres.conn.cursor() as cur:
            # 1. Exact match (case-insensitive)
            cur.execute(
                "SELECT * FROM entities WHERE LOWER(name) = LOWER(%s) LIMIT 1",
                (name,),
            )
            row = cur.fetchone()
            if row:
                return dict(row)

            # 2. Prefix / contains match
            cur.execute(
                "SELECT * FROM entities WHERE LOWER(name) ILIKE %s ORDER BY LENGTH(name) LIMIT 1",
                (f"%{name.lower()}%",),
            )
            row = cur.fetchone()
            if row:
                return dict(row)

        return None

    # ------------------------------------------------------------------
    # Timeline
    # ------------------------------------------------------------------

    def timeline(
        self,
        since: str | None = None,
        until: str | None = None,
        node_type: str | None = None,
        entity_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Query what happened in a time range.

        Returns entities ordered by created_at descending.
        """
        self.postgres._rollback_if_needed()
        with self.postgres.conn.cursor() as cur:
            conditions = []
            params: list[Any] = []

            if node_type:
                conditions.append("node_type = %s")
                params.append(node_type)
            if since:
                conditions.append("created_at >= %s")
                params.append(since)
            if until:
                conditions.append("created_at <= %s")
                params.append(until)

            where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            params.append(limit)
            cur.execute(
                f"SELECT * FROM entities {where} ORDER BY created_at DESC LIMIT %s",
                params,
            )
            rows = cur.fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return stats from all backends."""
        return {
            "redis": self.redis.stats(),
            "postgres": self.postgres.stats(),
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release all resources."""
        try:
            self.redis.close()
        except Exception:
            pass
        try:
            self.postgres.close()
        except Exception:
            pass
        try:
            self.graph.close()
        except Exception:
            pass
        try:
            self.embeddings.close()
        except Exception:
            pass

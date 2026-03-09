"""
MCP tool handler functions for agentmemory.md.

Each method corresponds to one MCP tool. They take plain Python arguments
(FastMCP handles schema generation from type hints) and return dicts.

All tools implement graceful degradation — they catch exceptions and return
structured error dicts instead of raising, so the agent can inform the user.

Tools:
  Core memory:     memory_store, memory_recall, memory_relate, memory_context, memory_forget
  Work structure:  goal_manage, initiative_manage, task_manage, timeline
  Knowledge:       learning_store, workflow_store
  Market:          competitor_manage, metric_record, metric_query, customer_feedback_store
"""

from __future__ import annotations

import logging
from typing import Any

from agentmemory.core.memory import MemoryService
from agentmemory.core.models import DEFAULT_IMPORTANCE
from agentmemory.core.retrieval import HybridRetrieval
from agentmemory.db.age_client import AGEClient

logger = logging.getLogger(__name__)

VALID_NODE_TYPES = AGEClient.ALLOWED_NODE_TYPES
VALID_EDGE_TYPES = AGEClient.ALLOWED_EDGE_TYPES


def _error(code: str, reason: str, service: str = "memory") -> dict[str, Any]:
    return {"error": code, "reason": reason, "service": service}


class MemoryTools:
    """
    Container for all MCP tool handlers.

    Instantiated once at server startup and shared across all tool calls.
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
        self.database_url = database_url
        self.memory = MemoryService(
            database_url=database_url,
            redis_url=redis_url,
            key_prefix=key_prefix,
            graph_name=graph_name,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
        )
        self.retrieval = HybridRetrieval(
            database_url=database_url,
            redis_url=redis_url,
            key_prefix=key_prefix,
            graph_name=graph_name,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
        )
        # Share the same MemoryService instance to avoid duplicate model loads
        self.retrieval.memory = self.memory

    # ------------------------------------------------------------------
    # Core memory tools
    # ------------------------------------------------------------------

    def memory_store(
        self,
        content: str,
        node_type: str = "Memory",
        name: str | None = None,
        tags: list[str] | None = None,
        source: str = "mcp",
        importance: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Save a memory, learning, decision, or preference.

        Args:
            content: The content to remember.
            node_type: Type of memory (Memory, Learning, Decision, Preference, etc.)
            name: Optional short name/title.
            tags: Optional tags for filtering.
            source: Where this memory came from (mcp, cli, conversation).
            importance: Importance score 0.0–1.0. Defaults to per-type value if not set.
            extra: Additional fields (e.g., rationale for Decision, what_failed for Learning).
        """
        if not content or not content.strip():
            return _error("invalid_input", "content cannot be empty")

        if node_type not in VALID_NODE_TYPES:
            return _error(
                "invalid_node_type",
                f"node_type must be one of: {sorted(VALID_NODE_TYPES)}",
            )

        if not tags:
            logger.warning(
                "memory_store called without tags for node_type=%s — add at least one project tag "
                "(e.g. tags=['busonmap']) to enable project-scoped filtering",
                node_type,
            )

        try:
            result = self.memory.store(
                content=content,
                node_type=node_type,
                name=name,
                tags=tags or [],
                source=source,
                importance=importance,
                extra=extra or {},
            )
            return result
        except Exception as e:
            logger.error("memory_store failed: %s", e)
            return _error("store_failed", str(e))

    def memory_recall(
        self,
        query: str,
        limit: int = 10,
        node_type: str | None = None,
        tags: list[str] | None = None,
        anchor_entity_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Semantic search across all knowledge (vector + graph + keyword hybrid).

        Args:
            query: Natural language query.
            limit: Maximum number of results (default 10).
            node_type: Filter by node type (Memory, Learning, Decision, etc.)
            tags: Filter by tags.
            anchor_entity_id: Boost results connected to this entity.
        """
        if not query or not query.strip():
            return _error("invalid_input", "query cannot be empty")

        try:
            results = self.retrieval.retrieve(
                query=query,
                limit=limit,
                tags=tags,
                node_type=node_type,
                anchor_entity_id=anchor_entity_id,
            )
            return {
                "query": query,
                "results": results,
                "total": len(results),
            }
        except Exception as e:
            logger.error("memory_recall failed: %s", e)
            return _error("recall_failed", str(e))

    def memory_relate(
        self,
        from_id: str,
        to_id: str,
        edge_type: str,
        properties: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Link two entities with a relationship.

        Args:
            from_id: Source entity ID.
            to_id: Target entity ID.
            edge_type: Relationship type (WORKS_ON, ABOUT, BELONGS_TO, etc.)
            properties: Optional edge properties.
        """
        if edge_type not in VALID_EDGE_TYPES:
            return _error(
                "invalid_edge_type",
                f"edge_type must be one of: {sorted(VALID_EDGE_TYPES)}",
            )

        try:
            success = self.memory.relate(from_id, to_id, edge_type, properties)
            return {"success": success, "from_id": from_id, "to_id": to_id, "edge_type": edge_type}
        except ValueError as e:
            return _error("invalid_edge_type", str(e))
        except Exception as e:
            logger.error("memory_relate failed: %s", e)
            return _error("relate_failed", str(e))

    def memory_context(
        self,
        entity_id: str,
        depth: int = 2,
    ) -> dict[str, Any]:
        """
        Get full context for any entity — all connected nodes, relationships, timeline.

        Args:
            entity_id: The entity ID to get context for.
            depth: How many hops to traverse in the graph (default 2).
        """
        try:
            ctx = self.memory.context(entity_id, depth=depth)
            if ctx is None:
                return _error("not_found", f"Entity {entity_id} not found")
            return ctx
        except Exception as e:
            logger.error("memory_context failed: %s", e)
            return _error("context_failed", str(e))

    def memory_entities(
        self,
        node_type: str | None = None,
    ) -> dict[str, Any]:
        """
        List all known entities in the knowledge graph.

        Returns id, node_type, and name for each entity — enough for the AI
        to decide which ones a new memory should be linked to via memory_relate.

        Args:
            node_type: Optional filter (e.g. "Project", "Person", "Decision").
        """
        try:
            import psycopg2
            conn = psycopg2.connect(self.database_url)
            cur = conn.cursor()
            if node_type:
                cur.execute(
                    "SELECT id, node_type, name FROM entities WHERE node_type = %s ORDER BY node_type, name",
                    (node_type,),
                )
            else:
                cur.execute(
                    "SELECT id, node_type, name FROM entities ORDER BY node_type, name"
                )
            rows = cur.fetchall()
            conn.close()
            return {
                "entities": [
                    {"id": r[0], "node_type": r[1], "name": r[2]} for r in rows
                ],
                "count": len(rows),
            }
        except Exception as e:
            logger.error("memory_entities failed: %s", e)
            return _error("entities_failed", str(e))

    def memory_list(
        self,
        node_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
        order_by: str = "created_at",
        order_dir: str = "desc",
    ) -> dict[str, Any]:
        """
        List all nodes in the memory corpus, optionally filtered by type.

        Useful for auditing what's stored, finding stale nodes, or picking
        IDs for memory_split / memory_batch_update.

        Args:
            node_type: Filter by node type (e.g. "Decision", "Goal").
            limit: Max nodes to return (default 50, max 200).
            offset: Pagination offset.
            order_by: Sort field — "created_at" or "importance".
            order_dir: "asc" or "desc".
        """
        limit = min(limit, 200)
        order_by = order_by if order_by in ("created_at", "importance") else "created_at"
        order_dir = "DESC" if order_dir.lower() != "asc" else "ASC"

        try:
            import psycopg2
            conn = psycopg2.connect(self.database_url)
            cur = conn.cursor()

            if node_type:
                cur.execute(
                    f"""
                    SELECT id, node_type, name, tags, created_at,
                           (metadata->>'importance')::float AS importance,
                           (metadata->>'access_count')::int AS access_count
                    FROM entities
                    WHERE node_type = %s
                    ORDER BY {order_by} {order_dir}
                    LIMIT %s OFFSET %s
                    """,
                    (node_type, limit, offset),
                )
            else:
                cur.execute(
                    f"""
                    SELECT id, node_type, name, tags, created_at,
                           (metadata->>'importance')::float AS importance,
                           (metadata->>'access_count')::int AS access_count
                    FROM entities
                    ORDER BY {order_by} {order_dir}
                    LIMIT %s OFFSET %s
                    """,
                    (limit, offset),
                )

            rows = cur.fetchall()
            col_names = [desc[0] for desc in cur.description]
            conn.close()

            nodes = [dict(zip(col_names, row)) for row in rows]
            # Normalize None values
            for n in nodes:
                if n.get("importance") is None:
                    n["importance"] = 0.5
                if n.get("access_count") is None:
                    n["access_count"] = 0
                if n.get("tags") is None:
                    n["tags"] = []

            return {
                "nodes": nodes,
                "count": len(nodes),
                "offset": offset,
                "limit": limit,
                "node_type_filter": node_type,
            }
        except Exception as e:
            logger.error("memory_list failed: %s", e)
            return _error("list_failed", str(e))

    def memory_profile(
        self,
        include_recent: bool = True,
        limit: int = 20,
    ) -> dict[str, Any]:
        """
        Return a user profile summary for bootstrapping a conversation with context.

        Aggregates persistent facts (preferences, active projects, key people, goals)
        and optionally the most recently stored memories.

        Args:
            include_recent: Include the most recently stored memories. Default True.
            limit: Max items per section (default 20).
        """
        try:
            import psycopg2

            conn = psycopg2.connect(self.database_url)
            cur = conn.cursor()

            def _fetch_by_type(node_type: str, n: int) -> list[dict]:
                cur.execute(
                    """
                    SELECT id, node_type, name, tags, created_at,
                           (metadata->>'importance')::float AS importance,
                           metadata->>'content' AS content
                    FROM entities
                    WHERE node_type = %s
                    ORDER BY importance DESC NULLS LAST, created_at DESC
                    LIMIT %s
                    """,
                    (node_type, n),
                )
                rows = cur.fetchall()
                col_names = [desc[0] for desc in cur.description]
                result = []
                for row in rows:
                    node = dict(zip(col_names, row))
                    if node.get("importance") is None:
                        node["importance"] = 0.5
                    if node.get("tags") is None:
                        node["tags"] = []
                    result.append(node)
                return result

            preferences = _fetch_by_type("Preference", limit)
            projects = _fetch_by_type("Project", limit)
            people = _fetch_by_type("Person", limit)
            goals = _fetch_by_type("Goal", limit)

            recent: list[dict] = []
            if include_recent:
                cur.execute(
                    """
                    SELECT id, node_type, name, tags, created_at,
                           (metadata->>'importance')::float AS importance,
                           metadata->>'content' AS content
                    FROM entities
                    WHERE node_type NOT IN ('Metric')
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cur.fetchall()
                col_names = [desc[0] for desc in cur.description]
                for row in rows:
                    node = dict(zip(col_names, row))
                    if node.get("importance") is None:
                        node["importance"] = 0.5
                    if node.get("tags") is None:
                        node["tags"] = []
                    recent.append(node)

            conn.close()

            return {
                "preferences": preferences,
                "projects": projects,
                "people": people,
                "goals": goals,
                "recent": recent,
                "counts": {
                    "preferences": len(preferences),
                    "projects": len(projects),
                    "people": len(people),
                    "goals": len(goals),
                    "recent": len(recent),
                },
            }

        except Exception as e:
            logger.error("memory_profile failed: %s", e)
            return _error("profile_failed", str(e))

    def memory_forget(self, memory_id: str) -> dict[str, Any]:
        """
        Mark a memory as obsolete and remove it from all backends.

        Args:
            memory_id: The ID of the memory to forget.
        """
        try:
            success = self.memory.forget(memory_id)
            if not success:
                return _error("not_found", f"Memory {memory_id} not found")
            return {"success": True, "memory_id": memory_id}
        except Exception as e:
            logger.error("memory_forget failed: %s", e)
            return _error("forget_failed", str(e))

    def memory_split(
        self,
        memory_id: str,
        chunks: list[str],
        node_type: str | None = None,
        importance: float | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Split a long or mixed-topic node into multiple focused atomic nodes.

        Use this when a stored memory contains several unrelated facts — splitting
        it into one-fact-per-node dramatically improves retrieval scores because
        each node's embedding is focused rather than diluted.

        The original node is deleted. Each chunk becomes a new node inheriting
        the original's node_type, importance, tags, and graph edges.

        Args:
            memory_id: ID of the node to split.
            chunks: List of focused content strings (one fact per string, max ~2 sentences each).
            node_type: Override node type for new nodes (default: inherit from original).
            importance: Override importance for new nodes (default: inherit from original).
            tags: Override tags for new nodes (default: inherit from original).
        """
        if not memory_id:
            return _error("invalid_input", "memory_id is required")
        if not chunks or len(chunks) < 2:
            return _error("invalid_input", "chunks must contain at least 2 items")

        try:
            result = self.memory.split(
                memory_id=memory_id,
                chunks=chunks,
                node_type=node_type,
                importance=importance,
                tags=tags,
            )
            return result
        except ValueError as e:
            return _error("not_found", str(e))
        except Exception as e:
            logger.error("memory_split failed: %s", e)
            return _error("split_failed", str(e))

    def memory_batch_update(
        self,
        importance: float,
        node_type: str | None = None,
        ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Bulk-update importance across many nodes at once.

        Use this to fix nodes that were stored with the wrong importance (e.g.,
        all nodes created before per-type defaults were applied have importance=0.5).

        Args:
            importance: New importance value (0.0–1.0).
            node_type: Update all nodes of this type (e.g. "Goal", "Decision").
            ids: Update specific nodes by ID list.
        """
        if not node_type and not ids:
            return _error("invalid_input", "Provide either node_type or ids")

        try:
            result = self.memory.batch_update(
                node_type=node_type,
                ids=ids,
                importance=importance,
            )
            return result
        except Exception as e:
            logger.error("memory_batch_update failed: %s", e)
            return _error("batch_update_failed", str(e))

    def memory_update(
        self,
        memory_id: str,
        content: str | None = None,
        name: str | None = None,
        tags: list[str] | None = None,
        importance: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Update an existing memory's content, name, tags, or importance.

        Only the fields you pass are changed. If content changes, the embedding
        is regenerated and search results will reflect the new content.

        Args:
            memory_id: The ID of the memory to update.
            content: New content (re-embeds if changed).
            name: New display name.
            tags: Replace all tags with this list.
            importance: New importance score (0.0–1.0).
            extra: Additional metadata fields to merge.
        """
        try:
            result = self.memory.update(
                memory_id=memory_id,
                content=content,
                name=name,
                tags=tags,
                importance=importance,
                extra=extra,
            )
            if result is None:
                return _error("not_found", f"Memory {memory_id} not found")
            return result
        except Exception as e:
            logger.error("memory_update failed: %s", e)
            return _error("update_failed", str(e))

    # ------------------------------------------------------------------
    # Work structure tools
    # ------------------------------------------------------------------

    def goal_manage(
        self,
        action: str,
        name: str | None = None,
        description: str = "",
        goal_id: str | None = None,
        project_id: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Create, update, or list goals and OKRs.

        Args:
            action: "create", "list", "get", or "complete".
            name: Goal name (required for create).
            description: Goal description.
            goal_id: Goal ID (required for get/complete).
            project_id: Link to a project.
            tags: Optional tags.
        """
        try:
            if action == "create":
                if not name:
                    return _error("invalid_input", "name is required for create")
                result = self.memory.store_goal(
                    name=name,
                    description=description,
                    project_id=project_id,
                    tags=tags or [],
                )
                return result

            elif action == "list":
                goals = self.memory.postgres.list_entities_by_type("Goal")
                return {"goals": goals, "total": len(goals)}

            elif action == "get":
                if not goal_id:
                    return _error("invalid_input", "goal_id is required for get")
                entity = self.memory.postgres.get_entity(goal_id)
                if not entity:
                    return _error("not_found", f"Goal {goal_id} not found")
                return entity

            elif action == "complete":
                if not goal_id:
                    return _error("invalid_input", "goal_id is required for complete")
                self.memory.postgres.upsert_entity({
                    "id": goal_id,
                    "name": name or goal_id,
                    "node_type": "Goal",
                    "metadata": {"completed": True},
                    "tags": tags or [],
                })
                return {"success": True, "goal_id": goal_id}

            else:
                return _error("invalid_action", f"Unknown action: {action}. Use: create, list, get, complete")

        except Exception as e:
            logger.error("goal_manage failed: %s", e)
            return _error("goal_manage_failed", str(e))

    def initiative_manage(
        self,
        action: str,
        name: str | None = None,
        description: str = "",
        initiative_id: str | None = None,
        goal_id: str | None = None,
        project_id: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Create, update, or list initiatives under goals.

        Args:
            action: "create", "list", or "get".
            name: Initiative name (required for create).
            description: Initiative description.
            initiative_id: Initiative ID (required for get).
            goal_id: Link to a goal.
            project_id: Link to a project.
            tags: Optional tags.
        """
        try:
            if action == "create":
                if not name:
                    return _error("invalid_input", "name is required for create")
                result = self.memory.store_initiative(
                    name=name,
                    description=description,
                    goal_id=goal_id,
                    project_id=project_id,
                    tags=tags or [],
                )
                return result

            elif action == "list":
                initiatives = self.memory.postgres.list_entities_by_type("Initiative")
                return {"initiatives": initiatives, "total": len(initiatives)}

            elif action == "get":
                if not initiative_id:
                    return _error("invalid_input", "initiative_id is required for get")
                entity = self.memory.postgres.get_entity(initiative_id)
                if not entity:
                    return _error("not_found", f"Initiative {initiative_id} not found")
                return entity

            else:
                return _error("invalid_action", f"Unknown action: {action}")

        except Exception as e:
            logger.error("initiative_manage failed: %s", e)
            return _error("initiative_manage_failed", str(e))

    def task_manage(
        self,
        action: str,
        name: str | None = None,
        description: str = "",
        task_id: str | None = None,
        initiative_id: str | None = None,
        project_id: str | None = None,
        assigned_to: str | None = None,
        result_summary: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Create, update, complete, or list tasks.

        Args:
            action: "create", "list", "get", "complete", or "fail".
            name: Task name (required for create).
            description: Task description.
            task_id: Task ID (required for get/complete/fail).
            initiative_id: Link to an initiative.
            project_id: Link to a project.
            assigned_to: Person assigned to this task.
            result_summary: Brief summary of what was done (for complete).
            tags: Optional tags.
        """
        try:
            if action == "create":
                if not name:
                    return _error("invalid_input", "name is required for create")
                result = self.memory.store_task(
                    name=name,
                    description=description,
                    initiative_id=initiative_id,
                    project_id=project_id,
                    assigned_to=assigned_to,
                    tags=tags or [],
                )
                return result

            elif action == "list":
                tasks = self.memory.postgres.list_entities_by_type("Task")
                return {"tasks": tasks, "total": len(tasks)}

            elif action == "get":
                if not task_id:
                    return _error("invalid_input", "task_id is required for get")
                entity = self.memory.postgres.get_entity(task_id)
                if not entity:
                    return _error("not_found", f"Task {task_id} not found")
                return entity

            elif action in ("complete", "fail"):
                if not task_id:
                    return _error("invalid_input", "task_id is required")
                status = "done" if action == "complete" else "failed"
                entity = self.memory.postgres.get_entity(task_id)
                if not entity:
                    return _error("not_found", f"Task {task_id} not found")

                # Update status in postgres
                meta = entity.get("metadata", {})
                if isinstance(meta, str):
                    import json
                    meta = json.loads(meta)
                meta["status"] = status
                if result_summary:
                    meta["result_summary"] = result_summary

                self.memory.postgres.upsert_entity({
                    "id": task_id,
                    "name": entity["name"],
                    "node_type": "Task",
                    "metadata": meta,
                    "tags": entity.get("tags", []),
                })

                # Store a brief audit memory
                if result_summary:
                    self.memory.store(
                        content=f"Task '{entity['name']}' {status}: {result_summary}",
                        node_type="Memory",
                        source="mcp",
                        importance=0.6,
                    )

                return {"success": True, "task_id": task_id, "status": status}

            else:
                return _error("invalid_action", f"Unknown action: {action}")

        except Exception as e:
            logger.error("task_manage failed: %s", e)
            return _error("task_manage_failed", str(e))

    def timeline(
        self,
        since: str | None = None,
        until: str | None = None,
        node_type: str | None = None,
        entity_id: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """
        Query what happened in a time range.

        Args:
            since: ISO datetime string (e.g., "2026-01-01" or "7d" for 7 days ago).
            until: ISO datetime string.
            node_type: Filter by entity type.
            entity_id: Filter by entity.
            limit: Maximum number of events.
        """
        try:
            # Handle relative time like "7d"
            since_dt = _parse_relative_time(since) if since else None
            until_dt = _parse_relative_time(until) if until else None

            events = self.memory.timeline(
                since=since_dt,
                until=until_dt,
                node_type=node_type,
                entity_id=entity_id,
                limit=limit,
            )
            return {"events": events, "total": len(events)}
        except Exception as e:
            logger.error("timeline failed: %s", e)
            return _error("timeline_failed", str(e))

    # ------------------------------------------------------------------
    # Knowledge tools
    # ------------------------------------------------------------------

    def learning_store(
        self,
        content: str,
        what_failed: str,
        why_it_failed: str,
        what_to_avoid: str,
        project_id: str | None = None,
        tags: list[str] | None = None,
        importance: float | None = None,
    ) -> dict[str, Any]:
        """
        Record a failed experiment or lesson learned.

        Args:
            content: Brief description of the learning.
            what_failed: What specifically failed.
            why_it_failed: Root cause.
            what_to_avoid: What to avoid in the future.
            project_id: Optional project this learning is about.
            tags: Optional tags.
            importance: Importance score 0.0–1.0 (default: 0.65 for Learning).
        """
        try:
            result = self.memory.store(
                content=content,
                node_type="Learning",
                tags=tags or [],
                importance=importance,
                extra={
                    "what_failed": what_failed,
                    "why_it_failed": why_it_failed,
                    "what_to_avoid": what_to_avoid,
                },
            )
            if project_id:
                try:
                    self.memory.relate(result["id"], project_id, "ABOUT")
                except Exception:
                    pass
            return result
        except Exception as e:
            logger.error("learning_store failed: %s", e)
            return _error("learning_store_failed", str(e))

    def workflow_store(
        self,
        name: str,
        content: str,
        steps: list[str] | None = None,
        project_id: str | None = None,
        tags: list[str] | None = None,
        importance: float | None = None,
    ) -> dict[str, Any]:
        """
        Save a reusable process or how-to guide.

        Args:
            name: Workflow name (e.g., "Deploy feedback1 to production").
            content: Description of the workflow.
            steps: Ordered list of steps.
            project_id: Optional project this workflow is for.
            tags: Optional tags.
            importance: Importance score 0.0–1.0 (default: 0.65 for Workflow).
        """
        try:
            result = self.memory.store(
                content=content,
                node_type="Workflow",
                name=name,
                tags=tags or [],
                importance=importance,
                extra={"steps": steps or []},
            )
            if project_id:
                try:
                    self.memory.relate(result["id"], project_id, "FOR")
                except Exception:
                    pass
            return result
        except Exception as e:
            logger.error("workflow_store failed: %s", e)
            return _error("workflow_store_failed", str(e))

    # ------------------------------------------------------------------
    # Market intelligence tools
    # ------------------------------------------------------------------

    def competitor_manage(
        self,
        action: str = "list",
        name: str | None = None,
        website: str | None = None,
        positioning: str | None = None,
        strengths: list[str] | None = None,
        weaknesses: list[str] | None = None,
        project_id: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Track competitors and competitive research.

        Args:
            action: "create" or "list".
            name: Competitor name.
            website: Competitor website.
            positioning: How they position themselves.
            strengths: List of strengths.
            weaknesses: List of weaknesses.
            project_id: Which of your projects they compete with.
            tags: Optional tags.
        """
        try:
            if action == "create":
                if not name:
                    return _error("invalid_input", "name is required for create")
                result = self.memory.store_competitor(
                    name=name,
                    website=website,
                    positioning=positioning,
                    project_id=project_id,
                    tags=tags or [],
                )
                return result

            elif action == "list":
                competitors = self.memory.postgres.list_entities_by_type("Competitor")
                return {"competitors": competitors, "total": len(competitors)}

            else:
                return _error("invalid_action", f"Unknown action: {action}")

        except Exception as e:
            logger.error("competitor_manage failed: %s", e)
            return _error("competitor_manage_failed", str(e))

    def metric_record(
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
        """
        Record a metric data point (visitors, trials, revenue, etc.)

        Args:
            name: Metric name (e.g., "website_visitors", "monthly_revenue").
            metric_type: Type of metric (visitors, trials, users, revenue, etc.)
            value: The numeric value.
            unit: Unit of measurement (count, USD, %, etc.)
            project_id: Which project this metric tracks.
            source: Data source (manual, analytics, stripe, etc.)
            notes: Optional notes about this data point.
            tags: Optional tags.
        """
        try:
            result = self.memory.store_metric(
                name=name,
                metric_type=metric_type,
                value=value,
                unit=unit,
                project_id=project_id,
                source=source,
                notes=notes,
                tags=tags or [],
            )
            return result
        except Exception as e:
            logger.error("metric_record failed: %s", e)
            return _error("metric_record_failed", str(e))

    def metric_query(
        self,
        name: str,
        since: str | None = None,
        until: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """
        Query metrics over time.

        Args:
            name: Metric name.
            since: Start date (ISO string or relative like "30d").
            until: End date.
            limit: Maximum data points to return.
        """
        try:
            # Find the metric entity
            metrics = self.memory.postgres.list_entities_by_type("Metric")
            metric = next((m for m in metrics if m["name"] == name), None)
            if not metric:
                return _error("not_found", f"Metric '{name}' not found")

            since_dt = _parse_relative_time(since) if since else None
            until_dt = _parse_relative_time(until) if until else None

            points = self.memory.postgres.get_metric_points(
                metric_id=metric["id"],
                since=since_dt,
                until=until_dt,
                limit=limit,
            )
            return {
                "metric": metric,
                "data_points": points,
                "total": len(points),
            }
        except Exception as e:
            logger.error("metric_query failed: %s", e)
            return _error("metric_query_failed", str(e))

    def customer_feedback_store(
        self,
        content: str,
        project_id: str | None = None,
        contact_id: str | None = None,
        sentiment: str | None = None,
        source: str = "direct",
        tags: list[str] | None = None,
        importance: float | None = None,
    ) -> dict[str, Any]:
        """
        Store customer feedback about a product.

        Args:
            content: The feedback content.
            project_id: Which product/project the feedback is about.
            contact_id: Who submitted the feedback.
            sentiment: positive, negative, or neutral.
            source: Where the feedback came from (direct, feedback1, email).
            tags: Optional tags.
            importance: Importance score 0.0–1.0 (default: 0.6 for CustomerFeedback).
        """
        try:
            extra: dict[str, Any] = {"source": source}
            if sentiment:
                extra["sentiment"] = sentiment
            if contact_id:
                extra["contact_id"] = contact_id
            if project_id:
                extra["project_id"] = project_id

            result = self.memory.store(
                content=content,
                node_type="CustomerFeedback",
                tags=tags or [],
                importance=importance,
                extra=extra,
            )

            if project_id:
                try:
                    self.memory.relate(result["id"], project_id, "ABOUT")
                except Exception:
                    pass
            if contact_id:
                try:
                    self.memory.relate(result["id"], contact_id, "FROM")
                except Exception:
                    pass

            return result
        except Exception as e:
            logger.error("customer_feedback_store failed: %s", e)
            return _error("feedback_store_failed", str(e))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release resources."""
        try:
            self.memory.close()
        except Exception:
            pass


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _parse_relative_time(value: str) -> str | None:
    """
    Parse a relative time string like "7d", "30d", "1h" to ISO datetime.
    Returns ISO string or the original value if it's already a datetime string.
    """
    if not value:
        return None

    import re
    from datetime import datetime, timedelta

    match = re.match(r"^(\d+)([dhm])$", value.strip())
    if match:
        amount = int(match.group(1))
        unit = match.group(2)
        delta = {
            "d": timedelta(days=amount),
            "h": timedelta(hours=amount),
            "m": timedelta(minutes=amount),
        }[unit]
        return (datetime.utcnow() - delta).isoformat()

    return value

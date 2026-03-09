"""
MCP server for agentmemory.md using FastMCP v3.

Exposes 21 tools to any MCP-compatible agent via:
- stdio transport (primary — for agents that spawn this as a child process)
- SSE transport (legacy remote access)
- Streamable HTTP transport (modern remote access — recommended for Cursor / Claude Desktop)

Usage (stdio, for OpenClaw or local agents):
    python -m agentmemory.mcp.server

Usage (Streamable HTTP, for remote access — works with Cursor mcp.json):
    python -m agentmemory.mcp.server --transport streamable-http --port 59999

Usage (SSE, for legacy remote access):
    python -m agentmemory.mcp.server --transport sse --port 59999
"""

from __future__ import annotations

import logging
import os
from typing import Any

import fastmcp

from agentmemory.config import settings
from agentmemory.mcp.tools import MemoryTools

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastMCP server instance
# ---------------------------------------------------------------------------

mcp = fastmcp.FastMCP(
    name="agentmemory",
    instructions=(
        "Persistent memory system for AI agents. "
        "Use memory_store to save important information, "
        "memory_recall to search for relevant context, "
        "memory_profile to get a user profile summary, "
        "memory_context to get full context about an entity, "
        "and memory_relate to link entities together. "
        "Always store key decisions, learnings, and project information."
    ),
)

# Lazy-initialized tools instance (created on first tool call)
_tools: MemoryTools | None = None


def get_tools() -> MemoryTools:
    global _tools
    if _tools is None:
        _tools = MemoryTools(
            database_url=settings.database_url,
            redis_url=settings.redis_url,
            graph_name=settings.graph_name,
            embedding_model=settings.embedding_model,
            embedding_dim=settings.embedding_dim,
        )
    return _tools


# ---------------------------------------------------------------------------
# Core memory tools
# ---------------------------------------------------------------------------


@mcp.tool()
def memory_store(
    content: str,
    node_type: str = "Memory",
    name: str | None = None,
    tags: list[str] | None = None,
    source: str = "mcp",
    importance: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Save a memory, learning, decision, or preference to persistent storage.

    Use this to remember important information across conversations.
    node_type options: Memory, Learning, Decision, Preference, Workflow,
    Project, Goal, Initiative, Task, Person, ExternalContact, Competitor,
    CustomerFeedback, Topic.
    importance: 0.0–1.0 (defaults to per-type value: Goal=0.8, Decision=0.7, etc.)
    """
    return get_tools().memory_store(
        content=content,
        node_type=node_type,
        name=name,
        tags=tags,
        source=source,
        importance=importance,
        extra=extra,
    )


@mcp.tool()
def memory_recall(
    query: str,
    limit: int = 10,
    node_type: str | None = None,
    tags: list[str] | None = None,
    anchor_entity_id: str | None = None,
) -> dict[str, Any]:
    """
    Search memory using semantic search (vector + keyword + graph).

    Returns the most relevant memories for the given query.
    Use anchor_entity_id to boost results connected to a specific entity.
    """
    return get_tools().memory_recall(
        query=query,
        limit=limit,
        node_type=node_type,
        tags=tags,
        anchor_entity_id=anchor_entity_id,
    )


@mcp.tool()
def memory_relate(
    from_id: str,
    to_id: str,
    edge_type: str,
    properties: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Create a relationship between two entities in the knowledge graph.

    Common edge types: WORKS_ON, ABOUT, BELONGS_TO, FOR, OWNS,
    COLLABORATES_ON, RELATED_TO, PREVENTED, AFFECTS, TRACKS, FROM.
    """
    return get_tools().memory_relate(
        from_id=from_id,
        to_id=to_id,
        edge_type=edge_type,
        properties=properties,
    )


@mcp.tool()
def memory_entities(node_type: str | None = None) -> dict[str, Any]:
    """
    List all known entities in the knowledge graph.

    Returns id, node_type, and name for each entity.
    Use this to find which existing entities a new memory should be linked to
    via memory_relate. Optionally filter by node_type (e.g. "Project", "Person").
    """
    return get_tools().memory_entities(node_type=node_type)


@mcp.tool()
def memory_context(
    entity_id: str,
    depth: int = 2,
) -> dict[str, Any]:
    """
    Get full context for any entity — all connected nodes, relationships, and timeline.

    Use this to understand everything known about a project, person, or goal.
    """
    return get_tools().memory_context(entity_id=entity_id, depth=depth)


@mcp.tool()
def memory_forget(memory_id: str) -> dict[str, Any]:
    """
    Remove a memory from all storage backends.

    Use when information is outdated or incorrect.
    """
    return get_tools().memory_forget(memory_id=memory_id)


@mcp.tool()
def memory_list(
    node_type: str | None = None,
    limit: int = 50,
    offset: int = 0,
    order_by: str = "created_at",
    order_dir: str = "desc",
) -> dict[str, Any]:
    """
    List all nodes in the memory corpus.

    Use this to audit what's stored, find stale nodes, or get IDs for
    memory_split or memory_batch_update.

    Args:
        node_type: Optional filter (e.g. "Decision", "Goal", "Memory").
        limit: Max nodes to return (default 50, max 200).
        offset: Pagination offset for large corpora.
        order_by: "created_at" (default) or "importance".
        order_dir: "desc" (newest/highest first) or "asc".
    """
    return get_tools().memory_list(
        node_type=node_type,
        limit=limit,
        offset=offset,
        order_by=order_by,
        order_dir=order_dir,
    )


@mcp.tool()
def memory_update(
    memory_id: str,
    content: str | None = None,
    name: str | None = None,
    tags: list[str] | None = None,
    importance: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Update an existing memory's content, name, tags, or importance.

    Only the fields you pass are changed — omitted fields stay as-is.
    If content changes, the embedding is regenerated automatically.
    Use this instead of forget+store to preserve the entity's ID and graph edges.
    """
    return get_tools().memory_update(
        memory_id=memory_id,
        content=content,
        name=name,
        tags=tags,
        importance=importance,
        extra=extra,
    )


@mcp.tool()
def memory_split(
    memory_id: str,
    chunks: list[str],
    node_type: str | None = None,
    importance: float | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """
    Split a long or mixed-topic node into multiple focused atomic nodes.

    Use when a stored memory contains several unrelated facts — splitting into
    one-fact-per-node dramatically improves retrieval scores because each node's
    embedding is focused rather than diluted across topics.

    The original node is deleted. Each chunk becomes a new node inheriting the
    original's node_type, importance, tags, and graph edges.

    chunks: list of focused content strings, one fact per string (~1-2 sentences each).
    """
    return get_tools().memory_split(
        memory_id=memory_id,
        chunks=chunks,
        node_type=node_type,
        importance=importance,
        tags=tags,
    )


@mcp.tool()
def memory_batch_update(
    importance: float,
    node_type: str | None = None,
    ids: list[str] | None = None,
) -> dict[str, Any]:
    """
    Bulk-update importance across many nodes at once.

    Use this to fix nodes stored with wrong importance (e.g., all nodes created
    before per-type defaults were applied have importance=0.5).
    Provide either node_type to update all nodes of that type, or ids for specific nodes.

    Examples:
    - Bump all Goals: memory_batch_update(importance=0.8, node_type="Goal")
    - Bump specific nodes: memory_batch_update(importance=0.9, ids=["uuid1", "uuid2"])
    """
    return get_tools().memory_batch_update(
        importance=importance,
        node_type=node_type,
        ids=ids,
    )


@mcp.tool()
def memory_profile(
    include_recent: bool = True,
    limit: int = 20,
) -> dict[str, Any]:
    """
    Get a user profile summary to bootstrap a conversation with context.

    Returns persistent facts grouped by type: preferences, active projects,
    key people, goals, and optionally the most recently stored memories.

    Call this at the start of a session alongside memory_recall and goal_manage
    to get the full picture of who the user is and what they're working on.

    include_recent: Include the most recently stored memories (default True).
    limit: Max items per section (default 20).
    """
    return get_tools().memory_profile(
        include_recent=include_recent,
        limit=limit,
    )


# ---------------------------------------------------------------------------
# Work structure tools
# ---------------------------------------------------------------------------


@mcp.tool()
def goal_manage(
    action: str,
    name: str | None = None,
    description: str = "",
    goal_id: str | None = None,
    project_id: str | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """
    Create, update, or list goals and OKRs.

    action: "create", "list", "get", or "complete".
    """
    return get_tools().goal_manage(
        action=action,
        name=name,
        description=description,
        goal_id=goal_id,
        project_id=project_id,
        tags=tags,
    )


@mcp.tool()
def initiative_manage(
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

    action: "create", "list", or "get".
    """
    return get_tools().initiative_manage(
        action=action,
        name=name,
        description=description,
        initiative_id=initiative_id,
        goal_id=goal_id,
        project_id=project_id,
        tags=tags,
    )


@mcp.tool()
def task_manage(
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

    action: "create", "list", "get", "complete", or "fail".
    """
    return get_tools().task_manage(
        action=action,
        name=name,
        description=description,
        task_id=task_id,
        initiative_id=initiative_id,
        project_id=project_id,
        assigned_to=assigned_to,
        result_summary=result_summary,
        tags=tags,
    )


@mcp.tool()
def timeline(
    since: str | None = None,
    until: str | None = None,
    node_type: str | None = None,
    entity_id: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """
    Query what happened in a time range.

    since/until: ISO datetime or relative (e.g., "7d" for 7 days ago).
    """
    return get_tools().timeline(
        since=since,
        until=until,
        node_type=node_type,
        entity_id=entity_id,
        limit=limit,
    )


# ---------------------------------------------------------------------------
# Knowledge tools
# ---------------------------------------------------------------------------


@mcp.tool()
def learning_store(
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

    Use this to prevent repeating mistakes. The learning will be retrievable
    when similar situations arise.
    importance: 0.0–1.0 (default: 0.65 for Learning nodes).
    """
    return get_tools().learning_store(
        content=content,
        what_failed=what_failed,
        why_it_failed=why_it_failed,
        what_to_avoid=what_to_avoid,
        project_id=project_id,
        tags=tags,
        importance=importance,
    )


@mcp.tool()
def workflow_store(
    name: str,
    content: str,
    steps: list[str] | None = None,
    project_id: str | None = None,
    tags: list[str] | None = None,
    importance: float | None = None,
) -> dict[str, Any]:
    """
    Save a reusable process or how-to guide.

    Use this to remember how to do recurring tasks (e.g., deploy, test, release).
    importance: 0.0–1.0 (default: 0.65 for Workflow nodes).
    """
    return get_tools().workflow_store(
        name=name,
        content=content,
        steps=steps,
        project_id=project_id,
        tags=tags,
        importance=importance,
    )


# ---------------------------------------------------------------------------
# Market intelligence tools
# ---------------------------------------------------------------------------


@mcp.tool()
def competitor_manage(
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
    Track competitors and competitive intelligence.

    action: "create" or "list".
    """
    return get_tools().competitor_manage(
        action=action,
        name=name,
        website=website,
        positioning=positioning,
        strengths=strengths,
        weaknesses=weaknesses,
        project_id=project_id,
        tags=tags,
    )


@mcp.tool()
def metric_record(
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
    Record a metric data point (visitors, trials, users, revenue, etc.)

    metric_type examples: visitors, trials, users, revenue, churn, nps.
    """
    return get_tools().metric_record(
        name=name,
        metric_type=metric_type,
        value=value,
        unit=unit,
        project_id=project_id,
        source=source,
        notes=notes,
        tags=tags,
    )


@mcp.tool()
def metric_query(
    name: str,
    since: str | None = None,
    until: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    """
    Query a metric's historical data points.

    since/until: ISO datetime or relative (e.g., "30d" for last 30 days).
    """
    return get_tools().metric_query(name=name, since=since, until=until, limit=limit)


@mcp.tool()
def customer_feedback_store(
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

    sentiment: "positive", "negative", or "neutral".
    source: "direct", "feedback1", "email", "support".
    importance: 0.0–1.0 (default: 0.6 for CustomerFeedback nodes).
    """
    return get_tools().customer_feedback_store(
        content=content,
        project_id=project_id,
        contact_id=contact_id,
        sentiment=sentiment,
        source=source,
        tags=tags,
        importance=importance,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the MCP server (stdio by default, SSE or Streamable HTTP for remote access)."""
    import argparse

    parser = argparse.ArgumentParser(description="agentmemory.md MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Transport mode (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default=settings.mcp_host,
        help="Host for SSE/HTTP transport",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.mcp_port,
        help="Port for SSE/HTTP transport",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.transport == "sse":
        logger.info("Starting MCP server in SSE mode on %s:%d", args.host, args.port)
        mcp.run(transport="sse", host=args.host, port=args.port)
    elif args.transport == "streamable-http":
        logger.info("Starting MCP server in Streamable HTTP mode on %s:%d", args.host, args.port)
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
    else:
        logger.info("Starting MCP server in stdio mode")
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

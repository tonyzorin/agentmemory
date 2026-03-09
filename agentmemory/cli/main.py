"""
agentmemory.md CLI — manual memory management.

Usage:
    memory store "Anton prefers Claude for coding" --tags preferences,tools
    memory recall "what does Anton prefer for coding"
    memory learn "psycopg2-binary doesn't work on Python 3.14" --project feedback1
    memory decide "Use Redis 8.4 for vector search" --rationale "proven in feedback1"
    memory goal create "Launch Feedback1 GTM" --project feedback1
    memory initiative create "MCP for customers" --goal <goal-id>
    memory task create "Build SSE endpoint" --initiative <init-id>
    memory task done <task-id>
    memory relate "Anton" --works-on "feedback1"
    memory context "feedback1"
    memory timeline --since 7d --about feedback1
    memory stats
    memory export --format json > backup.json
    memory import backup.json
    memory forget <memory-id>
"""

from __future__ import annotations

import json
import sys
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

console = Console()


def get_tools():
    """Lazy-load MemoryTools with settings from environment."""
    from agentmemory.config import settings
    from agentmemory.mcp.tools import MemoryTools

    return MemoryTools(
        database_url=settings.database_url,
        redis_url=settings.redis_url,
        graph_name=settings.graph_name,
        embedding_model=settings.embedding_model,
    )


def _print_result(result: dict[str, Any], title: str = "") -> None:
    """Print a result dict nicely."""
    if "error" in result:
        console.print(f"[red]Error:[/red] {result['error']}: {result.get('reason', '')}")
        sys.exit(1)
    if title:
        console.print(f"[green]✓[/green] {title}")
    if "id" in result:
        console.print(f"  ID: [cyan]{result['id']}[/cyan]")


# ---------------------------------------------------------------------------
# Main CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(version="0.1.0", prog_name="memory")
def cli():
    """agentmemory.md — store, recall, and manage knowledge across sessions."""
    pass


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("content")
@click.option("--type", "node_type", default="Memory", help="Node type (Memory, Decision, etc.)")
@click.option("--name", default=None, help="Short name/title")
@click.option("--tags", default="", help="Comma-separated tags")
@click.option("--importance", default=None, type=float, help="Importance 0.0–1.0 (default: per-type, e.g. Decision=0.7, Goal=0.8)")
@click.option("--source", default="cli", help="Source of this memory")
def store(content, node_type, name, tags, importance, source):
    """Store a memory, learning, decision, or preference."""
    tools = get_tools()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    result = tools.memory_store(
        content=content,
        node_type=node_type,
        name=name,
        tags=tag_list,
        source=source,
        importance=importance,
    )
    _print_result(result, f"Stored {node_type}")


# ---------------------------------------------------------------------------
# recall
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("query")
@click.option("--limit", default=10, type=int, help="Max results")
@click.option("--type", "node_type", default=None, help="Filter by node type")
@click.option("--tags", default="", help="Comma-separated tag filter")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def recall(query, limit, node_type, tags, as_json):
    """Search memory using semantic search."""
    tools = get_tools()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    result = tools.memory_recall(
        query=query,
        limit=limit,
        node_type=node_type,
        tags=tag_list or None,
    )

    if "error" in result:
        console.print(f"[red]Error:[/red] {result['error']}")
        sys.exit(1)

    results = result.get("results", [])

    if as_json:
        click.echo(json.dumps(results, indent=2, default=str))
        return

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(title=f"Results for: {query}", show_lines=True)
    table.add_column("Score", style="cyan", width=6)
    table.add_column("Type", style="magenta", width=12)
    table.add_column("Content", style="white", max_width=60)
    table.add_column("ID", style="dim", width=36)

    for r in results:
        content_text = r.get("content", r.get("name", ""))
        if isinstance(content_text, bytes):
            content_text = content_text.decode()
        table.add_row(
            f"{r.get('score', 0):.2f}",
            str(r.get("node_type", "Memory")),
            str(content_text)[:120],
            str(r.get("id", "")),
        )

    console.print(table)
    console.print(f"[dim]Total: {result.get('total', len(results))} results[/dim]")


# ---------------------------------------------------------------------------
# learn
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("content")
@click.option("--what-failed", default="", help="What specifically failed")
@click.option("--why", default="", help="Root cause")
@click.option("--avoid", default="", help="What to avoid in the future")
@click.option("--project", default=None, help="Project ID this learning is about")
@click.option("--tags", default="", help="Comma-separated tags")
@click.option("--importance", default=None, type=float, help="Importance 0.0–1.0 (default: 0.65)")
def learn(content, what_failed, why, avoid, project, tags, importance):
    """Record a failed experiment or lesson learned."""
    tools = get_tools()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    result = tools.learning_store(
        content=content,
        what_failed=what_failed or content,
        why_it_failed=why or "Unknown",
        what_to_avoid=avoid or content,
        project_id=project,
        tags=tag_list,
        importance=importance,
    )
    _print_result(result, "Learning recorded")


# ---------------------------------------------------------------------------
# decide
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("content")
@click.option("--rationale", default="", help="Why this decision was made")
@click.option("--affects", default=None, help="Project/entity ID this affects")
@click.option("--tags", default="", help="Comma-separated tags")
@click.option("--importance", default=None, type=float, help="Importance 0.0–1.0 (default: 0.7)")
def decide(content, rationale, affects, tags, importance):
    """Record a key decision with rationale."""
    tools = get_tools()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    result = tools.memory_store(
        content=content,
        node_type="Decision",
        tags=tag_list,
        importance=importance,
        extra={"rationale": rationale},
    )
    if "error" not in result and affects:
        tools.memory_relate(result["id"], affects, "AFFECTS")
    _print_result(result, "Decision recorded")


# ---------------------------------------------------------------------------
# goal group
# ---------------------------------------------------------------------------


@cli.group()
def goal():
    """Manage goals and OKRs."""
    pass


@goal.command("create")
@click.argument("name")
@click.option("--description", default="", help="Goal description")
@click.option("--project", default=None, help="Project ID")
@click.option("--tags", default="", help="Comma-separated tags")
def goal_create(name, description, project, tags):
    """Create a new goal or OKR."""
    tools = get_tools()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    result = tools.goal_manage(
        action="create",
        name=name,
        description=description,
        project_id=project,
        tags=tag_list,
    )
    _print_result(result, f"Goal created: {name}")


@goal.command("list")
@click.option("--json", "as_json", is_flag=True)
def goal_list(as_json):
    """List all goals."""
    tools = get_tools()
    result = tools.goal_manage(action="list")
    if as_json:
        click.echo(json.dumps(result, indent=2, default=str))
        return
    goals = result.get("goals", [])
    if not goals:
        console.print("[yellow]No goals found.[/yellow]")
        return
    table = Table(title="Goals", show_lines=True)
    table.add_column("Name", style="cyan")
    table.add_column("ID", style="dim", width=36)
    for g in goals:
        table.add_row(g.get("name", ""), g.get("id", ""))
    console.print(table)


# ---------------------------------------------------------------------------
# initiative group
# ---------------------------------------------------------------------------


@cli.group()
def initiative():
    """Manage initiatives and campaigns."""
    pass


@initiative.command("create")
@click.argument("name")
@click.option("--description", default="")
@click.option("--goal", "goal_id", default=None, help="Goal ID")
@click.option("--project", default=None, help="Project ID")
@click.option("--tags", default="")
def initiative_create(name, description, goal_id, project, tags):
    """Create a new initiative under a goal."""
    tools = get_tools()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    result = tools.initiative_manage(
        action="create",
        name=name,
        description=description,
        goal_id=goal_id,
        project_id=project,
        tags=tag_list,
    )
    _print_result(result, f"Initiative created: {name}")


@initiative.command("list")
def initiative_list():
    """List all initiatives."""
    tools = get_tools()
    result = tools.initiative_manage(action="list")
    initiatives = result.get("initiatives", [])
    if not initiatives:
        console.print("[yellow]No initiatives found.[/yellow]")
        return
    table = Table(title="Initiatives")
    table.add_column("Name", style="cyan")
    table.add_column("ID", style="dim")
    for i in initiatives:
        table.add_row(i.get("name", ""), i.get("id", ""))
    console.print(table)


# ---------------------------------------------------------------------------
# task group
# ---------------------------------------------------------------------------


@cli.group()
def task():
    """Manage tasks and todos."""
    pass


@task.command("create")
@click.argument("name")
@click.option("--description", default="")
@click.option("--initiative", "initiative_id", default=None)
@click.option("--project", default=None)
@click.option("--assign", default=None, help="Assign to person ID")
@click.option("--tags", default="")
def task_create(name, description, initiative_id, project, assign, tags):
    """Create a new task."""
    tools = get_tools()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    result = tools.task_manage(
        action="create",
        name=name,
        description=description,
        initiative_id=initiative_id,
        project_id=project,
        assigned_to=assign,
        tags=tag_list,
    )
    _print_result(result, f"Task created: {name}")


@task.command("done")
@click.argument("task_id")
@click.option("--summary", default=None, help="Brief summary of what was done")
def task_done(task_id, summary):
    """Mark a task as done."""
    tools = get_tools()
    result = tools.task_manage(
        action="complete",
        task_id=task_id,
        result_summary=summary,
    )
    _print_result(result, "Task completed")


@task.command("list")
@click.option("--json", "as_json", is_flag=True)
def task_list(as_json):
    """List all tasks."""
    tools = get_tools()
    result = tools.task_manage(action="list")
    if as_json:
        click.echo(json.dumps(result, indent=2, default=str))
        return
    tasks = result.get("tasks", [])
    if not tasks:
        console.print("[yellow]No tasks found.[/yellow]")
        return
    table = Table(title="Tasks")
    table.add_column("Name", style="cyan")
    table.add_column("Status", style="yellow")
    table.add_column("ID", style="dim")
    for t in tasks:
        meta = t.get("metadata", {})
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        status = meta.get("status", "todo") if isinstance(meta, dict) else "todo"
        table.add_row(t.get("name", ""), status, t.get("id", ""))
    console.print(table)


# ---------------------------------------------------------------------------
# relate
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("from_id")
@click.argument("to_id")
@click.argument("edge_type")
def relate(from_id, to_id, edge_type):
    """Create a relationship between two entities.

    Example: memory relate <person-id> <project-id> WORKS_ON
    """
    tools = get_tools()
    result = tools.memory_relate(from_id=from_id, to_id=to_id, edge_type=edge_type)
    if "error" in result:
        console.print(f"[red]Error:[/red] {result['error']}: {result.get('reason', '')}")
        sys.exit(1)
    console.print(f"[green]✓[/green] Relationship created: {from_id} -[{edge_type}]-> {to_id}")


# ---------------------------------------------------------------------------
# context
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("entity_id")
@click.option("--depth", default=2, type=int, help="Graph traversal depth")
@click.option("--json", "as_json", is_flag=True)
def context(entity_id, depth, as_json):
    """Get full context for an entity — all connected nodes and relationships."""
    tools = get_tools()
    result = tools.memory_context(entity_id=entity_id, depth=depth)

    if "error" in result:
        console.print(f"[red]Error:[/red] {result['error']}: {result.get('reason', '')}")
        sys.exit(1)

    if as_json:
        click.echo(json.dumps(result, indent=2, default=str))
        return

    entity = result.get("entity", {})
    console.print(Panel(
        f"[bold]{entity.get('name', entity_id)}[/bold]\n"
        f"Type: {entity.get('node_type', 'Unknown')}\n"
        f"ID: {entity_id}",
        title="Entity",
    ))

    neighbors = result.get("neighbors", [])
    if neighbors:
        table = Table(title=f"Connected nodes (depth={depth})")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("ID", style="dim")
        for n in neighbors[:20]:
            table.add_row(
                str(n.get("name", "")),
                str(n.get("node_type", "")),
                str(n.get("id", "")),
            )
        console.print(table)

    relations = result.get("outgoing_relations", [])
    if relations:
        console.print(f"\n[dim]Outgoing relations: {len(relations)}[/dim]")


# ---------------------------------------------------------------------------
# timeline
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--since", default=None, help="Start time (ISO or relative like 7d)")
@click.option("--until", default=None, help="End time")
@click.option("--type", "node_type", default=None, help="Filter by node type")
@click.option("--limit", default=20, type=int)
@click.option("--json", "as_json", is_flag=True)
def timeline(since, until, node_type, limit, as_json):
    """Show what happened in a time range."""
    tools = get_tools()
    result = tools.timeline(since=since, until=until, node_type=node_type, limit=limit)

    if "error" in result:
        console.print(f"[red]Error:[/red] {result['error']}")
        sys.exit(1)

    events = result.get("events", [])
    if as_json:
        click.echo(json.dumps(events, indent=2, default=str))
        return

    if not events:
        console.print("[yellow]No events found.[/yellow]")
        return

    table = Table(title="Timeline", show_lines=True)
    table.add_column("When", style="dim", width=20)
    table.add_column("Type", style="magenta", width=14)
    table.add_column("Name", style="cyan", max_width=50)
    table.add_column("ID", style="dim", width=36)

    for e in events:
        created = str(e.get("created_at", ""))[:19]
        table.add_row(
            created,
            str(e.get("node_type", "")),
            str(e.get("name", ""))[:60],
            str(e.get("id", "")),
        )
    console.print(table)


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("memory_id")
@click.option("--content", "-c", default=None, help="New content (re-embeds if changed)")
@click.option("--name", "-n", default=None, help="New display name")
@click.option("--tags", default=None, help="Replace tags (comma-separated)")
@click.option("--importance", "-i", type=float, default=None, help="New importance (0.0–1.0)")
def update(memory_id, content, name, tags, importance):
    """Update an existing memory's content, name, tags, or importance."""
    if not any([content, name, tags, importance is not None]):
        console.print("[yellow]Nothing to update — pass at least one of --content, --name, --tags, --importance[/yellow]")
        sys.exit(1)
    tag_list = [t.strip() for t in tags.split(",")] if tags else None
    tools = get_tools()
    result = tools.memory_update(
        memory_id=memory_id,
        content=content,
        name=name,
        tags=tag_list,
        importance=importance,
    )
    if "error" in result:
        console.print(f"[red]Error:[/red] {result['error']}: {result.get('reason', '')}")
        sys.exit(1)
    console.print(f"[green]✓[/green] Memory updated")
    console.print(f"  ID: [cyan]{result['id']}[/cyan]")
    if content:
        console.print(f"  Content: {result['content'][:80]}{'...' if len(result['content']) > 80 else ''}")
    if importance is not None:
        console.print(f"  Importance: {result['importance']}")


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("memory_id")
@click.option("--chunk", "chunks", multiple=True, required=True,
              help="A focused content chunk (repeat for each chunk, min 2)")
@click.option("--type", "node_type", default=None, help="Override node type for new nodes")
@click.option("--importance", "-i", type=float, default=None, help="Override importance for new nodes")
@click.option("--tags", default=None, help="Override tags (comma-separated)")
def split(memory_id, chunks, node_type, importance, tags):
    """Split a long node into multiple focused atomic nodes.

    The original node is deleted. Each --chunk becomes a new node inheriting
    the original's type, importance, tags, and graph edges.

    \b
    Example:
      memory split <id> \\
        --chunk "Anton worked at MSP360 as Head of Product 2019-2022" \\
        --chunk "Anton led EMEL grant application for busonmap in 2026" \\
        --chunk "Anton's career focus is B2B SaaS and public transport tech"
    """
    if len(chunks) < 2:
        console.print("[red]Error:[/red] Provide at least 2 --chunk values")
        sys.exit(1)

    tag_list = [t.strip() for t in tags.split(",")] if tags else None
    tools = get_tools()
    result = tools.memory_split(
        memory_id=memory_id,
        chunks=list(chunks),
        node_type=node_type,
        importance=importance,
        tags=tag_list,
    )
    if "error" in result:
        console.print(f"[red]Error:[/red] {result['error']}: {result.get('reason', '')}")
        sys.exit(1)
    console.print(f"[green]✓[/green] Split into {result['chunks_created']} focused nodes")
    for new_id in result.get("new_ids", []):
        console.print(f"  [cyan]{new_id}[/cyan]")


# ---------------------------------------------------------------------------
# batch-update
# ---------------------------------------------------------------------------


@cli.command("batch-update")
@click.option("--type", "node_type", default=None, help="Update all nodes of this type (e.g. Goal, Decision)")
@click.option("--ids", default=None, help="Comma-separated list of specific node IDs to update")
@click.option("--importance", "-i", type=float, required=True, help="New importance value (0.0–1.0)")
@click.option("--dry-run", is_flag=True, help="Show what would be updated without writing")
def batch_update(node_type, ids, importance, dry_run):
    """Bulk-update importance across many nodes at once.

    Examples:

    \b
    # Bump all Goals to 0.8
    memory batch-update --type Goal --importance 0.8

    \b
    # Bump specific nodes
    memory batch-update --ids uuid1,uuid2 --importance 0.9

    \b
    # Preview without writing
    memory batch-update --type Decision --importance 0.7 --dry-run
    """
    if not node_type and not ids:
        console.print("[red]Error:[/red] Provide --type or --ids")
        sys.exit(1)

    from agentmemory.config import settings
    from agentmemory.core.memory import MemoryService

    svc = MemoryService(
        database_url=settings.database_url,
        redis_url=settings.redis_url,
        graph_name=settings.graph_name,
        embedding_model=settings.embedding_model,
        embedding_dim=settings.embedding_dim,
    )

    id_list = [i.strip() for i in ids.split(",")] if ids else None

    if dry_run:
        if id_list:
            entities = [svc.postgres.get_entity(eid) for eid in id_list]
            entities = [e for e in entities if e]
        else:
            entities = svc.postgres.list_entities_by_type(node_type)
        svc.close()
        console.print(f"[yellow]Dry run:[/yellow] Would update {len(entities)} nodes → importance={importance}")
        for e in entities[:20]:
            meta = e.get("metadata", {})
            if isinstance(meta, str):
                import json as _json
                try:
                    meta = _json.loads(meta)
                except Exception:
                    meta = {}
            current = meta.get("importance", "?")
            console.print(f"  {e.get('node_type', '?'):16} {e.get('name', '')[:50]:50} {current} → {importance}")
        if len(entities) > 20:
            console.print(f"  [dim]... and {len(entities) - 20} more[/dim]")
        return

    result = svc.batch_update(node_type=node_type, ids=id_list, importance=importance)
    svc.close()

    console.print(
        f"[green]✓[/green] Updated {result['updated']} nodes to importance={importance}"
        + (f" ({result['errors']} errors)" if result["errors"] else "")
    )


# ---------------------------------------------------------------------------
# forget
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("memory_id")
@click.option("--yes", is_flag=True, help="Skip confirmation")
def forget(memory_id, yes):
    """Remove a memory from all storage backends."""
    if not yes:
        click.confirm(f"Forget memory {memory_id}?", abort=True)
    tools = get_tools()
    result = tools.memory_forget(memory_id=memory_id)
    if "error" in result:
        console.print(f"[red]Error:[/red] {result['error']}: {result.get('reason', '')}")
        sys.exit(1)
    console.print(f"[green]✓[/green] Memory {memory_id} forgotten")


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--no-recent", is_flag=True, help="Skip recent memories section")
@click.option("--limit", default=20, show_default=True, help="Max items per section")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def profile(no_recent, limit, as_json):
    """Show user profile: preferences, projects, people, goals, and recent memories."""
    tools = get_tools()
    result = tools.memory_profile(include_recent=not no_recent, limit=limit)

    if "error" in result:
        console.print(f"[red]Error:[/red] {result['error']}: {result.get('reason', '')}")
        sys.exit(1)

    if as_json:
        click.echo(json.dumps(result, indent=2, default=str))
        return

    def _print_section(title: str, items: list, name_key: str = "name", content_key: str = "content") -> None:
        if not items:
            console.print(f"[dim]  (none)[/dim]")
            return
        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
        table.add_column("Name / Content", style="cyan", ratio=3)
        table.add_column("Tags", style="dim", ratio=1)
        table.add_column("Importance", justify="right", ratio=1)
        for item in items:
            name = item.get(name_key) or item.get(content_key) or item.get("id", "")
            tags = ", ".join(item.get("tags") or [])
            imp = f"{item.get('importance', 0.5):.2f}"
            table.add_row(str(name)[:80], tags, imp)
        console.print(table)

    counts = result.get("counts", {})
    console.print(Panel(
        f"Preferences: {counts.get('preferences', 0)}  "
        f"Projects: {counts.get('projects', 0)}  "
        f"People: {counts.get('people', 0)}  "
        f"Goals: {counts.get('goals', 0)}  "
        f"Recent: {counts.get('recent', 0)}",
        title="[bold]agentmemory.md — User Profile[/bold]",
    ))

    console.print("\n[bold yellow]Preferences[/bold yellow]")
    _print_section("Preferences", result.get("preferences", []))

    console.print("\n[bold yellow]Active Projects[/bold yellow]")
    _print_section("Projects", result.get("projects", []))

    console.print("\n[bold yellow]People[/bold yellow]")
    _print_section("People", result.get("people", []))

    console.print("\n[bold yellow]Goals[/bold yellow]")
    _print_section("Goals", result.get("goals", []))

    if not no_recent:
        console.print("\n[bold yellow]Recent Memories[/bold yellow]")
        _print_section("Recent", result.get("recent", []))


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--json", "as_json", is_flag=True)
def stats(as_json):
    """Show database statistics."""
    from agentmemory.config import settings
    from agentmemory.core.memory import MemoryService

    svc = MemoryService(
        database_url=settings.database_url,
        redis_url=settings.redis_url,
        graph_name=settings.graph_name,
        embedding_model=settings.embedding_model,
    )
    result = svc.stats()
    svc.close()

    if as_json:
        click.echo(json.dumps(result, indent=2, default=str))
        return

    console.print(Panel(
        f"[bold]Redis[/bold]\n"
        f"  Memories: {result['redis']['memory_count']}\n"
        f"  Index: {result['redis']['index_name']}\n\n"
        f"[bold]PostgreSQL[/bold]\n"
        f"  Total entities: {result['postgres']['total_entities']}\n"
        + "\n".join(
            f"  {k}: {v}" for k, v in result["postgres"].get("by_type", {}).items()
        ),
        title="Memory System Stats",
    ))


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


@cli.command("export")
@click.option("--format", "fmt", default="json", help="Export format (json)")
@click.option("--output", "-o", default=None, help="Output file (default: stdout)")
def export_cmd(fmt, output):
    """Export all memories to JSON."""
    from agentmemory.config import settings
    from agentmemory.db.postgres import PostgresClient

    pg = PostgresClient(dsn=settings.database_url)

    with pg.conn.cursor() as cur:
        cur.execute("SELECT * FROM entities ORDER BY created_at")
        entities = [dict(r) for r in cur.fetchall()]
        cur.execute("SELECT * FROM relations ORDER BY created_at")
        relations = [dict(r) for r in cur.fetchall()]

    pg.close()

    data = {
        "version": "0.1.0",
        "entities": entities,
        "relations": relations,
    }

    output_str = json.dumps(data, indent=2, default=str)
    if output:
        with open(output, "w") as f:
            f.write(output_str)
        console.print(f"[green]✓[/green] Exported to {output}")
    else:
        click.echo(output_str)


# ---------------------------------------------------------------------------
# import
# ---------------------------------------------------------------------------


@cli.command("import")
@click.argument("filepath")
@click.option("--yes", is_flag=True, help="Skip confirmation")
def import_cmd(filepath, yes):
    """Import memories from a JSON export file."""
    from agentmemory.config import settings
    from agentmemory.db.postgres import PostgresClient

    with open(filepath) as f:
        data = json.load(f)

    entities = data.get("entities", [])
    relations = data.get("relations", [])

    if not yes:
        click.confirm(
            f"Import {len(entities)} entities and {len(relations)} relations?",
            abort=True,
        )

    pg = PostgresClient(dsn=settings.database_url)
    imported = 0
    errors = 0

    for entity in entities:
        try:
            pg.upsert_entity({
                "id": entity["id"],
                "name": entity["name"],
                "node_type": entity["node_type"],
                "metadata": entity.get("metadata", {}),
                "tags": entity.get("tags", []),
            })
            imported += 1
        except Exception as e:
            errors += 1
            logger.warning("Failed to import entity %s: %s", entity.get("id"), e)

    for relation in relations:
        try:
            pg.upsert_relation({
                "id": relation["id"],
                "from_id": relation["from_id"],
                "to_id": relation["to_id"],
                "edge_type": relation["edge_type"],
                "properties": relation.get("properties", {}),
            })
        except Exception as e:
            errors += 1

    pg.close()
    console.print(f"[green]✓[/green] Imported {imported} entities ({errors} errors)")


# ---------------------------------------------------------------------------
# competitor group
# ---------------------------------------------------------------------------


@cli.group()
def competitor():
    """Track competitors and competitive intelligence."""
    pass


@competitor.command("add")
@click.argument("name")
@click.option("--website", default=None)
@click.option("--positioning", default=None)
@click.option("--project", default=None, help="Project they compete with")
@click.option("--tags", default="")
def competitor_add(name, website, positioning, project, tags):
    """Add a competitor."""
    tools = get_tools()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    result = tools.competitor_manage(
        action="create",
        name=name,
        website=website,
        positioning=positioning,
        project_id=project,
        tags=tag_list,
    )
    _print_result(result, f"Competitor added: {name}")


@competitor.command("list")
def competitor_list():
    """List all competitors."""
    tools = get_tools()
    result = tools.competitor_manage(action="list")
    competitors = result.get("competitors", [])
    if not competitors:
        console.print("[yellow]No competitors tracked.[/yellow]")
        return
    table = Table(title="Competitors")
    table.add_column("Name", style="cyan")
    table.add_column("ID", style="dim")
    for c in competitors:
        table.add_row(c.get("name", ""), c.get("id", ""))
    console.print(table)


# ---------------------------------------------------------------------------
# metric group
# ---------------------------------------------------------------------------


@cli.group()
def metric():
    """Track metrics and KPIs."""
    pass


@metric.command("record")
@click.argument("name")
@click.argument("value", type=float)
@click.option("--type", "metric_type", default="custom")
@click.option("--unit", default="")
@click.option("--project", default=None)
@click.option("--notes", default=None)
def metric_record_cmd(name, value, metric_type, unit, project, notes):
    """Record a metric data point."""
    tools = get_tools()
    result = tools.metric_record(
        name=name,
        metric_type=metric_type,
        value=value,
        unit=unit,
        project_id=project,
        notes=notes,
    )
    _print_result(result, f"Metric recorded: {name} = {value}")


@metric.command("query")
@click.argument("name")
@click.option("--since", default=None)
@click.option("--until", default=None)
@click.option("--limit", default=20, type=int)
@click.option("--json", "as_json", is_flag=True)
def metric_query_cmd(name, since, until, limit, as_json):
    """Query metric history."""
    tools = get_tools()
    result = tools.metric_query(name=name, since=since, until=until, limit=limit)
    if as_json:
        click.echo(json.dumps(result, indent=2, default=str))
        return
    if "error" in result:
        console.print(f"[red]Error:[/red] {result['error']}")
        sys.exit(1)
    points = result.get("data_points", [])
    console.print(f"[bold]{name}[/bold] — {len(points)} data points")
    for p in points[:limit]:
        console.print(f"  {str(p.get('recorded_at', ''))[:19]}  {p.get('value')} {result.get('metric', {}).get('metadata', {}).get('unit', '')}")


# ---------------------------------------------------------------------------
# graph command
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("entity_id")
@click.option("--depth", default=2, type=int)
@click.option("--json", "as_json", is_flag=True)
def graph(entity_id, depth, as_json):
    """Show the graph neighborhood of an entity."""
    from agentmemory.config import settings
    from agentmemory.db.age_client import AGEClient

    age = AGEClient(dsn=settings.database_url, graph_name=settings.graph_name)
    neighbors = age.get_neighborhood(entity_id, depth=depth)
    edges = age.get_edges(from_id=entity_id)
    age.close()

    if as_json:
        click.echo(json.dumps({"neighbors": neighbors, "edges": edges}, indent=2, default=str))
        return

    console.print(f"[bold]Graph neighborhood of {entity_id}[/bold] (depth={depth})")
    if not neighbors:
        console.print("[yellow]No connected nodes found.[/yellow]")
        return

    table = Table(show_lines=True)
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("ID", style="dim")
    for n in neighbors:
        table.add_row(
            str(n.get("name", "")),
            str(n.get("node_type", "")),
            str(n.get("id", "")),
        )
    console.print(table)


# ---------------------------------------------------------------------------
# workflow command
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("name")
@click.argument("content")
@click.option("--step", "steps", multiple=True, help="Add a step (repeatable)")
@click.option("--project", default=None)
@click.option("--tags", default="")
@click.option("--importance", default=None, type=float, help="Importance 0.0–1.0 (default: 0.65)")
def workflow(name, content, steps, project, tags, importance):
    """Save a reusable workflow or process."""
    tools = get_tools()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    result = tools.workflow_store(
        name=name,
        content=content,
        steps=list(steps),
        project_id=project,
        tags=tag_list,
        importance=importance,
    )
    _print_result(result, f"Workflow saved: {name}")


import logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# auto-link
# ---------------------------------------------------------------------------


@cli.command("auto-link")
@click.option("--dry-run", is_flag=True, help="Show what would be linked without writing")
@click.option("--limit", default=100, type=int, help="Max memories to process")
@click.option("--min-similarity", default=0.5, type=float, help="Minimum similarity threshold")
def auto_link_cmd(dry_run, limit, min_similarity):
    """Batch-link existing memories to each other via RELATED_TO edges.

    Finds semantically similar memories and creates graph relationships
    between them. Run this after importing memories or upgrading the
    embedding model.
    """
    from agentmemory.config import settings
    from agentmemory.core.memory import MemoryService

    svc = MemoryService(
        database_url=settings.database_url,
        redis_url=settings.redis_url,
        graph_name=settings.graph_name,
        embedding_model=settings.embedding_model,
        embedding_dim=settings.embedding_dim,
    )

    # Fetch searchable memories from PostgreSQL
    searchable_types = ["Memory", "Learning", "Decision", "Preference", "Workflow", "CustomerFeedback"]
    all_entities = []
    for node_type in searchable_types:
        entities = svc.postgres.list_entities_by_type(node_type)
        all_entities.extend(entities[:limit // len(searchable_types) + 1])

    all_entities = all_entities[:limit]
    console.print(f"[dim]Processing {len(all_entities)} memories...[/dim]")

    total_links = 0
    for entity in all_entities:
        entity_id = entity.get("id", "")
        meta = entity.get("metadata", {})
        if isinstance(meta, str):
            import json as _json
            try:
                meta = _json.loads(meta)
            except Exception:
                meta = {}
        content = meta.get("content", entity.get("name", ""))
        node_type = entity.get("node_type", "Memory")

        if not content or not entity_id:
            continue

        if dry_run:
            console.print(f"  [dim]Would link: {entity_id[:8]}... ({node_type})[/dim]")
            continue

        linked = svc.auto_link(
            node_id=entity_id,
            content=content,
            node_type=node_type,
            min_similarity=min_similarity,
        )
        total_links += len(linked)

    svc.close()

    if dry_run:
        console.print("[yellow]Dry run — no changes made.[/yellow]")
    else:
        console.print(f"[green]✓[/green] Created {total_links} RELATED_TO edges across {len(all_entities)} memories")


# ---------------------------------------------------------------------------
# reindex
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--yes", is_flag=True, help="Skip confirmation")
def reindex(yes):
    """Re-embed all memories with the current embedding model.

    Use this after upgrading the embedding model (e.g., from MiniLM to bge-base).
    Drops and recreates the Redis vector index, then re-stores all memories.
    """
    from agentmemory.config import settings
    from agentmemory.core.memory import MemoryService

    if not yes:
        click.confirm(
            "This will drop and recreate the Redis vector index. Continue?",
            abort=True,
        )

    svc = MemoryService(
        database_url=settings.database_url,
        redis_url=settings.redis_url,
        graph_name=settings.graph_name,
        embedding_model=settings.embedding_model,
        embedding_dim=settings.embedding_dim,
    )

    # Drop and recreate the index
    try:
        svc.redis.redis.execute_command("FT.DROPINDEX", svc.redis.index_name, "DD")
        console.print(f"[dim]Dropped index {svc.redis.index_name}[/dim]")
    except Exception:
        pass

    svc.redis._create_index()
    console.print(f"[dim]Created new index (dim={settings.embedding_dim})[/dim]")

    # Fetch all searchable entities and re-embed
    searchable_types = ["Memory", "Learning", "Decision", "Preference", "Workflow", "CustomerFeedback"]
    total = 0
    errors = 0

    for node_type in searchable_types:
        entities = svc.postgres.list_entities_by_type(node_type)
        for entity in entities:
            entity_id = entity.get("id", "")
            meta = entity.get("metadata", {})
            if isinstance(meta, str):
                import json as _json
                try:
                    meta = _json.loads(meta)
                except Exception:
                    meta = {}
            content = meta.get("content", entity.get("name", ""))

            if not content or not entity_id:
                continue

            try:
                embedding = svc.embeddings.encode(content, use_cache=False)
                doc = {
                    "id": entity_id,
                    "name": entity.get("name", ""),
                    "content": content,
                    "node_type": node_type,
                    "source": meta.get("source", "import"),
                    "tags": entity.get("tags", []),
                    "importance": meta.get("importance", 0.5),
                    "created_at": str(entity.get("created_at", "")),
                    "embedding": embedding,
                }
                svc.redis.store_memory(doc)
                total += 1
            except Exception as e:
                errors += 1
                logger.warning("Failed to reindex %s: %s", entity_id, e)

    svc.close()
    console.print(
        f"[green]✓[/green] Reindexed {total} memories "
        f"({errors} errors) with model [cyan]{settings.embedding_model}[/cyan]"
    )


# ---------------------------------------------------------------------------
# gc (garbage collect expired nodes)
# ---------------------------------------------------------------------------


@cli.command("gc")
@click.option("--dry-run", is_flag=True, help="List expired nodes without deleting")
@click.option("--type", "node_type", default=None, help="Filter by node type")
@click.option("--yes", is_flag=True, help="Skip confirmation before deleting")
def gc(dry_run, node_type, yes):
    """List or delete nodes that have passed their TTL.

    Nodes with a ttl_days value set will naturally decay in search scores,
    but this command lets you permanently remove them to keep the graph clean.

    \b
    Examples:
      memory gc --dry-run           # Preview what would be deleted
      memory gc --type Task         # Delete expired Task nodes
      memory gc --yes               # Delete all expired nodes without prompt
    """
    import json as _json
    from datetime import datetime, timedelta, timezone

    from agentmemory.config import settings
    from agentmemory.db.postgres import PostgresClient

    pg = PostgresClient(dsn=settings.database_url)
    pg._rollback_if_needed()

    with pg.conn.cursor() as cur:
        if node_type:
            cur.execute(
                "SELECT * FROM entities WHERE node_type = %s ORDER BY created_at",
                (node_type,),
            )
        else:
            cur.execute("SELECT * FROM entities ORDER BY created_at")
        entities = [dict(r) for r in cur.fetchall()]

    pg.close()

    now = datetime.now(timezone.utc)
    expired = []
    for entity in entities:
        meta = entity.get("metadata", {})
        if isinstance(meta, str):
            try:
                meta = _json.loads(meta)
            except Exception:
                meta = {}
        ttl = meta.get("ttl_days") if isinstance(meta, dict) else None
        if ttl is None:
            continue
        created = entity.get("created_at")
        if not created:
            continue
        try:
            if hasattr(created, "replace"):
                created_dt = created if created.tzinfo else created.replace(tzinfo=timezone.utc)
            else:
                dt_str = str(created).replace("Z", "+00:00")
                created_dt = datetime.fromisoformat(dt_str)
                if created_dt.tzinfo is None:
                    created_dt = created_dt.replace(tzinfo=timezone.utc)
            expiry = created_dt + timedelta(days=int(ttl))
            if now > expiry:
                expired.append((entity, expiry))
        except Exception:
            continue

    if not expired:
        console.print("[green]No expired nodes found.[/green]")
        return

    table = Table(title=f"Expired nodes ({len(expired)})", show_lines=True)
    table.add_column("Type", style="magenta", width=14)
    table.add_column("Name", style="cyan", max_width=50)
    table.add_column("Expired", style="red", width=20)
    table.add_column("ID", style="dim", width=36)
    for entity, expiry in expired:
        table.add_row(
            entity.get("node_type", ""),
            entity.get("name", "")[:50],
            str(expiry)[:19],
            entity.get("id", ""),
        )
    console.print(table)

    if dry_run:
        console.print("[yellow]Dry run — no changes made.[/yellow]")
        return

    if not yes:
        click.confirm(f"Delete {len(expired)} expired nodes?", abort=True)

    from agentmemory.core.memory import MemoryService

    svc = MemoryService(
        database_url=settings.database_url,
        redis_url=settings.redis_url,
        graph_name=settings.graph_name,
        embedding_model=settings.embedding_model,
        embedding_dim=settings.embedding_dim,
    )
    deleted = 0
    errors = 0
    for entity, _ in expired:
        try:
            svc.forget(entity["id"])
            deleted += 1
        except Exception as e:
            errors += 1
            logger.warning("gc: failed to delete %s: %s", entity.get("id"), e)
    svc.close()

    console.print(
        f"[green]✓[/green] Deleted {deleted} expired nodes"
        + (f" ({errors} errors)" if errors else "")
    )


# ---------------------------------------------------------------------------
# todo-import command
# ---------------------------------------------------------------------------


@cli.command("todo-import")
@click.argument("project_name")
@click.option("--goal", "goal_name", required=True, help="Goal name for this project's work")
@click.option("--initiative", "initiative_name", default=None, help="Optional initiative name to group tasks under")
@click.option("--dry-run", is_flag=True, help="Show what would be created without writing")
@click.pass_context
def todo_import(ctx, project_name, goal_name, initiative_name, dry_run):
    """Import pending tasks into MCP Goal/Task hierarchy.

    Creates a Goal and (optionally) an Initiative for the given project,
    then prompts for task names to add as Task nodes.

    Example:
        memory todo-import prodcamp --goal "Connect prod app to Hetzner OpenSearch"
        memory todo-import anton-cv --goal "Launch Anton CV" --initiative "Security & Async Refactor"

    After running, use `memory goal list` and `memory task list` to see the nodes.
    """
    from agentmemory.config import settings
    from agentmemory.core.memory import MemoryService

    console.print(f"[bold]Importing todos for project:[/bold] {project_name}")
    console.print(f"  Goal: {goal_name}")
    if initiative_name:
        console.print(f"  Initiative: {initiative_name}")

    if dry_run:
        console.print("[yellow]Dry run — showing what would be created:[/yellow]")
        console.print(f"  → goal_manage(create, name='{goal_name}', tags=['{project_name}'])")
        if initiative_name:
            console.print(f"  → initiative_manage(create, name='{initiative_name}', tags=['{project_name}'])")
        console.print("  → (interactive: add Task nodes one by one)")
        return

    svc = MemoryService(
        database_url=settings.database_url,
        redis_url=settings.redis_url,
        graph_name=settings.graph_name,
        embedding_model=settings.embedding_model,
        embedding_dim=settings.embedding_dim,
    )
    tools = get_tools()

    # Find or create the project entity
    projects = svc.postgres.list_entities_by_type("Project")
    project_entity = next((p for p in projects if p["name"].lower() == project_name.lower()), None)
    project_id = project_entity["id"] if project_entity else None
    if project_id:
        console.print(f"[dim]Found existing project: {project_entity['name']} ({project_id})[/dim]")

    # Create goal
    goal_result = tools.goal_manage(
        action="create",
        name=goal_name,
        description=f"Goal for {project_name}",
        project_id=project_id,
        tags=[project_name],
    )
    if "error" in goal_result:
        console.print(f"[red]Failed to create goal: {goal_result}[/red]")
        svc.close()
        return
    goal_id = goal_result.get("id") or goal_result.get("goal", {}).get("id")
    console.print(f"[green]✓[/green] Created goal: {goal_name} ({goal_id})")

    # Create initiative (optional)
    initiative_id = None
    if initiative_name:
        init_result = tools.initiative_manage(
            action="create",
            name=initiative_name,
            goal_id=goal_id,
            project_id=project_id,
            tags=[project_name],
        )
        if "error" not in init_result:
            initiative_id = init_result.get("id") or init_result.get("initiative", {}).get("id")
            console.print(f"[green]✓[/green] Created initiative: {initiative_name} ({initiative_id})")

    # Interactive task entry
    console.print("\n[bold]Add pending tasks (empty line to finish):[/bold]")
    task_count = 0
    while True:
        task_name = click.prompt("  Task name", default="", show_default=False)
        if not task_name.strip():
            break
        task_result = tools.task_manage(
            action="create",
            name=task_name.strip(),
            initiative_id=initiative_id,
            project_id=project_id,
            tags=[project_name],
        )
        if "error" in task_result:
            console.print(f"[red]  Failed: {task_result}[/red]")
        else:
            task_id = task_result.get("id") or task_result.get("task", {}).get("id", "?")
            console.print(f"  [green]✓[/green] {task_name} ({task_id})")
            task_count += 1

    svc.close()
    console.print(
        f"\n[green]✓[/green] Imported {task_count} task(s) under goal '{goal_name}'"
    )
    console.print(
        f"[dim]View with: memory task list[/dim]"
    )


# ---------------------------------------------------------------------------
# dedup command
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--dry-run", is_flag=True, help="Show duplicates without deleting")
@click.option("--similarity", default=0.92, type=float, help="Similarity threshold (default: 0.92)")
@click.option("--yes", is_flag=True, help="Skip confirmation prompts")
@click.option("--node-type", default=None, help="Restrict to a specific node type")
def dedup(dry_run, similarity, yes, node_type):
    """Find and remove near-duplicate memory nodes.

    Scans searchable nodes (Memory, Learning, Decision, etc.) and finds pairs
    with cosine similarity >= threshold. For each cluster, keeps the node with
    the highest importance (or most recently created) and removes the rest.

    Safe to run with --dry-run first to preview what would be deleted.
    """
    from agentmemory.config import settings
    from agentmemory.core.memory import MemoryService, SEARCHABLE_NODE_TYPES

    svc = MemoryService(
        database_url=settings.database_url,
        redis_url=settings.redis_url,
        graph_name=settings.graph_name,
        embedding_model=settings.embedding_model,
        embedding_dim=settings.embedding_dim,
    )

    types_to_scan = [node_type] if node_type else list(SEARCHABLE_NODE_TYPES)
    all_entities = []
    for nt in types_to_scan:
        entities = svc.postgres.list_entities_by_type(nt, limit=500)
        all_entities.extend(entities)

    console.print(f"[dim]Scanning {len(all_entities)} nodes for duplicates (threshold={similarity})...[/dim]")

    import json as _json

    # Build list of (id, content, importance, node_type) for each entity
    indexed = []
    for entity in all_entities:
        meta = entity.get("metadata", {})
        if isinstance(meta, str):
            try:
                meta = _json.loads(meta)
            except Exception:
                meta = {}
        content = meta.get("content", entity.get("name", ""))
        if content:
            indexed.append({
                "id": entity["id"],
                "name": entity.get("name", ""),
                "content": content,
                "node_type": entity.get("node_type", ""),
                "importance": meta.get("importance", 0.5),
                "created_at": entity.get("created_at"),
            })

    # For each node, find similar nodes and group into clusters
    to_delete: set[str] = set()
    duplicate_clusters: list[dict] = []

    processed: set[str] = set()
    for item in indexed:
        if item["id"] in to_delete or item["id"] in processed:
            continue
        try:
            embedding = svc.embeddings.encode(item["content"], is_query=True)
            candidates = svc.redis.vector_search(
                query_embedding=embedding,
                limit=10,
                min_score=similarity,
            )
        except Exception as e:
            logger.warning("dedup: search failed for %s: %s", item["id"], e)
            continue

        # Filter candidates: same node_type, not self, not already marked
        dupes = [
            c for c in candidates
            if c.get("id") and c["id"] != item["id"]
            and c.get("node_type", item["node_type"]) == item["node_type"]
            and c["id"] not in to_delete
        ]

        if dupes:
            cluster_ids = [item["id"]] + [d["id"] for d in dupes]
            cluster_full = [i for i in indexed if i["id"] in cluster_ids]
            if not cluster_full:
                continue
            # Keep node with highest importance, tiebreak by most recent created_at
            keeper = max(cluster_full, key=lambda x: (x["importance"], str(x.get("created_at", ""))))
            removals = [i for i in cluster_full if i["id"] != keeper["id"]]
            for r in removals:
                to_delete.add(r["id"])
            duplicate_clusters.append({"keeper": keeper, "duplicates": removals})
            processed.update(cluster_ids)

    if not duplicate_clusters:
        console.print("[green]✓ No duplicates found.[/green]")
        svc.close()
        return

    # Show what was found
    table = Table(
        title=f"Found {len(to_delete)} duplicate(s) in {len(duplicate_clusters)} cluster(s)",
        show_lines=True,
    )
    table.add_column("Action", width=8)
    table.add_column("Type", style="magenta", width=16)
    table.add_column("Name", style="cyan", max_width=45)
    table.add_column("Importance", width=10)
    table.add_column("ID", style="dim", width=36)
    for cluster in duplicate_clusters:
        k = cluster["keeper"]
        table.add_row("[green]KEEP[/green]", k["node_type"], k["name"][:45], f"{k['importance']:.2f}", k["id"])
        for d in cluster["duplicates"]:
            table.add_row("[red]DELETE[/red]", d["node_type"], d["name"][:45], f"{d['importance']:.2f}", d["id"])
    console.print(table)

    if dry_run:
        console.print("[yellow]Dry run — no changes made.[/yellow]")
        svc.close()
        return

    if not yes:
        click.confirm(f"Delete {len(to_delete)} duplicate node(s)?", abort=True)

    deleted = 0
    errors = 0
    for dup_id in to_delete:
        try:
            svc.forget(dup_id)
            deleted += 1
        except Exception as e:
            errors += 1
            logger.warning("dedup: failed to delete %s: %s", dup_id, e)
    svc.close()

    console.print(
        f"[green]✓[/green] Deleted {deleted} duplicate node(s)"
        + (f" ({errors} errors)" if errors else "")
    )


# ---------------------------------------------------------------------------
# audit command
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--min-sentences", default=3, type=int, help="Flag nodes with this many sentences or more")
@click.option("--limit", default=200, type=int, help="Max nodes to scan")
def audit(min_sentences, limit):
    """Audit the node corpus for coarse, multi-fact nodes that should be split.

    Lists nodes that are too long (>= min-sentences) and likely contain
    multiple facts. Use `memory split` on each to improve retrieval quality.

    Run `memory auto-link` afterwards to re-establish semantic relationships.
    """
    from agentmemory.config import settings
    from agentmemory.core.memory import MemoryService, SEARCHABLE_NODE_TYPES
    import json as _json

    svc = MemoryService(
        database_url=settings.database_url,
        redis_url=settings.redis_url,
        graph_name=settings.graph_name,
        embedding_model=settings.embedding_model,
        embedding_dim=settings.embedding_dim,
    )

    candidates = []
    for nt in SEARCHABLE_NODE_TYPES:
        entities = svc.postgres.list_entities_by_type(nt, limit=limit)
        for entity in entities:
            meta = entity.get("metadata", {})
            if isinstance(meta, str):
                try:
                    meta = _json.loads(meta)
                except Exception:
                    meta = {}
            content = meta.get("content", entity.get("name", ""))
            if not content:
                continue
            # Count sentences by splitting on . ! ?
            sentences = [s.strip() for s in content.replace("!", ".").replace("?", ".").split(".") if s.strip()]
            if len(sentences) >= min_sentences:
                candidates.append({
                    "id": entity["id"],
                    "node_type": entity.get("node_type", ""),
                    "name": entity.get("name", ""),
                    "sentences": len(sentences),
                    "chars": len(content),
                    "content_preview": content[:120],
                })

    svc.close()

    if not candidates:
        console.print(f"[green]✓ No nodes with {min_sentences}+ sentences found. Corpus looks atomic.[/green]")
        return

    # Sort by sentence count descending
    candidates.sort(key=lambda x: x["sentences"], reverse=True)

    table = Table(
        title=f"{len(candidates)} node(s) with {min_sentences}+ sentences (candidates for memory split)",
        show_lines=True,
    )
    table.add_column("Type", style="magenta", width=16)
    table.add_column("Sentences", width=9)
    table.add_column("Name", style="cyan", max_width=35)
    table.add_column("Preview", max_width=60)
    table.add_column("ID", style="dim", width=36)
    for c in candidates:
        table.add_row(
            c["node_type"],
            str(c["sentences"]),
            c["name"][:35],
            c["content_preview"],
            c["id"],
        )
    console.print(table)
    console.print(
        f"\n[dim]To split a node: [/dim][cyan]memory split <id> --chunk 'fact 1' --chunk 'fact 2' ...[/cyan]"
    )
    console.print(
        f"[dim]After splitting all: [/dim][cyan]memory auto-link[/cyan]"
    )


# ---------------------------------------------------------------------------
# tag-backfill command
# ---------------------------------------------------------------------------


@cli.command("tag-backfill")
@click.option("--dry-run", is_flag=True, help="Show what would be tagged without writing")
@click.option("--limit", default=500, type=int, help="Max nodes to scan")
def tag_backfill(dry_run, limit):
    """List nodes that have no project tags.

    Shows all untagged nodes so you can identify which project they belong to
    and update them with: memory update <id> --tags project-name

    Use this to improve project-scoped filtering (memory recall --tags busonmap).
    """
    from agentmemory.config import settings
    from agentmemory.core.memory import MemoryService
    import json as _json

    svc = MemoryService(
        database_url=settings.database_url,
        redis_url=settings.redis_url,
        graph_name=settings.graph_name,
        embedding_model=settings.embedding_model,
        embedding_dim=settings.embedding_dim,
    )

    # Scan all node types
    all_types = [
        "Memory", "Learning", "Decision", "Preference", "Workflow",
        "CustomerFeedback", "Goal", "Task", "Initiative", "Project",
    ]
    untagged = []
    for nt in all_types:
        entities = svc.postgres.list_entities_by_type(nt, limit=limit)
        for entity in entities:
            tags = entity.get("tags") or []
            if not tags:
                meta = entity.get("metadata", {})
                if isinstance(meta, str):
                    try:
                        meta = _json.loads(meta)
                    except Exception:
                        meta = {}
                content = meta.get("content", entity.get("name", ""))
                untagged.append({
                    "id": entity["id"],
                    "node_type": entity.get("node_type", ""),
                    "name": entity.get("name", ""),
                    "content_preview": content[:100],
                })

    svc.close()

    if not untagged:
        console.print("[green]✓ All nodes have at least one tag.[/green]")
        return

    table = Table(
        title=f"{len(untagged)} untagged node(s) — add project tags with: memory update <id> --tags project",
        show_lines=True,
    )
    table.add_column("Type", style="magenta", width=16)
    table.add_column("Name", style="cyan", max_width=40)
    table.add_column("Preview", max_width=60)
    table.add_column("ID", style="dim", width=36)
    for n in untagged:
        table.add_row(
            n["node_type"],
            n["name"][:40],
            n["content_preview"],
            n["id"],
        )
    console.print(table)
    console.print(
        f"\n[dim]To tag a node: [/dim][cyan]memory update <id> --tags project-name[/cyan]"
    )
    console.print(
        "[dim]Known project tags: busonmap, feedback1, prodcamp, cyprus-bus, agentmemory, anton-cv[/dim]"
    )

    if dry_run:
        console.print("[yellow]Dry run — no changes made.[/yellow]")


if __name__ == "__main__":
    cli()

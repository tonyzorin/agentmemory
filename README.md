# agentmemory.md

Persistent memory for AI agents — semantic search, knowledge graph, and structured nodes.

Works with any MCP-compatible agent: [Cursor](https://cursor.so), [Claude Desktop](https://claude.ai), [OpenClaw](https://github.com/openclaw/openclaw), and more.

Hosted at: https://gitlab.com/tonyzorin/agentmemory.md

---

## Why agentmemory.md?

Most AI coding assistants forget everything between sessions. agentmemory.md gives your agent a brain that persists:

- **What decisions you've made** — so it doesn't re-ask or suggest the opposite next session
- **What your projects are** — so it understands context without re-explaining
- **What failed** — so it doesn't repeat mistakes
- **Who the key people are** — so it knows who owns what

## Architecture

```
AI Agent (Cursor / Claude Desktop / OpenClaw / any MCP client)
    │
    ├─[MCP / HTTP]──► MCP Server (FastMCP v3)
    │                     │
    │                     ▼
    │              Memory Core Library
    │              ├── Embeddings (BAAI/bge-base-en-v1.5, 768-dim)
    │              ├── Hybrid Retrieval (Redis FT.HYBRID + AGE graph)
    │              └── Storage
    │                   ├── Redis 8.6 (FT.HYBRID: BM25 + vector, RRF fusion)
    │                   └── PostgreSQL 18 + Apache AGE 1.7.0 (graph)
    │
    └─[CLI]──────► memory store / recall / goal / task / ...
                   mem store / recall / ...  (short alias)
```

## Quick Start

```bash
# Start storage services
docker compose up -d

# Install
pip install -e .

# Store a memory
memory store "Anton prefers Claude for coding tasks" --tags preferences,tools

# Recall
memory recall "what does Anton prefer for coding"

# Get user profile
memory profile

# Stats
memory stats
```

## Connecting Agents

agentmemory.md runs as a server on your self-hosted machine (a VPS, homelab, or VM).
Your AI agents connect to it remotely. The recommended way to expose it privately is
[Tailscale](https://tailscale.com) — a zero-config VPN that gives every device on your
network a stable private IP, with no firewall rules or port-forwarding required.

### Why Tailscale?

Port 8006 must not be open to the public internet — it has no authentication.
Tailscale puts your server and your laptop on the same private overlay network,
so you can reach `http://100.x.x.x:8006` from anywhere without exposing anything publicly.

### Step 1 — Install Tailscale on the server

```bash
# On the server running agentmemory.md
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```

Note the Tailscale IP shown (`100.x.x.x`). You can also find it later with:

```bash
tailscale ip -4
```

### Step 2 — Install Tailscale on your laptop/desktop

Download from [tailscale.com/download](https://tailscale.com/download), sign in with the
same account. Both machines will appear in your [Tailscale admin console](https://login.tailscale.com/admin/machines).

### Step 3 — Start agentmemory.md

The server needs to bind on `0.0.0.0` (all interfaces) so Tailscale traffic can reach it.

**Streamable HTTP** (Cursor, Claude Code, any modern MCP client):

```bash
python -m agentmemory.mcp.server --transport streamable-http --port 8006
```

**SSE** (Claude Desktop, legacy clients):

```bash
python -m agentmemory.mcp.server --transport sse --port 8006
```

Or run both via Docker Compose (already configured for `0.0.0.0:8006`):

```bash
docker compose up -d
```

---

### Connect — Cursor IDE

Cursor supports the modern Streamable HTTP transport natively. In `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "agentmemory": {
      "url": "http://100.x.x.x:8006/mcp"
    }
  }
}
```

Replace `100.x.x.x` with your server's Tailscale IP.

---

### Connect — Claude Desktop

Claude Desktop does not yet support Streamable HTTP directly. Use
[`mcp-remote`](https://www.npmjs.com/package/mcp-remote) as a bridge, which connects
over SSE. In `~/Library/Application Support/Claude/claude_desktop_config.json`
(macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "agentmemory.md": {
      "command": "npx",
      "args": [
        "mcp-remote",
        "http://100.x.x.x:8006/sse",
        "--allow-http"
      ]
    }
  }
}
```

Replace `100.x.x.x` with your server's Tailscale IP. The `--allow-http` flag is needed
because `mcp-remote` defaults to HTTPS — Tailscale traffic is already encrypted at the
network layer so plain HTTP is safe here.

Make sure the server is running in SSE mode (see Step 3 above), then restart Claude Desktop.

---

### Connect — OpenClaw

For OpenClaw (stdio mode), see [OPENCLAW_SETUP.md](OPENCLAW_SETUP.md).
For automatic recall/capture, see the [agentmemory-openclaw-plugin](https://gitlab.com/tonyzorin/agentmemory-openclaw-plugin).

## Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.14 |
| MCP Server | FastMCP v3 (Streamable HTTP on port 8006) |
| Vector Search | Redis 8.6 (FT.HYBRID: BM25 + vector, RRF fusion) |
| Knowledge Graph | PostgreSQL 18 + Apache AGE 1.7.0 |
| Embeddings | `BAAI/bge-base-en-v1.5` (768-dim, CPU, MTEB ~63) |
| CLI | Click + Rich (`memory` or `mem`) |

## Scoring

Every `memory_recall` result includes these fields:

| Field | Description |
|-------|-------------|
| `score` | Final combined score (0–1) |
| `similarity` | Normalized RRF score from Redis FT.HYBRID (0–1) |
| `graph_boost` | +0.3 for 1-hop graph neighbors, +0.1 for 2-hop |
| `recency` | Exponential decay: 1.0 = just stored, ~0.5 = 29 days old |
| `importance` | Stored importance value (varies by node type) |

**Formula (similarity-adaptive):**
- Strong match (`similarity > 0.6`): `(sim×0.80 + graph×0.15 + recency×0.05) × importance_weight`
- Weak/medium match: `(sim×0.50 + graph×0.20 + recency×0.20 + importance×0.10) × importance_weight`

## Knowledge Graph

17 node types: Memory, Learning, Decision, Goal, Initiative, Task, Project, Person,
ExternalContact, Preference, Environment, Tool, Workflow, Resource, Competitor, Metric,
CustomerFeedback

25+ edge types: WORKS_ON, ABOUT, BELONGS_TO, FOR, ACHIEVED_VIA, BROKEN_INTO, TRACKS,
COMPETES_WITH, PREVENTED, and more.

## MCP Tools (21)

**Core memory:** `memory_store`, `memory_recall`, `memory_update`, `memory_relate`, `memory_context`, `memory_forget`, `memory_entities`, `memory_split`, `memory_batch_update`, `memory_profile`

**Work structure:** `goal_manage`, `initiative_manage`, `task_manage`, `timeline`

**Knowledge:** `learning_store`, `workflow_store`

**Market intelligence:** `competitor_manage`, `metric_record`, `metric_query`, `customer_feedback_store`

## CLI Reference

`memory` and `mem` are interchangeable — `mem` is the short alias.

```bash
mem store "content" [--type Memory|Learning|Decision|...] [--tags tag1,tag2] [--importance 0.8]
mem recall "query" [--limit 10] [--type Memory]
mem profile [--no-recent] [--limit 20]
mem update <memory-id> [--content "..."] [--name "..."] [--tags tag1,tag2] [--importance 0.8]
mem forget <memory-id> [--yes]
mem learn "what failed" --what-failed "..." --why "..." --avoid "..."
mem decide "decision" --rationale "why"
mem goal create "name" [--project <id>]
mem goal list
mem initiative create "name" [--goal <id>]
mem task create "name" [--initiative <id>]
mem task done <task-id> [--summary "what was done"]
mem task list
mem relate <from-id> <to-id> <EDGE_TYPE>
mem context <entity-id> [--depth 2]
mem timeline [--since 7d] [--type Memory]
mem graph <entity-id> [--depth 2]
mem competitor add "name" [--website url] [--positioning "..."]
mem metric record "name" <value> [--type visitors] [--unit count]
mem metric query "name" [--since 30d]
mem workflow "name" "description" [--step "step 1"] [--step "step 2"]
mem split <memory-id> --chunk "fact 1" --chunk "fact 2"
mem batch-update --type Goal --importance 0.8
mem gc [--dry-run] [--type Task]
mem stats
mem export [-o backup.json]
mem import backup.json
```

## Running Tests

```bash
# Start services first
docker compose up -d

# Run all tests
pytest

# Run specific phase
pytest tests/test_redis_client.py -v
pytest tests/test_age_client.py -v
pytest tests/test_e2e.py -v
```

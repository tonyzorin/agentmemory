# Connecting OpenClaw to agentmemory.md

## Prerequisites

1. The memory system is installed on VM 817 at `/home/agent1/agentmemory`
2. Docker Compose services are running (PostgreSQL + AGE, Redis)
3. Python virtual environment is set up

## Installation on VM 817

```bash
# Clone/copy the project
git clone git@gitlab.com:tonyzorin/agentmemory.md.git /home/agent1/agentmemory
cd /home/agent1/agentmemory

# Create virtual environment
python3 -m venv .venv

# Install dependencies
.venv/bin/pip install -e .

# Copy and configure environment
cp .env.example .env
# Edit .env if needed (default ports: PG=5433, Redis=6380)

# Start storage services
docker compose up -d

# Verify everything works
.venv/bin/memory stats
```

## OpenClaw Configuration

Add to `~/.openclaw/openclaw.json` on VM 817:

```json
{
  "mcpServers": {
    "memory": {
      "command": "/home/agent1/agentmemory/.venv/bin/python",
      "args": ["-m", "agentmemory.mcp.server"],
      "cwd": "/home/agent1/agentmemory",
      "env": {
        "DATABASE_URL": "postgresql://openclaw:openclaw@localhost:5433/openclaw_memory",
        "REDIS_URL": "redis://localhost:6380/0"
      }
    }
  }
}
```

## Remote Access (Streamable HTTP)

To access the memory system from other machines (e.g., your laptop's Cursor IDE), use Docker Compose (recommended) and expose via Tailscale:

```bash
# Start all services via Docker Compose
docker compose up -d

# Expose port 8081 on your Tailscale IP (persists across reboots)
tailscale serve --bg --tcp 8081 tcp://localhost:8081
```

Then in your local Cursor `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "agentmemory": {
      "url": "http://ai-agent:8081/mcp"
    }
  }
}
```

## Available MCP Tools

Once connected, the agent has access to these tools:

| Tool | Description |
|------|-------------|
| `memory_store` | Save memory/learning/decision/preference |
| `memory_recall` | Semantic search across all knowledge |
| `memory_relate` | Link entities in the knowledge graph |
| `memory_context` | Get full context for any entity |
| `memory_forget` | Remove outdated information |
| `memory_profile` | Get user profile summary (preferences, projects, people, goals) |
| `goal_manage` | Create/list/complete goals and OKRs |
| `initiative_manage` | Manage initiatives under goals |
| `task_manage` | Create/complete tasks |
| `timeline` | Query what happened in a time range |
| `learning_store` | Record failed experiments |
| `workflow_store` | Save reusable processes |
| `competitor_manage` | Track competitive intelligence |
| `metric_record` | Record KPI data points |
| `metric_query` | Query metrics over time |
| `customer_feedback_store` | Store customer feedback |

## CLI Usage

```bash
# Store a memory
memory store "Anton prefers Claude for coding tasks" --tags preferences,tools

# Recall memories
memory recall "what does Anton prefer for coding"

# Record a learning
memory learn "psycopg2-binary doesn't work on Python 3.14" \
  --what-failed "psycopg2-binary installation" \
  --why "No wheel for Python 3.14" \
  --avoid "psycopg2-binary on Python 3.14"

# Record a decision
memory decide "Use Redis 8.4 for vector search" \
  --rationale "Already proven in feedback1"

# Create a goal
memory goal create "Launch Feedback1 GTM" --project <project-id>

# Get user profile
memory profile

# Get context about an entity
memory context <entity-id>

# Show timeline
memory timeline --since 7d

# Stats
memory stats

# Weekly hygiene (wire orphans, consolidate dupes, gc stale tasks)
memory wire-orphans --dry-run
memory wire-orphans
memory consolidate --dry-run
memory consolidate --threshold 0.88 --type Memory
memory gc --dry-run
```

## Nightly Hygiene Routine (when OpenClaw works)

Until your automation agent is restored, run hygiene manually from Cursor or SSH on your deployment host.

When automation is back, add a nightly cron on the host where agentmemory runs:

```bash
# /etc/cron.d/agentmemory-hygiene (example: 03:00 daily)
0 3 * * * deploy-user cd /path/to/agentmemory && .venv/bin/memory wire-orphans >> /var/log/agentmemory-hygiene.log 2>&1
15 3 * * * deploy-user cd /path/to/agentmemory && .venv/bin/memory consolidate --threshold 0.88 --type Memory >> /var/log/agentmemory-hygiene.log 2>&1
30 3 * * * deploy-user cd /path/to/agentmemory && .venv/bin/memory gc --yes >> /var/log/agentmemory-hygiene.log 2>&1
```

Ensure `PROJECT_ANCHORS_JSON` is set in that host's `.env` so `wire-orphans` knows your tag → Project UUID map.

**What each step does:**

| Step | Command | Purpose |
|------|---------|---------|
| Wire orphans | `memory wire-orphans` | Link tagged facts → project anchors via `ABOUT` edges |
| Consolidate | `memory consolidate --threshold 0.88` | Merge near-duplicate Memory nodes |
| GC | `memory gc --yes` | Remove expired Task/Initiative nodes |

**Interim (Cursor / manual):** After substantive sessions, agents should `memory_supersede` on `potential_conflict` from `memory_store`. Run `memory wire-orphans --dry-run` weekly to catch untagged orphans.

**OpenClaw agent prompt (future):** Split coarse nodes (>2 sentences), wire `ABOUT` for new facts with project tags, supersede expired deadlines and contradictions.

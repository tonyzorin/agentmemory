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

To access the memory system from other machines (e.g., your laptop's Cursor IDE):

```bash
# On VM 817 — start Streamable HTTP server (recommended)
.venv/bin/python -m agentmemory.mcp.server --transport streamable-http --port 8081
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
```

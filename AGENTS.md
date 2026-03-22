# agentmemory.md — Agent Instructions

You have access to a persistent memory system via MCP tools. Use it proactively
to remember important information across conversations and sessions.

## Memory Hygiene — One Fact Per Node

**Critical rule:** Store one atomic fact per node. Do NOT pack multiple unrelated facts into a single node.

A node's embedding is computed from its entire content. When a node contains "John career history + Acme Corp + product launch + SaaS metrics all in one", every query about any of those topics gets a diluted match. Splitting into focused nodes pushes similarity scores from ~0.55 toward 0.85+.

**Good (focused):**
```python
memory_store("John worked at Acme Corp as Head of Product from 2018 to 2022", node_type="Memory")
memory_store("John joined Acme Corp as VP of Product in 2023", node_type="Memory")
memory_store("John's career focus is B2B SaaS and logistics technology", node_type="Memory")
```

**Bad (mixed):**
```python
memory_store("John career: Acme Corp Head of Product 2018-2022, joined Dashboard Co 2023, focus on B2B SaaS and analytics", node_type="Memory")
```

**Rule of thumb:** If a node's content contains more than one subject, or is longer than ~2 sentences, split it.

To split an existing node:
```python
memory_split(
    memory_id="<id-of-long-node>",
    chunks=[
        "John worked at Acme Corp as Head of Product from 2018 to 2022",
        "John joined Acme Corp as VP of Product in 2023",
        "John's career focus is B2B SaaS and logistics technology",
    ]
)
```

## Embedding Model

The system uses **BAAI/bge-base-en-v1.5** (768-dim, MTEB ~63) for semantic search.
This model produces higher-quality embeddings than the previous MiniLM model —
expect similarity scores in the 0.5–0.85 range for related content.

Queries are automatically prefixed with the BGE query instruction for optimal retrieval.
Documents are stored as-is (no prefix needed when storing).

## Scoring

Every recall result includes these fields:

| Field | Description |
|-------|-------------|
| `score` | Final combined score (0–1) |
| `similarity` | Normalized vector/keyword similarity (0–1) |
| `graph_boost` | 0.3 for 1-hop graph neighbors, 0.1 for 2-hop, 0 otherwise |
| `recency` | Exponential decay: 1.0 = just stored, ~0.5 = 29 days old |
| `importance` | The stored base importance value (0–1) |
| `effective_importance` | Dynamic importance after access_boost × age_penalty |
| `access_count` | How many times this node has been recalled |

Score formula is **similarity-adaptive**:
- When `similarity > 0.6` (strong match): `(sim×0.80 + graph×0.15 + recency×0.05) × importance_weight`
- Otherwise (weak/medium match): `(sim×0.50 + graph×0.20 + recency×0.20 + importance×0.10) × importance_weight`

This prevents a recent-but-unrelated node from outranking a well-matched older node.

**Dynamic importance** (`effective_importance`):
- `access_boost = min(1.5, 1 + access_count × 0.05)` — frequently recalled nodes get up to 50% boost
- `age_penalty` — type-specific decay for ephemeral nodes past their prime:
  - Task nodes > 30 days old: ×0.3
  - Initiative nodes > 90 days old: ×0.5
  - Memory nodes > 180 days old: ×0.7
  - CustomerFeedback nodes > 60 days old: ×0.5
  - Decision, Goal, Learning, Workflow: no decay (permanent)

## Default Importance by Node Type

When `importance` is not specified, the system uses these defaults:

| Node Type | Default Importance |
|-----------|-------------------|
| Goal | 0.80 |
| Decision | 0.70 |
| Learning | 0.65 |
| Workflow | 0.65 |
| CustomerFeedback | 0.60 |
| Project | 0.60 |
| Memory | 0.50 |
| Preference | 0.55 |

Always override with an explicit `importance` value when you know a memory is
especially critical (e.g., `importance=0.95` for a critical production decision).

## Auto-Linking

When a memory is stored, the system **automatically creates RELATED_TO edges**
to the top-3 most semantically similar existing memories. This means:

- Graph boost starts working immediately after the second memory is stored
- You don't need to manually call `memory_relate` for semantic connections
- Use `memory_relate` for explicit structural relationships (WORKS_ON, ABOUT, etc.)

To batch-link existing memories after a model upgrade or import:
```
memory auto-link
```

## When to Store Memories

**Always store:**
- Key decisions made (use `memory_store` with `node_type="Decision"`)
- Failed experiments and lessons learned (use `learning_store`)
- Project-specific preferences discovered (use `memory_store` with `node_type="Preference"`)
- New project profiles when starting work on a project (use `memory_store` with `node_type="Project"`)
- Important facts about people, tools, and systems

**Store when asked:**
- Goals and OKRs (use `goal_manage`)
- Initiatives and campaigns (use `initiative_manage`)
- Tasks and todos (use `task_manage`)
- Competitor information (use `competitor_manage`)
- Metrics and KPIs (use `metric_record`)
- Customer feedback (use `customer_feedback_store`)

## When to Recall Memories

Do NOT call memory tools unconditionally. Classify the request first, then fetch only what the tier requires.

### Step 1 — Classify the request (no MCP calls yet)

Read the user's message and any open files. Assign the lowest-numbered tier that fits:

| Tier | When | Signals |
|------|------|---------|
| **0** | Direct memory instruction | "check mem", "what do you know about me", explicit request for full memory context |
| **1** | Planning / prioritization | "what should I work on", "morning briefing", cross-project strategy, goal review |
| **2** | Any work — project unclear | Question where no open files or project name disambiguates which project it belongs to |
| **3** | Project work — project is obvious | Open files or explicit project name makes the project clear; feature work, refactoring, deployment |
| **4** | Narrow technical work — project is obvious | Bug fix, linter error, specific file edit where the project is clear from open files |

### Step 2 — Fetch only what the tier requires

**Tier 0 — Full memory dump:**
```python
memory_profile()  # full, include_recent=True
```

**Tier 1 — Compact planning context:**
```python
memory_profile(include_recent=False, limit=5)
goal_manage(action="list")
task_manage(action="list")
```

**Tier 2 — Discover project, then pull its context:**
```python
# Lightweight directory — returns only id, node_type, name (~1K tokens)
memory_entities(node_type="Project")

# Identify the relevant project from the list, then pull just that project
memory_context(entity_id="<matched-project-id>", depth=2)
```

**Tier 3 — Scoped recall for the known project:**
```python
# Use the project tag inferred from workspace or project name
memory_recall("<specific question>", tags=["acme-api"])
```

**Tier 4 — Nothing.** Project is obvious from open files. Do not call any memory tools.

### Examples

| User message | Open files | Tier | Calls |
|---|---|---|---|
| "fix the linter error in utils.py" | `utils.py` (project clear) | 4 | none |
| "fix the linter error in utils.py" | none / unclear | 2 | `memory_entities` → pick project → `memory_context` |
| "add auth to the API" | `acme-api/` files open | 3 | `memory_recall("auth decisions", tags=["acme-api"])` |
| "how should I structure the auth flow?" | none / ambiguous | 2 | `memory_entities` → pick project → `memory_context` |
| "what should I work on today?" | any | 1 | compact `memory_profile` + goals + tasks |
| "check mem" | any | 0 | full `memory_profile` |

### Key rules

- **Never call all tools at once** unless the user explicitly asks for a full memory dump (Tier 0).
- **Project unclear = Tier 2 minimum**, even for bug fixes — use `memory_entities` to find the project first.
- **Project obvious = check scope**: feature/refactor/deploy → Tier 3; narrow fix → Tier 4 (nothing).
- **`memory_context` is preferred over `memory_recall`** once you have the entity ID (richer, no query drift).

**Recall before:**
- Making architectural decisions → `memory_recall("decisions about <topic>", tags=["<project>"])`
- Trying something that might have been tried before → `memory_recall("<approach>", tags=["<project>"])`

**Use anchor_entity_id for richer results:**
```python
memory_recall("deployment issues", anchor_entity_id="<project-id>")
```
This boosts memories connected to the project in the knowledge graph.

## Project Tags

Tags scope memories to projects. Include at least one project tag in every `memory_store`, `learning_store`, `goal_manage`, and `task_manage` call.

**Discover existing project tags dynamically:**
```python
memory_entities(node_type="Project")
```

**Use tags in every store/recall call:**
```python
memory_store("Switched to Redis 8", node_type="Decision", tags=["acme-api"])
memory_recall("deployment issues", tags=["acme-api"])
task_manage(action="create", name="Add auth endpoint", tags=["acme-api"])
```

**Bootstrap: if no projects exist yet, store the first one:**
```python
memory_store("Acme API: Python/FastAPI SaaS backend", node_type="Project", tags=["acme-api"])
```

Tags are free-form strings — use short, lowercase, hyphenated slugs that match your project names. Multiple tags are allowed when a memory spans projects.

## Memory Types

| Type | When to Use | Default Importance | Example |
|------|-------------|-------------------|---------|
| `Memory` | General facts, observations | 0.50 | "John is based in Berlin" |
| `Learning` | Failed experiments, what NOT to do | 0.65 | "psycopg2-binary doesn't work on Python 3.14" |
| `Decision` | Key choices with rationale | 0.70 | "Use Redis for vector search" |
| `Preference` | Project-specific preferences | 0.55 | "acme-api uses Black for formatting" |
| `Workflow` | Reusable processes | 0.65 | "How to deploy Acme API to production" |
| `Project` | Project profiles | 0.60 | "Acme API: Python/FastAPI, runs on VM prod-01" |
| `Goal` | OKRs and objectives | 0.80 | "Launch Acme API GTM by Q2 2026" |
| `Initiative` | Campaigns under goals | 0.55 | "MCP integration for customers" |
| `Task` | Concrete work items | 0.45 | "Build SSE endpoint for MCP" |
| `Competitor` | Competing products | 0.55 | "Acme Competitor: SaaS analytics tool" |
| `Metric` | KPIs over time | 0.50 | "acme-api monthly active users" |
| `CustomerFeedback` | User feedback | 0.60 | "User says onboarding is confusing" |

## Relationship Types

Use `memory_relate` to link entities explicitly:

- `WORKS_ON` — person works on a project
- `ABOUT` — memory/feedback is about a project/person
- `BELONGS_TO` — task belongs to a project
- `FOR` — goal/workflow is for a project
- `ACHIEVED_VIA` — goal achieved via initiative
- `BROKEN_INTO` — initiative broken into tasks
- `COMPETES_WITH` — competitor competes with your project
- `TRACKS` — metric tracks a project/initiative
- `PREVENTED` — learning prevented a task
- `RELATED_TO` — generic semantic relationship (created automatically by auto-link)
- `SUPERSEDES` — new node replaces an older contradicting node (created via `memory_supersede`)

## Common Workflows

### Start of session

See "When to Recall Memories" above — classify first, then fetch only what the tier requires. For a planning session (Tier 1):
```python
memory_profile(include_recent=False, limit=5)
goal_manage(action="list")
task_manage(action="list")
```

### Record a critical decision
```python
memory_store(
  content="Decided to use FastMCP v3 instead of custom JSON-RPC",
  node_type="Decision",
  importance=0.85,
  extra={"rationale": "Less boilerplate, native stdio+SSE support"}
)
```

### Record a lesson learned
```python
learning_store(
  content="apache/age PG16 image doesn't work with PG18 volumes",
  what_failed="Docker volume from old PG16 image",
  why_it_failed="Incompatible data directory format",
  what_to_avoid="Reusing old volumes when upgrading PG version",
  importance=0.75,
)
```

### Wire relationships after storing (AI-driven graph linking)

After storing any important memory, use `memory_entities` to find existing entities
it should be connected to, then call `memory_relate` to create the edges.
This is how you build a rich knowledge graph automatically:

```python
# 1. Store the new memory
result = memory_store(
    content="Sarah is the sales lead for the Acme API enterprise deal",
    node_type="Person",
    name="Sarah",
)
new_id = result["id"]

# 2. Find existing entities to link to
entities = memory_entities()  # or memory_entities(node_type="Project")

# 3. Identify relevant ones (e.g. Acme API project, enterprise deal decision)
#    and create explicit edges
memory_relate(from_id=new_id, to_id="<acme-api-project-id>", edge_type="INVOLVED_IN")
memory_relate(from_id=new_id, to_id="<enterprise-deal-decision-id>", edge_type="RELATED_TO")

# 4. Now recall with graph boost — Sarah surfaces when asking about Acme API
memory_recall("enterprise deal contacts", anchor_entity_id="<acme-api-project-id>")
```

**Edge type guide for wiring:**
- Person → Project: `WORKS_ON` or `INVOLVED_IN`
- Memory/Decision → Project: `ABOUT`
- Task → Project: `BELONGS_TO`
- Learning → Task it prevented: `PREVENTED`
- Goal → Project: `FOR`
- Memory → another Memory: `RELATED_TO` (auto-created by auto-link, but use this for explicit semantic links)

### Link people to projects (simple case)
```python
person = memory_store(content="Sarah — enterprise deal contact", node_type="Person", name="Sarah")
project = memory_store(content="Acme API project", node_type="Project", name="acme-api")
memory_relate(from_id=person["id"], to_id=project["id"], edge_type="INVOLVED_IN")
memory_recall("enterprise deal contacts", anchor_entity_id=project["id"])
```

### After upgrading embedding model or importing data
```bash
# Re-embed all memories with new model
memory reindex --yes

# Batch-link memories to each other
memory auto-link
```

### Complete a task
```python
task_manage(
  action="complete",
  task_id="<id>",
  result_summary="Implemented SSE transport, tested with mcp-remote"
)
```

## New Tools Reference

### `memory_list` — audit the full node corpus
```python
# List all nodes (newest first)
memory_list()

# List only Decision nodes, sorted by importance
memory_list(node_type="Decision", order_by="importance", order_dir="desc")

# Paginate through large corpora
memory_list(limit=50, offset=50)
```
Returns `id`, `node_type`, `name`, `tags`, `created_at`, `importance`, `access_count` for each node.

### `memory_split` — break a long node into focused atomic nodes
```python
memory_split(
    memory_id="<id-of-long-node>",
    chunks=[
        "John worked at Acme Corp as Head of Product from 2018 to 2022",
        "John joined Acme Corp as VP of Product in 2023",
        "John's career focus is B2B SaaS and logistics technology",
    ]
)
```
The original node is deleted. Each chunk becomes a new node inheriting the original's type, importance, tags, and graph edges.

### `memory_batch_update` — bulk-update importance
```python
# Fix all Goals that were stored at default importance=0.5
memory_batch_update(importance=0.8, node_type="Goal")

# Fix specific critical nodes
memory_batch_update(importance=0.9, ids=["uuid1", "uuid2"])
```

## Important Notes

- IDs are UUIDs — always use the exact ID returned when relating entities
- The memory system stores data in PostgreSQL + Redis + Apache AGE graph
- All tools return structured errors (`{"error": "...", "reason": "..."}`) on failure
- Use `memory_context(<entity_id>)` to get the full picture of any entity
- Use `timeline(since="7d")` to see what happened recently
- Auto-linking runs automatically on every `store` — no manual action needed for new memories
- After importing old data or upgrading the model, run `memory reindex` then `memory auto-link`
- **anchor_entity_id is now auto-detected** from query text — you no longer need to pass it manually, though explicit passing still takes priority
- **graph_boost is now graduated**: +0.3 for direct (1-hop) neighbors, +0.1 for 2-hop — unrelated nodes get 0
- **Scoring is similarity-adaptive**: when similarity > 0.6, similarity dominates (80% weight); recency can no longer outrank a well-matched node
- **Importance is now dynamic**: `effective_importance = base × access_boost × age_penalty` — frequently recalled nodes get up to 50% boost; stale Task/Initiative/CustomerFeedback nodes decay
- **Task nodes auto-expire** after 90 days (TTL decay) — run `memory gc` to clean them up
- **One fact per node** — split long/mixed nodes with `memory_split` for best retrieval scores

## Conflict Resolution

When you store a new memory that is similar (but not identical) to an existing one, `memory_store` may return a `potential_conflict` field:

```python
result = memory_store("John prefers Rust for systems programming", node_type="Preference")
# result might be:
# {
#   "id": "<new-id>",
#   "node_type": "Preference",
#   "content": "...",
#   "potential_conflict": {
#     "id": "<old-id>",
#     "content": "John prefers Python for all programming tasks",
#     "similarity": 0.81
#   }
# }
```

When this happens, decide whether the new memory supersedes the old one:

```python
# If yes — the old memory is outdated/replaced:
memory_supersede(new_id=result["id"], old_id=result["potential_conflict"]["id"])
# The old node remains in the graph (audit trail) but is excluded from future searches.

# If no — they're genuinely different memories, both should be kept:
# Do nothing. Both nodes will be in search results.
```

**When to supersede:** Preferences, decisions, or facts that have changed over time.
**When to keep both:** Related but distinct memories that don't contradict each other.

## Corpus Maintenance

As the corpus grows over weeks/months, run these commands periodically:

```bash
# Remove near-duplicate memories (weekly or monthly)
memory consolidate --dry-run     # preview first
memory consolidate --no-dry-run  # then merge

# Remove expired Task/Initiative nodes
memory gc --dry-run
memory gc --yes
```

Consolidation merges semantically similar nodes (default threshold: cosine similarity ≥ 0.85) into a single canonical node, preserving the best content and all graph edges. It never runs automatically — you control when it happens.

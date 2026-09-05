<!-- agentmemory rules v2026.07.28 — shared policy (platform-agnostic) -->

# agentmemory.md — Core Usage Policy

Shared fetch/store rules for all MCP clients (Cursor, Claude, Grok, ChatGPT, OpenClaw). Platform adapters add install/paste instructions only.

## Session Start — Classify First, Then Fetch

Do NOT call memory tools unconditionally. Before doing any work, silently classify the request and call only what the tier requires.

### Step 1 — Classify the request (no MCP calls yet)

Read the user's message and workspace / chat context. Assign the lowest-numbered tier that fits:

| Tier | When | Signals |
|---|---|---|
| **0** | Direct memory instruction | "check mem", "add to mem", "what do you know about me", explicit request for full memory context |
| **1** | Planning / prioritization | "what should I work on", "morning briefing", cross-project strategy, goal review |
| **2** | Any work — project unclear | Feature, bug, or refactor where workspace / chat context does not disambiguate the project |
| **3** | Project work — project is obvious | Project clear from context; feature work, refactoring, deployment, security review |
| **4** | Narrow technical work — project is obvious | Bug fix, linter error, specific file edit — workspace / chat context is sufficient |

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
memory_entities(node_type="Project")  # lightweight directory (~1K tokens)
memory_context(entity_id="<matched-project-id>", depth=2)
```

**Tier 3 — Known project, scoped fetch:**
```python
# Prefer when project entity id is known (from Step 2, prior turn, or project anchors):
memory_context(entity_id="<project-id>", depth=2)

# Else — scoped semantic search by project slug:
memory_recall("<specific question>", tags=["project-slug"])
```

**Tier 4 — Nothing.** Project is obvious and workspace / chat context is sufficient. Do not call memory tools.

### Reclassify and continuation

- **Mid-session:** If the active project changes, re-run Tier 2 or 3 for the new project. Do not keep using the previous project's recalled context.
- **Continuation:** If this turn continues the same task and relevant facts are already in the thread, treat as Tier 4 — do not re-fetch.

### Examples

| User message | Context | Tier | Calls |
|---|---|---|---|
| "fix the linter error in utils.py" | Project clear | 4 | none |
| "fix the linter error in utils.py" | Project unclear | 2 | `memory_entities` → `memory_context` |
| "add auth to the API" | `acme-api` clear | 3 | `memory_context` if id known, else `memory_recall(..., tags=["acme-api"])` |
| "review OAuth for vulns before commit" | Repo files open | 3 or 4 | context-only if diff local; else scoped fetch |
| "how should I structure the data model?" | Ambiguous | 2 | `memory_entities` → `memory_context` |
| "what should I work on today?" | Any | 1 | compact `memory_profile` + goals + tasks |
| "check mem" | Any | 0 | full `memory_profile` |

### Key fetch rules

- **Never combine** `memory_profile` + `memory_entities` + `memory_context` + `memory_recall` in one turn unless Tier 0.
- **Project unclear = Tier 2 minimum**, even for bug fixes.
- **Project obvious = check scope:** feature/refactor/deploy/review → Tier 3; narrow fix → Tier 4.
- **`memory_context` is preferred over `memory_recall`** once you have the entity ID (richer, no query drift).
- Tiers govern **fetching** only. **Store** rules apply on every turn when facts arise.

## Store Immediately — Do NOT Wait Until Session End

Store facts **as soon as they happen**. Session-end review is a safety net, not the primary mechanism.

**Store immediately when:**

1. **A decision is made** — user agrees to a recommendation or chooses an approach:
   ```python
   memory_store("Decided to use Postgres for primary storage", node_type="Decision", importance=0.85, tags=["acme-api"])
   ```

2. **A new fact is learned** — projects, tools, people, deadlines:
   ```python
   memory_store("Acme API deploys via Docker Compose on Hetzner", node_type="Memory", importance=0.7, tags=["acme-api"])
   ```

3. **A preference is stated** — likes, dislikes, process defaults:
   ```python
   memory_store("Prefers concise commit messages focused on why", node_type="Preference", importance=0.5, tags=["acme-api"])
   ```

4. **A task is completed** — user confirms something is done:
   ```python
   task_manage(action="complete", task_id="<id>", result_summary="...")
   ```

5. **A lesson is learned** — failure, workaround, or constraint:
   ```python
   learning_store(content="...", what_failed="...", why_it_failed="...", what_to_avoid="...", tags=["acme-api"])
   ```

6. **A new goal or task is created:**
   ```python
   task_manage(action="create", name="...", tags=["acme-api"])
   ```

### Wire on store (same turn)

When project entity id is known, link immediately — do not wait for session end:

```python
result = memory_store("...", node_type="Decision", tags=["acme-api"])
if isinstance(result, dict) and result.get("id"):
    memory_relate(from_id=result["id"], to_id="<project-id>", edge_type="ABOUT")
```

## Do NOT Store

- **Secrets:** API keys, Bearer `am_…` / `amo_…`, OAuth client secrets, passwords, `.env` values
- **Ephemeral debug:** stack traces, one-off curl output, transient errors
- **Code in git:** file contents, diffs, PR bodies — store the *decision*, not the patch
- **Duplicates:** if recall already has the same fact, skip or supersede

## Importance Rubric

| Range | Use for |
|---|---|
| 0.9–1.0 | Hard constraints, contracts, "never do X" |
| 0.8–0.85 | Decisions, deadlines, deploy topology |
| 0.6–0.7 | Useful facts, contacts, tool quirks |
| ≤0.5 | Soft preferences |

## Project Tags — Mandatory

Every `memory_store`, `learning_store`, `goal_manage`, `task_manage` call MUST include at least one project tag.

Discover existing projects and slugs:
```python
memory_entities(node_type="Project")
```

Use short, lowercase, hyphenated slugs. Bootstrap a new project (name is required — a short slug, not a sentence):
```python
memory_store("Python/FastAPI SaaS backend", node_type="Project", name="acme-api", tags=["acme-api"])
```

## Task Tracking

Use MCP `task_manage` for multi-day work — not ad-hoc markdown task files.

```python
task_manage(action="create", name="Add rate limiting to auth endpoint", tags=["acme-api"])
task_manage(action="complete", task_id="<id>", result_summary="Implemented with Redis token bucket")
```

## One Fact Per Node

Store atomic facts — one subject per node, max ~2 sentences:

```python
# Good
memory_store("Acme API uses Black for formatting", node_type="Preference", tags=["acme-api"])

# Bad — mixed subjects
memory_store("Acme uses Black, deploys on Hetzner, John is lead...", ...)
```

Split long nodes: `memory_split(memory_id="<id>", chunks=[...])`. Store the durable fact, then supersede play-by-play session diaries.

## Conflict Resolution — Supersede Same Turn

When `memory_store` returns `potential_conflict`, resolve in the **same turn**:

```python
result = memory_store("Employment status changed to full-time April 2026", node_type="Memory", tags=["acme-api"])
if isinstance(result, dict) and result.get("potential_conflict"):
    memory_supersede(
        new_id=result["id"],
        old_id=result["potential_conflict"]["id"],
    )
```

**Supersede when:** employment, deadlines, preferences, or project status changed — old fact is no longer true.
**Keep both when:** related but non-contradictory (e.g. two different contacts).

Per July 2026 policy: supersede chain only (old nodes kept for audit, excluded from recall). Do not delete after supersede unless the node is empty noise.

## Session End — Safety Net

Before ending any session where substantive work was done, review and store anything missed:

1. Key decisions → `memory_store(..., node_type="Decision", ...)`
2. Lessons → `learning_store(...)`
3. Finished tasks → `task_manage(action="complete", ...)`
4. Orphan wiring → `memory_relate(..., edge_type="ABOUT")`

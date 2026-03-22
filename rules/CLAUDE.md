<!-- agentmemory rules v2026.03.15 -->

# agentmemory.md — Usage Rules

Place this file as `CLAUDE.md` in your project root. Claude Code will load it automatically.

## Session Start — Classify First, Then Fetch

Do NOT call memory tools unconditionally. Before doing any work, silently classify
the request and call only what the tier requires.

### Step 1 — Classify the request (no MCP calls yet)

Read the user's message and any referenced files. Assign the lowest-numbered tier that fits:

| Tier | When | Signals |
|------|------|---------|
| **0** | Direct memory instruction | "check mem", "what do you know about me", explicit request for full memory context |
| **1** | Planning / prioritization | "what should I work on", "morning briefing", cross-project strategy, goal review |
| **2** | Any work — project unclear | Question where no referenced files or project name disambiguates which project it belongs to |
| **3** | Project work — project is obvious | Referenced files or explicit project name makes the project clear; feature work, refactoring, deployment |
| **4** | Narrow technical work — project is obvious | Bug fix, linter error, specific file edit where the project is clear |

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
memory_recall("<specific question>", tags=["your-project-tag"])
```

**Tier 4 — Nothing.** Project is obvious from context. Do not call any memory tools.

### Examples

| User message | Context | Tier | Calls |
|---|---|---|---|
| "fix the linter error in utils.py" | File path makes project clear | 4 | none |
| "fix the linter error in utils.py" | No file context | 2 | `memory_entities` → pick project → `memory_context` |
| "add auth to the API" | `acme-api/` referenced | 3 | `memory_recall("auth decisions", tags=["acme-api"])` |
| "how should I structure the data model?" | Ambiguous | 2 | `memory_entities` → pick project → `memory_context` |
| "what should I work on today?" | Any | 1 | compact `memory_profile` + goals + tasks |
| "check mem" | Any | 0 | full `memory_profile` |

### Key rules

- **Never call all tools at once** unless the user explicitly asks for a full memory dump (Tier 0).
- **Project unclear = Tier 2 minimum**, even for bug fixes — use `memory_entities` to find the project first.
- **Project obvious = check scope**: feature/refactor/deploy → Tier 3; narrow fix → Tier 4 (nothing).
- **`memory_context` is preferred over `memory_recall`** once you have the entity ID (richer, no query drift).

## Store Immediately — Do NOT Wait Until Session End

**Store immediately when:**

1. **A decision is made:**
   ```python
   memory_store("Decided to use Postgres for primary storage", node_type="Decision", importance=0.85, tags=["acme-api"])
   ```

2. **A new fact is learned:**
   ```python
   memory_store("Acme API deploys to Hetzner via Docker Compose", node_type="Memory", importance=0.8, tags=["acme-api"])
   ```

3. **A task is completed:**
   ```python
   task_manage(action="complete", task_id="<id>", result_summary="...")
   ```

4. **A lesson is learned:**
   ```python
   learning_store(content="...", what_failed="...", why_it_failed="...", what_to_avoid="...", tags=["acme-api"])
   ```

5. **A new goal or task is created:**
   ```python
   task_manage(action="create", name="...", tags=["acme-api"])
   ```

## Project Tags — Mandatory

Every store/recall call MUST include at least one project tag. Discover existing tags:

```python
memory_entities(node_type="Project")
```

Bootstrap your first project:
```python
memory_store("Acme API: Python/FastAPI SaaS backend", node_type="Project", tags=["acme-api"])
```

## Session End — Safety Net

Before ending any session where substantive work was done:

1. `memory_store(..., node_type="Decision", tags=["<project>"])` — key decisions
2. `learning_store(content=..., tags=["<project>"])` — lessons learned
3. `task_manage(action="complete", task_id=..., result_summary=...)` — finished tasks
4. `memory_relate(from_id=..., to_id="<project-id>", edge_type="ABOUT")` — wire new nodes

## One Fact Per Node

Store atomic facts — one subject per node, max ~2 sentences. Split long nodes:

```python
memory_split(memory_id="<id>", chunks=["fact 1", "fact 2", "fact 3"])
```

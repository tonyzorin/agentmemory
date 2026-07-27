<!-- agentmemory rules v2026.07.28 — Claude adapter; shared policy: rules/CORE.md -->

# agentmemory.md — Usage Rules (Claude Code)

Place this file as `CLAUDE.md` in your project root. Claude Code loads it automatically.

For the full shared policy, see [`rules/CORE.md`](CORE.md) in the agentmemory repo.

---

## Session Start — Classify First, Then Fetch

Do NOT call memory tools unconditionally. Silently classify, then fetch only what the tier requires.

### Step 1 — Classify (no MCP calls yet)

Read the user's message and referenced files. Lowest tier that fits:

| Tier | When | Signals |
|---|---|---|
| **0** | Direct memory instruction | "check mem", "what do you know about me", full memory context requested |
| **1** | Planning / prioritization | "what should I work on", morning briefing, goal review |
| **2** | Project unclear | No referenced files or name disambiguates project |
| **3** | Project obvious | Referenced files or name clear; feature, refactor, deploy, review |
| **4** | Narrow fix, project obvious | Linter error, small edit — context sufficient |

### Step 2 — Fetch

**Tier 0:** `memory_profile()`  
**Tier 1:** `memory_profile(include_recent=False, limit=5)` + `goal_manage(action="list")` + `task_manage(action="list")`  
**Tier 2:** `memory_entities(node_type="Project")` → `memory_context(entity_id="...", depth=2)`  
**Tier 3:** `memory_context(entity_id="...", depth=2)` if id known; else `memory_recall("...", tags=["slug"])`  
**Tier 4:** nothing

**Reclassify** when project changes mid-chat. **Continuation** of same task with facts in thread → Tier 4.

### Key fetch rules

- Never combine `memory_profile` + `memory_entities` + `memory_context` + `memory_recall` unless Tier 0.
- Project unclear = Tier 2 minimum.
- `memory_context` preferred over `memory_recall` when entity ID is known.

### Examples

| Message | Context | Tier | Calls |
|---|---|---|---|
| "fix linter in utils.py" | path clear | 4 | none |
| "fix linter in utils.py" | unclear | 2 | entities → context |
| "add auth to API" | acme-api clear | 3 | context or recall |
| "check mem" | any | 0 | full profile |

## Store Immediately

Store when: decision, fact, preference, task complete, lesson, new task. Wire `memory_relate(..., ABOUT)` same turn when project id known.

## Do NOT Store

Secrets (`am_…`/`amo_…`, OAuth secrets, passwords, `.env`), debug output, git diffs, duplicates.

## Importance

0.9–1.0 constraints · 0.8–0.85 decisions/deadlines · 0.6–0.7 facts · ≤0.5 preferences. Use `node_type="Preference"` for likes/dislikes.

## Tags — Mandatory

`memory_entities(node_type="Project")` to discover slugs. Every store/task/goal/learning needs ≥1 tag.

## One Fact Per Node

Max ~2 sentences. `memory_split` if too long.

## Supersede Same Turn

```python
result = memory_store("...", tags=["acme-api"])
if isinstance(result, dict) and result.get("potential_conflict"):
    memory_supersede(new_id=result["id"], old_id=result["potential_conflict"]["id"])
```

## Session End — Safety Net

Catch missed decisions, lessons, completions, and wiring.

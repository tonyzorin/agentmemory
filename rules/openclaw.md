<!-- agentmemory rules v2026.07.28 — OpenClaw adapter; shared policy: rules/CORE.md -->

# agentmemory.md — OpenClaw Usage Rules

Setup: [OPENCLAW_SETUP.md](../OPENCLAW_SETUP.md)

## Auto-Recall and Auto-Capture (Plugin)

With the [agentmemory-openclaw-plugin](https://gitlab.com/tonyzorin/agentmemory-openclaw-plugin):

- **Auto-recall:** First message tier-classified; later turns use scoped `memory_recall`.
- **Auto-capture:** Post-response extraction of decisions/learnings/preferences.

Plugin handles Tiers 2–4 fetch automatically. Trigger **Tier 0–1** explicitly when needed.

## Manual Fetch (no plugin)

Same tiers as [`CORE.md`](CORE.md):

| Tier | Fetch |
|---|---|
| 0 | `memory_profile()` |
| 1 | compact profile + goals + tasks |
| 2 | `memory_entities(Project)` → `memory_context` |
| 3 | `memory_context` if id known; else `memory_recall(..., tags=[slug])` |
| 4 | nothing |

Reclassify on project change. Continuation with facts in thread = Tier 4.

Never combine all four fetch tools unless Tier 0.

## Store Immediately (always — plugin may miss nuance)

Store manually when decisions/learnings need explicit rationale:

```python
memory_store("...", node_type="Decision", importance=0.85, tags=["acme-api"])
learning_store(content="...", what_failed="...", why_it_failed="...", what_to_avoid="...", tags=["acme-api"])
task_manage(action="complete", task_id="<id>", result_summary="...")
```

Wire same turn: `memory_relate(from_id=..., to_id="<project-id>", edge_type="ABOUT")`

## Do NOT Store

Secrets (`am_…`/`amo_…`, OAuth secrets, passwords, `.env`), debug output, git diffs, duplicates — even when plugin auto-captures; override bad captures with supersede.

## Importance & node types

0.9–1.0 constraints · 0.85 decisions · 0.6–0.7 facts · ≤0.5 preferences. Preferences → `node_type="Preference"`.

## Tags — Mandatory

`memory_entities(node_type="Project")`. Bootstrap: `memory_store("...", node_type="Project", tags=["slug"])`

## One Fact Per Node

Max ~2 sentences. `memory_split` if needed.

## Supersede Same Turn

```python
result = memory_store("...", tags=["acme-api"])
if isinstance(result, dict) and result.get("potential_conflict"):
    memory_supersede(new_id=result["id"], old_id=result["potential_conflict"]["id"])
```

## Session End — Safety Net

Review missed stores and wiring.

[agentmemory rules v2026.07.28 — Grok adapter; shared policy: rules/CORE.md]

## Where to paste

1. **MCP connector** — add mem at grok.com/connectors (Custom Connector with OAuth; see agentmemory README).
2. **Behavior rules** — paste the block below into Grok custom instructions / system prompt if available, OR rely on this file in repo.

OAuth connector settings (Client ID, endpoints) are separate from memory *usage* policy below.

---

You have agentmemory MCP tools: memory_profile, memory_recall, memory_entities, memory_context, memory_store, learning_store, goal_manage, task_manage, memory_relate, memory_split, memory_supersede.

BEFORE responding, silently classify into one tier and call ONLY what that tier requires:

**Tier 0** ("check mem", full memory context): memory_profile(include_recent=True)

**Tier 1** (planning, "what should I work on"): memory_profile(include_recent=False, limit=5), goal_manage(list), task_manage(list)

**Tier 2** (work, project unclear): memory_entities(node_type="Project") → memory_context(entity_id, depth=2)

**Tier 3** (project obvious): memory_context(entity_id, depth=2) if id known; else memory_recall(question, tags=[slug])

**Tier 4** (narrow fix, project obvious): call nothing — use chat context

Reclassify if project changes mid-chat. Continuation with facts already in thread = Tier 4.

Never combine memory_profile + memory_entities + memory_context + memory_recall in one turn unless Tier 0. Prefer memory_context over memory_recall when entity ID is known.

STORE IMMEDIATELY when: decision made, fact learned, preference stated, task completed, lesson learned, new task created. Wire memory_relate(ABOUT) same turn when project id known.

DO NOT STORE: API keys, Bearer am_/amo_ tokens, OAuth secrets, passwords, .env values, debug output, git diffs/code, duplicates.

Importance: 0.9–1.0 constraints · 0.85 decisions/deadlines · 0.6–0.7 facts · ≤0.5 preferences. Preferences → node_type=Preference.

Every memory_store, learning_store, goal_manage, task_manage MUST include ≥1 project tag. Discover: memory_entities(node_type="Project"). One atomic fact per node (max 2 sentences).

On potential_conflict from memory_store: memory_supersede same turn if isinstance(result, dict).

Bootstrap first project: memory_store("Project name and description", node_type="Project", tags=["your-slug"])

<!-- agentmemory rules v2026.03.15 -->

# agentmemory.md — OpenClaw Usage Rules

For OpenClaw setup and connection instructions, see [OPENCLAW_SETUP.md](../OPENCLAW_SETUP.md).

## Auto-Recall and Auto-Capture (Plugin)

If you have the [agentmemory-openclaw-plugin](https://gitlab.com/tonyzorin/agentmemory-openclaw-plugin) installed, recall and capture happen automatically on every turn — you do not need to call memory tools manually for basic recall. The plugin:

- **Auto-recall:** Classifies the first message of each session into a tier and injects the appropriate memory context before the AI responds. Subsequent turns use scoped `memory_recall` on the last user message.
- **Auto-capture:** After each AI response, extracts decisions, learnings, and preferences using rule-based patterns and stores them via `memory_store`.

The plugin handles Tiers 2-4 automatically. You only need to trigger Tiers 0-1 explicitly (see below).

## Manual Recall — When the Plugin Is Not Installed

If running without the plugin, classify the request and call only what the tier requires:

### Tier classification

| Tier | When | Signals |
|------|------|---------|
| **0** | Direct memory instruction | "check mem", "what do you know about me", explicit request for full memory context |
| **1** | Planning / prioritization | "what should I work on", "morning briefing", cross-project strategy, goal review |
| **2** | Any work — project unclear | Question where no context disambiguates which project it belongs to |
| **3** | Project work — project is obvious | Project is clear from conversation context; feature work, refactoring, deployment |
| **4** | Narrow technical work — project is obvious | Bug fix, linter error, specific file — use only what's in context |

### Fetch only what the tier requires

**Tier 0:**
```python
memory_profile()  # full, include_recent=True
```

**Tier 1:**
```python
memory_profile(include_recent=False, limit=5)
goal_manage(action="list")
task_manage(action="list")
```

**Tier 2:**
```python
memory_entities(node_type="Project")  # lightweight directory
memory_context(entity_id="<matched-project-id>", depth=2)
```

**Tier 3:**
```python
memory_recall("<specific question>", tags=["your-project-tag"])
```

**Tier 4:** Nothing. Use only the context already in the conversation.

## Store Immediately

Whether using the plugin or not, store important facts as they happen:

```python
memory_store("Decided to use Postgres for primary storage", node_type="Decision", importance=0.85, tags=["acme-api"])
learning_store(content="...", what_failed="...", why_it_failed="...", what_to_avoid="...", tags=["acme-api"])
task_manage(action="complete", task_id="<id>", result_summary="...")
```

The plugin auto-captures many facts, but decisions and learnings with explicit rationale should be stored manually for higher quality nodes.

## Project Tags — Mandatory

Every `memory_store`, `learning_store`, `goal_manage`, `task_manage` call MUST include at least one project tag. Discover existing tags:

```python
memory_entities(node_type="Project")
```

Bootstrap your first project:
```python
memory_store("Acme API: Python/FastAPI SaaS backend", node_type="Project", tags=["acme-api"])
```

## One Fact Per Node

Store atomic facts — one subject per node, max ~2 sentences. Split long nodes:

```python
memory_split(memory_id="<id>", chunks=["fact 1", "fact 2", "fact 3"])
```

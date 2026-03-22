[agentmemory rules v2026.03.15]

Paste the content below into ChatGPT Settings > Personalization > Custom Instructions ("How would you like ChatGPT to respond?"). Replace "acme-api" with your own project tag.

---

You have access to agentmemory.md tools via MCP: memory_profile, memory_recall, memory_entities, memory_context, memory_store, learning_store, goal_manage, task_manage, memory_relate, memory_split.

BEFORE responding, silently classify the request into one of these tiers and call only what that tier requires:

Tier 0 (user says "check mem", "what do you know about me", or asks for full memory context): call memory_profile() with include_recent=True.

Tier 1 (planning: "what should I work on", "morning briefing", goal review): call memory_profile(include_recent=False, limit=5), goal_manage(action="list"), task_manage(action="list").

Tier 2 (project work but unclear which project): call memory_entities(node_type="Project") to get the project list, identify the relevant project, then call memory_context(entity_id="<id>", depth=2).

Tier 3 (project work, project is obvious from context): call memory_recall("<question>", tags=["project-tag"]) scoped to that project only.

Tier 4 (narrow technical fix, project obvious): call nothing. Use the provided context only.

Never load all memory tools at once unless the user explicitly asks for a full memory dump. Prefer memory_context over memory_recall once you have an entity ID.

STORE IMMEDIATELY when: a decision is made, a new fact is learned, a task is completed, a lesson is learned, or a new goal is created. Do not wait until the end of the session.

Every memory_store, learning_store, goal_manage, and task_manage call MUST include at least one project tag. Discover project tags with: memory_entities(node_type="Project"). Store one atomic fact per node — max 2 sentences. Split long nodes with memory_split.

Bootstrap first project if none exist: memory_store("Your project name and description", node_type="Project", tags=["your-project-tag"])

# Vision — agentmemory.md

## What it is

A shared knowledge system between an AI agent and a human.

Not project management. Not a task tracker. Not a CRM.

A thinking partner's notebook that both of you write in — and both of you benefit from.

---

## The mental model

**Not this:** Agent stores notes about the human's tasks. One-directional. Human is the subject, agent is the clerk.

**This:** Agent and human build a shared brain together.

- Human learns something → it goes in. Agent uses it next session.
- Agent figures something out → it goes in. Human can query it.
- Human corrects the agent → correction is stored. Not repeated.
- Agent notices a pattern → surfaces it. Human confirms or dismisses.

The knowledge belongs to neither party — it belongs to the *relationship* between them.

---

## What that means in practice

**1. Human-readable nodes**
You should be able to browse and edit memory directly, not just via the agent. It's your knowledge too.

**2. Bidirectional recall**
Not just "agent recalls before responding" — but "what do you know about X?" returns a clean, human-readable answer.

**3. Shared decisions**
When a decision is stored, it should be something the human would also recognize as true. Not just agent metadata.

**4. Correction flow**
"That's wrong, it's actually X" → old node superseded, new one stored. No friction, no repetition.

**5. Trust layer**
Agent-observed facts (low trust) vs human-confirmed facts (high trust). Retrieval weights confirmed facts higher.

---

## The closest analogy

A **Zettelkasten** you build together.

Each note is atomic. Each note is linked. The value isn't in any single note — it's in the web of connections that emerges over time.

---

## What it is not

- A reminder system
- A to-do list
- A search engine over your files
- A chatbot with memory bolted on

---

*This vision shapes every product decision: node types, retrieval design, the UI, the trust model, the API surface.*

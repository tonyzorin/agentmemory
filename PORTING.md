# Porting shared retrieval code from Company Brain

**Brain** (`cloudberry/brain`) is the company knowledge graph (shared with colleagues). **agentmemory** is personal single-user memory. Retrieval fixes land in Brain first; port them here by hand.

## Shared files (same lineage)

| agentmemory | Brain |
|-------------|--------|
| `agentmemory/db/redis_client.py` | `app/db/redis_client.py` |
| `agentmemory/db/age_client.py` | `app/db/age_client.py` |
| `agentmemory/core/retrieval.py` | `app/core/retrieval.py` |
| `agentmemory/core/embeddings.py` | `app/core/embeddings.py` |

## Last ported from Brain

- **Commit:** _(set when you port)_
- **Notes:** search_mode on recall, Redis `keyword_vector_fusion`, `/health`, graph path tools

## Do not port into agentmemory

- PII guard (`app/core/pii.py`)
- Canon / quarantine / review portal
- `brain_users`, audit log, multi-user OAuth
- Company MCP tools (`customer_upsert`, live CRM/Desk/YouTrack gateway)

## Optional: promote personal → company

Use a deliberate human step (PII check + Brain quarantine). No automatic sync.

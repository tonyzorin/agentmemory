# Connecting OpenClaw to agentmemory.md

## Current deployment (VM 817 / ai-agent)

| Component | Location |
|-----------|----------|
| agentmemory server | Docker `agentmemory-app` on `192.168.122.17:8081` |
| agentmemory repo | `/home/anton/agentmemory` |
| OpenClaw plugin | `/home/anton/agentmemory-plugin` (GitLab: `agentmemory-openclaw-plugin`) |
| OpenClaw config | `/home/anton/.openclaw/openclaw.json` |
| Gateway unit | `~/.config/systemd/user/openclaw-gateway.service` |

Public HTTPS for Cursor/Grok: `https://mem.agentmemory.md/mcp` (Caddy on hetzner → Tailscale `100.120.20.16:8081`).

## Auth: use `am_` Bearer token, not Google Sign-In

The agentmemory server supports **two auth paths on the same `/mcp` endpoint**:

1. **Static `am_` API keys** — long-lived, for headless clients (OpenClaw gateway, Cursor).
2. **Google OAuth `amo_` tokens** — interactive browser consent, 7-day TTL (Grok, web).

OpenClaw is a headless daemon. It **cannot** complete Google Sign-In and should **not** use `amo_` tokens. Use an `am_` key instead. Google Sign-In remains available for browser clients; both paths coexist.

Create a token (if needed):

```bash
cd /home/anton/agentmemory
docker exec agentmemory-app python -m agentmemory.cli.main token create
# Add printed hash to AGENTMEMORY_TOKEN_HASHES in .env, restart container
```

## OpenClaw plugin setup

### 1. Install / update plugin

```bash
cd /home/anton/agentmemory-plugin
git pull origin main
npm install
npm run build
```

### 2. Enable in `~/.openclaw/openclaw.json`

```json
{
  "plugins": {
    "load": {
      "paths": ["/home/anton/agentmemory-plugin"]
    },
    "entries": {
      "agentmemory": {
        "enabled": true,
        "config": {
          "memoryUrl": "http://192.168.122.17:8081/mcp",
          "recall": { "enabled": true, "limit": 8, "minScore": 0.55 },
          "capture": { "enabled": true, "minTurnLength": 100, "dedupTtlHours": 4 }
        }
      }
    }
  }
}
```

Use the Docker bridge IP (`192.168.122.17`), not `localhost` — the container is not bound to loopback.

### 3. Provide the Bearer token via environment (recommended)

Keep secrets out of `openclaw.json`:

```bash
# ~/.openclaw/agentmemory.env  (chmod 600)
AGENTMEMORY_TOKEN=am_...
```

Wire into the gateway systemd unit (`~/.config/systemd/user/openclaw-gateway.service`):

```ini
[Service]
EnvironmentFile=/home/anton/.openclaw/agentmemory.env
```

Then reload and restart:

```bash
systemctl --user daemon-reload
systemctl --user restart openclaw-gateway.service
```

### 4. Verify

```bash
openclaw logs --limit 20 | grep agentmemory
```

Expected:

```
[agentmemory] started — recall: true, capture: true, auth: enabled, url: http://192.168.122.17:8081/mcp
[agentmemory] entity cache refreshed: N entities
```

If you see `auth: disabled` or `HTTP 401`, the gateway process is not loading `AGENTMEMORY_TOKEN`. Check `EnvironmentFile` and restart.

**Note:** `openclaw agent` from an SSH shell without sourcing the env file runs an embedded agent that lacks the token. Gateway + Telegram use the systemd env and work correctly.

## Remote access (Cursor / other machines)

In `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "agentmemory": {
      "url": "https://mem.agentmemory.md/mcp",
      "headers": {
        "Authorization": "Bearer am_..."
      }
    }
  }
}
```

Use HTTPS (`mem.agentmemory.md`), not `http://` — clients strip `Authorization` on redirect.

## Available MCP tools

| Tool | Description |
|------|-------------|
| `memory_store` | Save memory/learning/decision/preference |
| `memory_recall` | Semantic search across all knowledge |
| `memory_relate` | Link entities in the knowledge graph |
| `memory_context` | Get full context for any entity |
| `memory_forget` | Remove outdated information |
| `memory_profile` | Get user profile summary |
| `goal_manage` | Create/list/complete goals and OKRs |
| `task_manage` | Create/complete tasks |
| `timeline` | Query what happened in a time range |
| `learning_store` | Record failed experiments |
| `workflow_store` | Save reusable processes |

## CLI usage (on VM)

```bash
docker exec -it agentmemory-app python -m agentmemory.cli.main stats
docker exec -it agentmemory-app python -m agentmemory.cli.main recall "openclaw gateway"
```

Or from the repo with local venv — see main README.

## Nightly hygiene routine

Add a cron on ai-agent (example: 03:00 daily):

```bash
0 3 * * * anton docker exec agentmemory-app python -m agentmemory.cli.main wire-orphans >> /var/log/agentmemory-hygiene.log 2>&1
15 3 * * * anton docker exec agentmemory-app python -m agentmemory.cli.main consolidate --threshold 0.88 --type Memory >> /var/log/agentmemory-hygiene.log 2>&1
30 3 * * * anton docker exec agentmemory-app python -m agentmemory.cli.main gc --yes >> /var/log/agentmemory-hygiene.log 2>&1
```

Ensure `PROJECT_ANCHORS_JSON` is set in the container `.env` so `wire-orphans` knows your tag → Project UUID map.

| Step | Command | Purpose |
|------|---------|---------|
| Wire orphans | `wire-orphans` | Link tagged facts → project anchors via `ABOUT` edges |
| Consolidate | `consolidate --threshold 0.88` | Merge near-duplicate Memory nodes |
| GC | `gc --yes` | Remove expired Task/Initiative nodes |

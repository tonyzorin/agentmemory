"""
Configuration for agentmemory.md.
Reads from environment variables or .env file.
"""

from __future__ import annotations

import json

from pydantic_settings import BaseSettings, SettingsConfigDict


def parse_project_anchors(raw: str) -> dict[str, str]:
    """Parse PROJECT_ANCHORS_JSON into a lowercase tag → project UUID map."""
    if not raw or raw.strip() in ("", "{}"):
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("PROJECT_ANCHORS_JSON must be a JSON object")
    return {str(k).lower(): str(v) for k, v in data.items()}


def load_project_anchors_file(path: str) -> dict[str, str]:
    """Load tag → project UUID map from a JSON file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object mapping tags to UUIDs")
    return {str(k).lower(): str(v) for k, v in data.items()}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # PostgreSQL + Apache AGE
    database_url: str = "postgresql://openclaw:openclaw@localhost:5433/openclaw_memory"

    # Redis 8.4
    redis_url: str = "redis://localhost:6380/0"

    # Embedding model
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768
    embedding_cache_ttl: int = 86400  # 24 hours

    # MCP server (SSE/HTTP transport)
    mcp_host: str = "0.0.0.0"
    mcp_port: int = 8081

    # Graph name in Apache AGE
    graph_name: str = "memory_graph"

    # Log level
    log_level: str = "INFO"

    # Reranker (cross-encoder) — disabled by default.
    # Improves retrieval precision significantly at corpus sizes of ~1000+ nodes.
    # Enable with: RERANKER_ENABLED=true
    reranker_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    reranker_top_k: int = 20  # number of hybrid-search candidates to rerank

    # Tag → Project entity UUID map for wire-orphans (deployment-specific; not in source)
    # Example: PROJECT_ANCHORS_JSON={"my-app":"00000000-0000-0000-0000-000000000001"}
    project_anchors_json: str = "{}"

    # MCP HTTP/SSE Bearer auth (stdio transport is never authenticated)
    # AGENTMEMORY_TOKEN_HASHES: comma-separated SHA-256 hex digests (from `mem token create`)
    agentmemory_auth_required: bool = False
    agentmemory_token_hashes: str = ""
    agentmemory_token_pepper: str = ""

    # OAuth for browser/native MCP clients. App serves /.well-known for LAN
    # (OpenClaw). Public mem.agentmemory.md still 404s well-known via Caddy.
    agentmemory_oauth_enabled: bool = False
    agentmemory_oauth_password_hash: str = ""
    agentmemory_oauth_client_id: str = "agentmemory"
    agentmemory_public_base_url: str = "https://mem.agentmemory.md"
    agentmemory_oauth_redirect_allowlist: str = (
        "https://grok.com/connectors-oauth-exchange-code/"
    )
    agentmemory_oauth_allowed_email: str = "tonyzorin@gmail.com"
    agentmemory_google_client_id: str = ""
    agentmemory_google_client_secret: str = ""

    @property
    def oauth_allowed_email(self) -> str:
        return self.agentmemory_oauth_allowed_email.strip().lower()

    @property
    def oauth_google_enabled(self) -> bool:
        return bool(
            self.agentmemory_google_client_id.strip()
            and self.agentmemory_google_client_secret.strip()
        )

    @property
    def project_anchors(self) -> dict[str, str]:
        return parse_project_anchors(self.project_anchors_json)

    @property
    def oauth_redirect_allowlist(self) -> tuple[str, ...]:
        parts = [
            u.strip()
            for u in self.agentmemory_oauth_redirect_allowlist.split(",")
            if u.strip()
        ]
        return tuple(parts) if parts else ("https://grok.com/connectors-oauth-exchange-code/",)

    @property
    def http_auth_enabled(self) -> bool:
        """True when HTTP/SSE should require a valid Bearer token."""
        return (
            self.agentmemory_auth_required
            or bool(self.agentmemory_token_hashes.strip())
            or self.agentmemory_oauth_enabled
        )


# Singleton
settings = Settings()

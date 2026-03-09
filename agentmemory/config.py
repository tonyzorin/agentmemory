"""
Configuration for agentmemory.md.
Reads from environment variables or .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


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
    mcp_port: int = 8006

    # Graph name in Apache AGE
    graph_name: str = "memory_graph"

    # Log level
    log_level: str = "INFO"


# Singleton
settings = Settings()

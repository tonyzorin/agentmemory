"""OAuth token persistence: in-memory (tests) or Redis (production).

Auth codes and Google login sessions stay in process memory (10-minute TTL,
one-time use). Access and refresh tokens are hashed and stored so they
survive container restarts when Redis is configured.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

import redis


@dataclass
class PendingAuthCode:
    code: str
    client_id: str
    redirect_uri: str
    code_challenge: str
    code_challenge_method: str
    scopes: list[str]
    state: str | None
    expires_at: float
    subject: str | None = None


@dataclass
class StoredAccessToken:
    token_hash: str
    client_id: str
    scopes: list[str]
    expires_at: float
    subject: str | None = None


@dataclass
class StoredRefreshToken:
    token_hash: str
    client_id: str
    scopes: list[str]
    expires_at: float
    subject: str | None = None
    access_token_hash: str | None = None


def _alive(expires_at: float) -> bool:
    return expires_at > time.time()


class IssuedTokenStore:
    """Hashed amo_/amr_ tokens, with optional Redis backing."""

    def __init__(
        self,
        *,
        redis_url: str | None = None,
        key_prefix: str = "oauth:",
    ) -> None:
        self.codes: dict[str, PendingAuthCode] = {}
        self.oauth_sessions: dict[str, object] = {}
        # In-memory maps used when redis_url is unset (unit tests).
        self.access_tokens: dict[str, StoredAccessToken] = {}
        self.refresh_tokens: dict[str, StoredRefreshToken] = {}
        self._prefix = key_prefix
        self._redis: redis.Redis | None = None
        if redis_url:
            self._redis = redis.from_url(redis_url, decode_responses=True)

    def _access_key(self, digest: str) -> str:
        return f"{self._prefix}access:{digest}"

    def _refresh_key(self, digest: str) -> str:
        return f"{self._prefix}refresh:{digest}"

    def purge_expired(self) -> None:
        now = time.time()
        self.codes = {k: v for k, v in self.codes.items() if v.expires_at > now}
        self.access_tokens = {
            k: v for k, v in self.access_tokens.items() if v.expires_at > now
        }
        self.refresh_tokens = {
            k: v for k, v in self.refresh_tokens.items() if v.expires_at > now
        }
        from agentmemory.mcp.oauth_google import PendingOAuthSession

        self.oauth_sessions = {
            k: v
            for k, v in self.oauth_sessions.items()
            if isinstance(v, PendingOAuthSession) and v.expires_at > now
        }

    def put_access(self, token: StoredAccessToken, ttl_seconds: int) -> None:
        if self._redis is not None:
            ttl = max(1, int(ttl_seconds))
            self._redis.set(self._access_key(token.token_hash), json.dumps(asdict(token)), ex=ttl)
            return
        self.access_tokens[token.token_hash] = token

    def get_access(self, digest: str) -> StoredAccessToken | None:
        if self._redis is not None:
            raw = self._redis.get(self._access_key(digest))
            if not raw:
                return None
            data = json.loads(raw)
            token = StoredAccessToken(**data)
            return token if _alive(token.expires_at) else None
        token = self.access_tokens.get(digest)
        if token is None or not _alive(token.expires_at):
            return None
        return token

    def delete_access(self, digest: str) -> None:
        if self._redis is not None:
            self._redis.delete(self._access_key(digest))
            return
        self.access_tokens.pop(digest, None)

    def put_refresh(self, token: StoredRefreshToken, ttl_seconds: int) -> None:
        if self._redis is not None:
            ttl = max(1, int(ttl_seconds))
            self._redis.set(self._refresh_key(token.token_hash), json.dumps(asdict(token)), ex=ttl)
            return
        self.refresh_tokens[token.token_hash] = token

    def get_refresh(self, digest: str) -> StoredRefreshToken | None:
        if self._redis is not None:
            raw = self._redis.get(self._refresh_key(digest))
            if not raw:
                return None
            data = json.loads(raw)
            token = StoredRefreshToken(**data)
            return token if _alive(token.expires_at) else None
        token = self.refresh_tokens.get(digest)
        if token is None or not _alive(token.expires_at):
            return None
        return token

    def delete_refresh(self, digest: str) -> None:
        if self._redis is not None:
            self._redis.delete(self._refresh_key(digest))
            return
        self.refresh_tokens.pop(digest, None)

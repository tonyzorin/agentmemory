"""
Bearer token authentication for agentmemory MCP HTTP/SSE transports.

Stores only hashed tokens (never raw secrets in config). Clients send:
  Authorization: Bearer am_<token>
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

from fastmcp.server.auth import TokenVerifier
from mcp.server.auth.provider import AccessToken


def generate_api_token() -> str:
    """Generate a new API token: am_ + 48 url-safe bytes (~384 bits)."""
    return "am_" + secrets.token_urlsafe(48)


def hash_token(token: str, pepper: str = "") -> str:
    """Return lowercase hex SHA-256 of token (optionally prefixed with pepper)."""
    material = f"{pepper}{token}" if pepper else token
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def parse_token_hashes(raw: str) -> set[str]:
    """Parse comma-separated hex digests from AGENTMEMORY_TOKEN_HASHES."""
    if not raw or not raw.strip():
        return set()
    hashes: set[str] = set()
    for part in raw.split(","):
        digest = part.strip().lower()
        if digest:
            hashes.add(digest)
    return hashes


class HashedBearerTokenVerifier(TokenVerifier):
    """
    Verify Bearer tokens against stored SHA-256 digests.

    Uses constant-time comparison. No OAuth discovery endpoints — safe for
    Cursor static header auth.
    """

    def __init__(self, token_hashes: set[str], pepper: str = "") -> None:
        super().__init__()
        if not token_hashes:
            raise ValueError("HashedBearerTokenVerifier requires at least one token hash")
        self._token_hashes = frozenset(token_hashes)
        self._pepper = pepper

    async def verify_token(self, token: str) -> AccessToken | None:
        if not token or not token.strip():
            return None
        digest = hash_token(token.strip(), self._pepper)
        for stored in self._token_hashes:
            if hmac.compare_digest(digest, stored):
                return AccessToken(
                    token=token,
                    client_id="agentmemory",
                    scopes=["memory:full"],
                )
        return None


def build_auth_verifier(
    auth_required: bool,
    token_hashes_raw: str,
    pepper: str = "",
) -> HashedBearerTokenVerifier | None:
    """
    Build verifier when auth is enabled or hashes are configured.

    Raises SystemExit message via ValueError when auth_required but no hashes.
    """
    hashes = parse_token_hashes(token_hashes_raw)
    if auth_required and not hashes:
        raise ValueError(
            "AGENTMEMORY_AUTH_REQUIRED=true but AGENTMEMORY_TOKEN_HASHES is empty. "
            "Run: mem token create"
        )
    if not hashes:
        return None
    return HashedBearerTokenVerifier(hashes, pepper=pepper)

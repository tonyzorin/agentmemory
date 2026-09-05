"""
Bearer token authentication for agentmemory MCP HTTP/SSE transports.

Stores only hashed tokens (never raw secrets in config). Clients send:
  Authorization: Bearer am_<token>
"""

from __future__ import annotations

import hmac
from typing import TYPE_CHECKING

from fastmcp.server.auth import MultiAuth, TokenVerifier
from mcp.server.auth.provider import AccessToken

from agentmemory.mcp.tokens import (
    generate_api_token,
    hash_token,
    parse_token_hashes,
)

if TYPE_CHECKING:
    from agentmemory.mcp.oauth import OAuthAuthorizationServer


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


def build_http_auth(
    *,
    auth_required: bool,
    token_hashes_raw: str,
    pepper: str = "",
    oauth_enabled: bool = False,
    oauth_server: OAuthAuthorizationServer | None = None,
) -> HashedBearerTokenVerifier | MultiAuth | None:
    """
    Build HTTP auth: hashed Bearer API keys and/or OAuth-issued tokens.

    Uses MultiAuth with server=None so FastMCP does not auto-mount OAuth
    discovery. OpenClaw discovery is registered as custom_route on the app
    (LAN). Caddy 404s /.well-known on mem.agentmemory.md so Cursor keeps
    static Bearer headers. /authorize, /token, and /register are custom_route.
    """
    verifiers: list[TokenVerifier] = []

    bearer = build_auth_verifier(auth_required, token_hashes_raw, pepper)
    if bearer is not None:
        verifiers.append(bearer)

    if oauth_enabled:
        if oauth_server is None:
            raise ValueError(
                "AGENTMEMORY_OAUTH_ENABLED=true but OAuth server was not initialized"
            )
        verifiers.append(oauth_server.verifier)

    if not verifiers:
        if auth_required or oauth_enabled:
            raise ValueError(
                "HTTP auth enabled but no verifiers configured. "
                "Set AGENTMEMORY_TOKEN_HASHES and/or AGENTMEMORY_OAUTH_PASSWORD_HASH."
            )
        return None

    if len(verifiers) == 1:
        return verifiers[0]

    return MultiAuth(server=None, verifiers=verifiers)


# Re-export token helpers so existing imports from auth keep working.
__all__ = [
    "HashedBearerTokenVerifier",
    "build_auth_verifier",
    "build_http_auth",
    "generate_api_token",
    "hash_token",
    "parse_token_hashes",
]

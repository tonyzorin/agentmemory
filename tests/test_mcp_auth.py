"""Tests for Bearer token auth."""

from __future__ import annotations

import asyncio

import pytest

from agentmemory.mcp.auth import (
    HashedBearerTokenVerifier,
    build_auth_verifier,
    build_http_auth,
)
from agentmemory.mcp.tokens import generate_api_token, hash_token, parse_token_hashes


class TestTokenHashing:
    def test_generate_token_prefix(self):
        tok = generate_api_token()
        assert tok.startswith("am_")
        assert len(tok) > 20

    def test_hash_deterministic(self):
        assert hash_token("am_test") == hash_token("am_test")

    def test_hash_with_pepper(self):
        assert hash_token("am_test", "pepper") != hash_token("am_test")

    def test_parse_token_hashes(self):
        raw = "abc123, DEF456 ,"
        assert parse_token_hashes(raw) == {"abc123", "def456"}


class TestHashedBearerTokenVerifier:
    @pytest.mark.asyncio
    async def test_valid_token(self):
        raw = generate_api_token()
        verifier = HashedBearerTokenVerifier({hash_token(raw)})
        result = await verifier.verify_token(raw)
        assert result is not None
        assert result.client_id == "agentmemory"

    @pytest.mark.asyncio
    async def test_invalid_token(self):
        raw = generate_api_token()
        verifier = HashedBearerTokenVerifier({hash_token(raw)})
        assert await verifier.verify_token("am_wrong") is None
        assert await verifier.verify_token("") is None

    @pytest.mark.asyncio
    async def test_pepper_must_match(self):
        raw = generate_api_token()
        verifier = HashedBearerTokenVerifier({hash_token(raw, "secret")}, pepper="secret")
        assert await verifier.verify_token(raw) is not None
        verifier_no_pepper = HashedBearerTokenVerifier({hash_token(raw, "secret")})
        assert await verifier_no_pepper.verify_token(raw) is None


class TestBuildAuthVerifier:
    def test_disabled_when_no_hashes_and_not_required(self):
        assert build_auth_verifier(False, "") is None

    def test_enabled_when_hashes_present(self):
        digest = hash_token("am_test")
        verifier = build_auth_verifier(False, digest)
        assert verifier is not None

    def test_raises_when_required_without_hashes(self):
        with pytest.raises(ValueError, match="AGENTMEMORY_AUTH_REQUIRED"):
            build_auth_verifier(True, "")


class TestHttpAuthIntegration:
    def test_http_requires_bearer_when_auth_enabled(self):
        from fastmcp import FastMCP
        from starlette.testclient import TestClient

        raw = generate_api_token()
        verifier = HashedBearerTokenVerifier({hash_token(raw)})
        server = FastMCP("test-auth", auth=verifier)

        @server.tool()
        def ping() -> str:
            return "pong"

        init_body = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            },
            "id": 1,
        }

        with TestClient(server.http_app()) as client:
            denied = client.post("/mcp", json=init_body)
            assert denied.status_code == 401

            allowed = client.post(
                "/mcp",
                json=init_body,
                headers={"Authorization": f"Bearer {raw}"},
            )
            assert allowed.status_code != 401

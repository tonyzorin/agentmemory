"""Tests for OAuth authorization server."""

from __future__ import annotations

import base64
import hashlib
import secrets
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp import FastMCP
from fastmcp.server.auth import MultiAuth
from starlette.testclient import TestClient

from agentmemory.mcp.auth import HashedBearerTokenVerifier, build_http_auth
from agentmemory.mcp.oauth import (
    ACCESS_TOKEN_PREFIX,
    ACCESS_TOKEN_TTL_SECONDS,
    OAUTH_SESSION_COOKIE,
    REFRESH_TOKEN_PREFIX,
    OAuthAuthorizationServer,
    PasswordRateLimiter,
    _normalize_scopes,
    _redirect_allowed,
    _verify_pkce,
)
from agentmemory.mcp.tokens import generate_api_token, hash_token


def _make_pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


OAUTH_REDIRECT = "https://grok.com/connectors-oauth-exchange-code/"
CLIENT_ID = "agentmemory"
OAUTH_PASSWORD = "test-oauth-secret"
ALLOWED_EMAIL = "tonyzorin@gmail.com"
GOOGLE_CLIENT_ID = "test-google-client-id"
GOOGLE_CLIENT_SECRET = "test-google-client-secret"


@pytest.fixture
def oauth_password_hash() -> str:
    return hash_token(OAUTH_PASSWORD)


@pytest.fixture
def oauth_server(oauth_password_hash: str) -> OAuthAuthorizationServer:
    return OAuthAuthorizationServer(
        password_hash=oauth_password_hash,
        allowed_email=ALLOWED_EMAIL,
        client_id=CLIENT_ID,
        redirect_allowlist=(OAUTH_REDIRECT,),
    )


@pytest.fixture
def google_oauth_server() -> OAuthAuthorizationServer:
    return OAuthAuthorizationServer(
        google_client_id=GOOGLE_CLIENT_ID,
        google_client_secret=GOOGLE_CLIENT_SECRET,
        allowed_email=ALLOWED_EMAIL,
        client_id=CLIENT_ID,
        redirect_allowlist=(OAUTH_REDIRECT,),
    )


@pytest.fixture
def oauth_mcp(oauth_server: OAuthAuthorizationServer) -> FastMCP:
    server = FastMCP("test-oauth-mcp", auth=oauth_server.verifier)
    oauth_server.register_routes(server)

    @server.tool()
    def ping() -> str:
        return "pong"

    return server


@pytest.fixture
def google_oauth_mcp(google_oauth_server: OAuthAuthorizationServer) -> FastMCP:
    server = FastMCP("test-google-oauth-mcp", auth=google_oauth_server.verifier)
    google_oauth_server.register_routes(server)

    @server.tool()
    def ping() -> str:
        return "pong"

    return server


def _authorize_params(challenge: str, **extra) -> dict[str, str]:
    return {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": OAUTH_REDIRECT,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "scope": "memory:full",
        **extra,
    }


class TestPkce:
    def test_verify_s256(self):
        verifier, challenge = _make_pkce_pair()
        assert _verify_pkce(verifier, challenge, "S256") is True
        assert _verify_pkce(verifier + "x", challenge, "S256") is False
        assert _verify_pkce(verifier, challenge, "plain") is False


class TestRedirectAllowlist:
    def test_exact_match_allowed(self):
        assert _redirect_allowed(OAUTH_REDIRECT, (OAUTH_REDIRECT,)) is True

    def test_rejects_query_string(self):
        uri = OAUTH_REDIRECT + "?extra=1"
        assert _redirect_allowed(uri, (OAUTH_REDIRECT,)) is False

    def test_rejects_fragment(self):
        uri = OAUTH_REDIRECT + "#fragment"
        assert _redirect_allowed(uri, (OAUTH_REDIRECT,)) is False

    def test_rejects_http_non_loopback(self):
        assert _redirect_allowed("http://grok.com/callback", (OAUTH_REDIRECT,)) is False

    def test_allows_ipv4_loopback(self):
        assert _redirect_allowed("http://127.0.0.1:9876/callback", (OAUTH_REDIRECT,)) is True

    def test_allows_localhost_loopback(self):
        assert _redirect_allowed("http://localhost:8989/oauth/callback", (OAUTH_REDIRECT,)) is True

    def test_allows_ipv6_loopback(self):
        assert _redirect_allowed("http://[::1]:9999/callback", (OAUTH_REDIRECT,)) is True

    def test_rejects_loopback_without_port(self):
        assert _redirect_allowed("http://127.0.0.1/callback", (OAUTH_REDIRECT,)) is False

    def test_rejects_loopback_query(self):
        assert _redirect_allowed("http://127.0.0.1:9876/callback?x=1", (OAUTH_REDIRECT,)) is False

    def test_rejects_evil_https(self):
        assert _redirect_allowed("https://evil.example/callback", (OAUTH_REDIRECT,)) is False


class TestScopeAllowlist:
    def test_default_scope(self):
        scopes, err = _normalize_scopes("")
        assert err is None
        assert scopes == ["memory:full"]

    def test_memory_full_allowed(self):
        scopes, err = _normalize_scopes("memory:full")
        assert err is None
        assert scopes == ["memory:full"]

    def test_admin_scope_rejected(self):
        scopes, err = _normalize_scopes("admin")
        assert scopes is None
        assert err == "invalid_scope"


class TestOAuthIssuedTokenVerifier:
    @pytest.mark.asyncio
    async def test_valid_oauth_token_hashed(self, oauth_server: OAuthAuthorizationServer):
        from agentmemory.mcp.oauth import StoredAccessToken

        raw = ACCESS_TOKEN_PREFIX + secrets.token_urlsafe(16)
        digest = hash_token(raw, oauth_server._password_pepper)
        oauth_server.store.put_access(
            StoredAccessToken(
                token_hash=digest,
                client_id=CLIENT_ID,
                scopes=["memory:full"],
                expires_at=time.time() + 3600,
            ),
            ttl_seconds=3600,
        )
        result = await oauth_server.verifier.verify_token(raw)
        assert result is not None
        assert result.client_id == CLIENT_ID

    @pytest.mark.asyncio
    async def test_plaintext_token_not_stored(self, oauth_server: OAuthAuthorizationServer):
        raw = ACCESS_TOKEN_PREFIX + secrets.token_urlsafe(16)
        assert raw not in oauth_server.store.access_tokens
        assert await oauth_server.verifier.verify_token(raw) is None

    @pytest.mark.asyncio
    async def test_invalid_oauth_token(self, oauth_server: OAuthAuthorizationServer):
        assert await oauth_server.verifier.verify_token("amo_bad") is None


class TestOAuthFlow:
    def test_full_pkce_flow(self, oauth_mcp: FastMCP):
        verifier, challenge = _make_pkce_pair()
        state = "xyz-state"

        with TestClient(oauth_mcp.http_app()) as client:
            auth_get = client.get("/authorize", params=_authorize_params(challenge, state=state))
            assert auth_get.status_code == 200
            assert "password" in auth_get.text

            auth_post = client.post(
                "/authorize",
                data=_authorize_params(challenge, state=state, email=ALLOWED_EMAIL, password=OAUTH_PASSWORD),
                follow_redirects=False,
            )
            assert auth_post.status_code == 302
            location = auth_post.headers["location"]
            assert location.startswith(OAUTH_REDIRECT.rstrip("/"))
            from urllib.parse import parse_qs, urlparse

            parsed_qs = parse_qs(urlparse(location).query)
            assert "code" in parsed_qs
            assert parsed_qs.get("state", [None])[0] == state
            code = parsed_qs["code"][0]

            token_resp = client.post(
                "/token",
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": OAUTH_REDIRECT,
                    "code_verifier": verifier,
                    "client_id": CLIENT_ID,
                },
            )
            assert token_resp.status_code == 200
            body = token_resp.json()
            assert body["token_type"] == "Bearer"
            assert body["access_token"].startswith(ACCESS_TOKEN_PREFIX)
            assert body["refresh_token"].startswith(REFRESH_TOKEN_PREFIX)
            assert body["expires_in"] == ACCESS_TOKEN_TTL_SECONDS

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
            allowed = client.post(
                "/mcp",
                json=init_body,
                headers={"Authorization": f"Bearer {body['access_token']}"},
            )
            assert allowed.status_code != 401

    def test_bad_password(self, oauth_mcp: FastMCP):
        _, challenge = _make_pkce_pair()
        with TestClient(oauth_mcp.http_app()) as client:
            resp = client.post(
                "/authorize",
                data=_authorize_params(challenge, email=ALLOWED_EMAIL, password="wrong"),
            )
            assert resp.status_code == 401
            assert "Incorrect password" in resp.text

    def test_wrong_email(self, oauth_mcp: FastMCP):
        _, challenge = _make_pkce_pair()
        with TestClient(oauth_mcp.http_app()) as client:
            resp = client.post(
                "/authorize",
                data=_authorize_params(
                    challenge, email="other@example.com", password=OAUTH_PASSWORD
                ),
            )
            assert resp.status_code == 403
            assert ALLOWED_EMAIL in resp.text

    def test_bad_redirect(self, oauth_mcp: FastMCP):
        _, challenge = _make_pkce_pair()
        with TestClient(oauth_mcp.http_app()) as client:
            resp = client.get(
                "/authorize",
                params=_authorize_params(challenge, redirect_uri="https://evil.example/callback"),
            )
            assert resp.status_code == 400

    def test_redirect_with_query_rejected(self, oauth_mcp: FastMCP):
        _, challenge = _make_pkce_pair()
        with TestClient(oauth_mcp.http_app()) as client:
            resp = client.get(
                "/authorize",
                params=_authorize_params(
                    challenge,
                    redirect_uri=OAUTH_REDIRECT + "?extra=1",
                ),
            )
            assert resp.status_code == 400

    def test_invalid_scope_rejected(self, oauth_mcp: FastMCP):
        _, challenge = _make_pkce_pair()
        with TestClient(oauth_mcp.http_app()) as client:
            resp = client.get(
                "/authorize",
                params=_authorize_params(challenge, scope="admin"),
            )
            assert resp.status_code == 400
            assert resp.json()["error"] == "invalid_scope"

    def test_bad_pkce_on_token(self, oauth_mcp: FastMCP):
        verifier, challenge = _make_pkce_pair()
        with TestClient(oauth_mcp.http_app()) as client:
            auth_post = client.post(
                "/authorize",
                data=_authorize_params(
                    challenge, email=ALLOWED_EMAIL, password=OAUTH_PASSWORD
                ),
                follow_redirects=False,
            )
            from urllib.parse import parse_qs, urlparse

            code = parse_qs(urlparse(auth_post.headers["location"]).query)["code"][0]

            token_resp = client.post(
                "/token",
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": OAUTH_REDIRECT,
                    "code_verifier": verifier + "bad",
                    "client_id": CLIENT_ID,
                },
            )
            assert token_resp.status_code == 400
            assert token_resp.json()["error"] == "invalid_grant"

    def test_loopback_redirect_authorize_ok(self, oauth_mcp: FastMCP):
        _, challenge = _make_pkce_pair()
        loopback = "http://127.0.0.1:41234/callback"
        with TestClient(oauth_mcp.http_app()) as client:
            resp = client.get(
                "/authorize",
                params=_authorize_params(challenge, redirect_uri=loopback),
            )
            assert resp.status_code == 200

    def test_well_known_and_dcr(self, oauth_mcp: FastMCP):
        with TestClient(oauth_mcp.http_app()) as client:
            as_meta = client.get("/.well-known/oauth-authorization-server")
            assert as_meta.status_code == 200
            body = as_meta.json()
            assert body["token_endpoint"].endswith("/token")
            assert "refresh_token" in body["grant_types_supported"]
            assert body["code_challenge_methods_supported"] == ["S256"]
            assert body["registration_endpoint"].endswith("/register")

            resource = client.get("/.well-known/oauth-protected-resource")
            assert resource.status_code == 200
            assert resource.json()["resource"].endswith("/mcp")
            assert resource.json()["authorization_servers"][0] == "http://testserver"

            resource_mcp = client.get("/.well-known/oauth-protected-resource/mcp")
            assert resource_mcp.status_code == 200

            oidc = client.get("/.well-known/openid-configuration")
            assert oidc.status_code == 404

            dcr = client.post(
                "/register",
                json={"redirect_uris": ["http://127.0.0.1:41234/callback"]},
            )
            assert dcr.status_code == 201
            assert dcr.json()["client_id"] == CLIENT_ID
            assert dcr.json()["token_endpoint_auth_method"] == "none"

            evil = client.post(
                "/register",
                json={"redirect_uris": ["https://evil.example/callback"]},
            )
            assert evil.status_code == 400
            assert evil.json()["error"] == "invalid_redirect_uri"

    def test_refresh_token_rotates(self, oauth_mcp: FastMCP):
        verifier, challenge = _make_pkce_pair()
        with TestClient(oauth_mcp.http_app()) as client:
            auth_post = client.post(
                "/authorize",
                data=_authorize_params(
                    challenge, email=ALLOWED_EMAIL, password=OAUTH_PASSWORD
                ),
                follow_redirects=False,
            )
            from urllib.parse import parse_qs, urlparse

            code = parse_qs(urlparse(auth_post.headers["location"]).query)["code"][0]
            first = client.post(
                "/token",
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": OAUTH_REDIRECT,
                    "code_verifier": verifier,
                    "client_id": CLIENT_ID,
                },
            )
            assert first.status_code == 200
            old_refresh = first.json()["refresh_token"]
            old_access = first.json()["access_token"]

            refreshed = client.post(
                "/token",
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": old_refresh,
                    "client_id": CLIENT_ID,
                },
            )
            assert refreshed.status_code == 200
            body = refreshed.json()
            assert body["access_token"].startswith(ACCESS_TOKEN_PREFIX)
            assert body["access_token"] != old_access
            assert body["refresh_token"].startswith(REFRESH_TOKEN_PREFIX)
            assert body["refresh_token"] != old_refresh

            replay = client.post(
                "/token",
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": old_refresh,
                    "client_id": CLIENT_ID,
                },
            )
            assert replay.status_code == 400
            assert replay.json()["error"] == "invalid_grant"


class TestPasswordRateLimit:
    def test_blocks_after_max_failures(self, oauth_mcp: FastMCP, oauth_server: OAuthAuthorizationServer):
        oauth_server._password_rate_limiter = PasswordRateLimiter(max_failures=3, window_seconds=900)
        _, challenge = _make_pkce_pair()
        with TestClient(oauth_mcp.http_app()) as client:
            for _ in range(3):
                resp = client.post(
                    "/authorize",
                    data=_authorize_params(challenge, email=ALLOWED_EMAIL, password="wrong"),
                )
                assert resp.status_code == 401

            blocked = client.post(
                "/authorize",
                data=_authorize_params(challenge, email=ALLOWED_EMAIL, password="wrong"),
            )
            assert blocked.status_code == 429
            assert blocked.json()["error"] == "access_denied"


class TestGoogleOAuthSecurity:
    def test_authorize_page_uses_human_description(self, google_oauth_mcp: FastMCP):
        _, challenge = _make_pkce_pair()
        with TestClient(google_oauth_mcp.http_app()) as client:
            auth_get = client.get("/authorize", params=_authorize_params(challenge))
        assert auth_get.status_code == 200
        assert "store and recall memories" in auth_get.text
        assert "Sign in with Google" in auth_get.text
        assert "Client:" not in auth_get.text
        assert "Allowed account:" not in auth_get.text
        assert ALLOWED_EMAIL not in auth_get.text

    def test_google_login_without_cookie_rejected(self, google_oauth_mcp: FastMCP):
        _, challenge = _make_pkce_pair()
        with TestClient(google_oauth_mcp.http_app()) as client:
            auth_get = client.get("/authorize", params=_authorize_params(challenge))
            assert auth_get.status_code == 200
            session_id = auth_get.cookies[OAUTH_SESSION_COOKIE]

            no_cookie = client.get(
                f"/oauth/google/login?session={session_id}",
                cookies={},
            )
            assert no_cookie.status_code == 403

            wrong_cookie = client.get(
                f"/oauth/google/login?session={session_id}",
                cookies={OAUTH_SESSION_COOKIE: "wrong-session-id"},
            )
            assert wrong_cookie.status_code == 403

    def test_google_login_with_matching_cookie_allowed(self, google_oauth_mcp: FastMCP):
        _, challenge = _make_pkce_pair()
        with TestClient(google_oauth_mcp.http_app()) as client:
            auth_get = client.get("/authorize", params=_authorize_params(challenge))
            session_id = auth_get.cookies[OAUTH_SESSION_COOKIE]

            resp = client.get(
                f"/oauth/google/login?session={session_id}",
                cookies={OAUTH_SESSION_COOKIE: session_id},
                follow_redirects=False,
            )
            assert resp.status_code == 302
            assert "accounts.google.com" in resp.headers["location"]

    @pytest.mark.asyncio
    async def test_unverified_email_rejected(self, google_oauth_mcp: FastMCP):
        _, challenge = _make_pkce_pair()
        with TestClient(google_oauth_mcp.http_app()) as client:
            auth_get = client.get("/authorize", params=_authorize_params(challenge))
            session_id = auth_get.cookies[OAUTH_SESSION_COOKIE]
            cookies = {OAUTH_SESSION_COOKIE: session_id}

            mock_token_resp = MagicMock()
            mock_token_resp.status_code = 200
            mock_token_resp.json.return_value = {"access_token": "google-access"}

            mock_user_resp = MagicMock()
            mock_user_resp.status_code = 200
            mock_user_resp.json.return_value = {
                "email": ALLOWED_EMAIL,
                "email_verified": False,
            }

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_token_resp)
            mock_client.get = AsyncMock(return_value=mock_user_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)

            with patch("agentmemory.mcp.oauth_google.httpx.AsyncClient", return_value=mock_client):
                resp = client.get(
                    f"/oauth/google/callback?state={session_id}&code=google-code",
                    cookies=cookies,
                    follow_redirects=False,
                )

            assert resp.status_code == 403
            assert "not verified" in resp.text.lower()

    @pytest.mark.asyncio
    async def test_verified_email_issues_code(self, google_oauth_mcp: FastMCP):
        _, challenge = _make_pkce_pair()
        with TestClient(google_oauth_mcp.http_app()) as client:
            auth_get = client.get("/authorize", params=_authorize_params(challenge))
            session_id = auth_get.cookies[OAUTH_SESSION_COOKIE]
            cookies = {OAUTH_SESSION_COOKIE: session_id}

            mock_token_resp = MagicMock()
            mock_token_resp.status_code = 200
            mock_token_resp.json.return_value = {"access_token": "google-access"}

            mock_user_resp = MagicMock()
            mock_user_resp.status_code = 200
            mock_user_resp.json.return_value = {
                "email": ALLOWED_EMAIL,
                "email_verified": True,
            }

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_token_resp)
            mock_client.get = AsyncMock(return_value=mock_user_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)

            with patch("agentmemory.mcp.oauth_google.httpx.AsyncClient", return_value=mock_client):
                resp = client.get(
                    f"/oauth/google/callback?state={session_id}&code=google-code",
                    cookies=cookies,
                    follow_redirects=False,
                )

            assert resp.status_code == 302
            assert resp.headers["location"].startswith(OAUTH_REDIRECT.rstrip("/"))


class TestMultiAuthWithOAuth:
    def test_bearer_and_oauth_both_work(self, oauth_password_hash: str):
        raw_bearer = generate_api_token()
        oauth = OAuthAuthorizationServer(
            password_hash=oauth_password_hash,
            allowed_email=ALLOWED_EMAIL,
            client_id=CLIENT_ID,
            redirect_allowlist=(OAUTH_REDIRECT,),
        )
        auth = build_http_auth(
            auth_required=False,
            token_hashes_raw=hash_token(raw_bearer),
            oauth_enabled=True,
            oauth_server=oauth,
        )
        assert isinstance(auth, MultiAuth)

        server = FastMCP("test-multi", auth=auth)
        oauth.register_routes(server)

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
            bearer_resp = client.post(
                "/mcp",
                json=init_body,
                headers={"Authorization": f"Bearer {raw_bearer}"},
            )
            assert bearer_resp.status_code != 401

            verifier, challenge = _make_pkce_pair()
            auth_post = client.post(
                "/authorize",
                data=_authorize_params(
                    challenge, email=ALLOWED_EMAIL, password=OAUTH_PASSWORD
                ),
                follow_redirects=False,
            )
            from urllib.parse import parse_qs, urlparse

            code = parse_qs(urlparse(auth_post.headers["location"]).query)["code"][0]
            token_resp = client.post(
                "/token",
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": OAUTH_REDIRECT,
                    "code_verifier": verifier,
                    "client_id": CLIENT_ID,
                },
            )
            oauth_token = token_resp.json()["access_token"]

            oauth_resp = client.post(
                "/mcp",
                json=init_body,
                headers={"Authorization": f"Bearer {oauth_token}"},
            )
            assert oauth_resp.status_code != 401


class TestRedisOAuthPersistence:
    @pytest.mark.asyncio
    async def test_tokens_survive_new_server_instance(self, oauth_password_hash: str):
        from tests.conftest import TEST_REDIS_URL

        prefix = f"test:oauth:{secrets.token_hex(8)}:"
        kwargs = dict(
            password_hash=oauth_password_hash,
            allowed_email=ALLOWED_EMAIL,
            client_id=CLIENT_ID,
            redirect_allowlist=(OAUTH_REDIRECT,),
            redis_url=TEST_REDIS_URL,
            redis_key_prefix=prefix,
        )
        first = OAuthAuthorizationServer(**kwargs)
        mcp = FastMCP("test-oauth-redis", auth=first.verifier)
        first.register_routes(mcp)

        verifier, challenge = _make_pkce_pair()
        with TestClient(mcp.http_app()) as client:
            auth_post = client.post(
                "/authorize",
                data=_authorize_params(
                    challenge, email=ALLOWED_EMAIL, password=OAUTH_PASSWORD
                ),
                follow_redirects=False,
            )
            from urllib.parse import parse_qs, urlparse

            code = parse_qs(urlparse(auth_post.headers["location"]).query)["code"][0]
            token_resp = client.post(
                "/token",
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": OAUTH_REDIRECT,
                    "code_verifier": verifier,
                    "client_id": CLIENT_ID,
                },
            )
            assert token_resp.status_code == 200
            access = token_resp.json()["access_token"]
            refresh = token_resp.json()["refresh_token"]

        restarted = OAuthAuthorizationServer(**kwargs)
        result = await restarted.verifier.verify_token(access)
        assert result is not None
        assert result.client_id == CLIENT_ID

        mcp2 = FastMCP("test-oauth-redis-2", auth=restarted.verifier)
        restarted.register_routes(mcp2)
        with TestClient(mcp2.http_app()) as client:
            refreshed = client.post(
                "/token",
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh,
                    "client_id": CLIENT_ID,
                },
            )
            assert refreshed.status_code == 200
            assert refreshed.json()["access_token"].startswith(ACCESS_TOKEN_PREFIX)

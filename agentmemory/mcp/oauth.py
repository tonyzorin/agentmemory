"""
Minimal OAuth 2.1 authorization server for browser and native MCP clients.

Password or Google Sign-In consent + PKCE (S256, token_endpoint_auth_method=none).

Discovery (/.well-known) and DCR (/register) are mounted on the app so LAN
clients such as OpenClaw can complete MCP OAuth. Public Cursor on
mem.agentmemory.md still 404s /.well-known via Caddy and keeps static Bearer.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import html
import ipaddress
import secrets
import time
from collections import defaultdict
from typing import TYPE_CHECKING
from urllib.parse import urlencode, urlparse

from fastmcp.server.auth import TokenVerifier
from mcp.server.auth.provider import AccessToken
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from agentmemory.mcp.oauth_store import (
    IssuedTokenStore,
    PendingAuthCode,
    StoredAccessToken,
    StoredRefreshToken,
)
from agentmemory.mcp.tokens import hash_token

if TYPE_CHECKING:
    import fastmcp

DEFAULT_REDIRECT_ALLOWLIST = ("https://grok.com/connectors-oauth-exchange-code/",)
DEFAULT_SCOPE = "memory:full"
ALLOWED_SCOPES = frozenset({DEFAULT_SCOPE})
ACCESS_TOKEN_PREFIX = "amo_"
REFRESH_TOKEN_PREFIX = "amr_"
OAUTH_SESSION_COOKIE = "am_oauth_sid"
CODE_TTL_SECONDS = 600
ACCESS_TOKEN_TTL_SECONDS = 3600  # 1 hour; clients refresh with amr_
REFRESH_TOKEN_TTL_SECONDS = 90 * 24 * 3600  # 90 days
PASSWORD_RATE_LIMIT_MAX = 5
PASSWORD_RATE_LIMIT_WINDOW_SECONDS = 900  # 15 minutes
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

# Re-export for existing test imports.
__all__ = [
    "ACCESS_TOKEN_PREFIX",
    "ACCESS_TOKEN_TTL_SECONDS",
    "REFRESH_TOKEN_PREFIX",
    "REFRESH_TOKEN_TTL_SECONDS",
    "OAuthAuthorizationServer",
    "PendingAuthCode",
    "StoredAccessToken",
    "StoredRefreshToken",
    "IssuedTokenStore",
]


class PasswordRateLimiter:
    """In-memory password failure limiter (single-replica deploy)."""

    def __init__(
        self,
        *,
        max_failures: int = PASSWORD_RATE_LIMIT_MAX,
        window_seconds: int = PASSWORD_RATE_LIMIT_WINDOW_SECONDS,
    ) -> None:
        self._max_failures = max_failures
        self._window_seconds = window_seconds
        self._failures: dict[str, list[float]] = defaultdict(list)

    def _client_key(self, request: Request) -> str:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
        if request.client and request.client.host:
            return request.client.host
        return "unknown"

    def _purge_old(self, key: str, now: float) -> None:
        cutoff = now - self._window_seconds
        self._failures[key] = [t for t in self._failures[key] if t > cutoff]

    def is_blocked(self, request: Request) -> bool:
        now = time.time()
        key = self._client_key(request)
        self._purge_old(key, now)
        return len(self._failures[key]) >= self._max_failures

    def record_failure(self, request: Request) -> None:
        now = time.time()
        key = self._client_key(request)
        self._purge_old(key, now)
        self._failures[key].append(now)

    def reset(self, request: Request) -> None:
        key = self._client_key(request)
        self._failures.pop(key, None)


class OAuthIssuedTokenVerifier(TokenVerifier):
    """Verify opaque OAuth access tokens (amo_…) issued by this AS."""

    def __init__(self, store: IssuedTokenStore, pepper: str = "") -> None:
        super().__init__()
        self._store = store
        self._pepper = pepper

    async def verify_token(self, token: str) -> AccessToken | None:
        if not token or not token.strip():
            return None
        raw = token.strip()
        if not raw.startswith(ACCESS_TOKEN_PREFIX):
            return None
        digest = hash_token(raw, self._pepper)
        stored = self._store.get_access(digest)
        if stored is None:
            return None
        return AccessToken(
            token=raw,
            client_id=stored.client_id,
            scopes=stored.scopes,
            subject=stored.subject,
        )


def _verify_pkce(code_verifier: str, code_challenge: str, method: str) -> bool:
    if method != "S256":
        return False
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    computed = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return hmac.compare_digest(computed, code_challenge)


def _is_loopback_redirect(redirect_uri: str) -> bool:
    """RFC 8252 native-app loopback: http://127.0.0.1|localhost|::1:<port>/<path>."""
    parsed = urlparse(redirect_uri)
    if parsed.scheme != "http":
        return False
    if parsed.query or parsed.fragment:
        return False
    if parsed.username or parsed.password:
        return False
    if not parsed.path or parsed.path == "/":
        return False
    host = (parsed.hostname or "").lower()
    if host not in LOOPBACK_HOSTS:
        return False
    if parsed.port is None or not (1 <= parsed.port <= 65535):
        return False
    if host not in {"localhost"}:
        try:
            ipaddress.ip_address(host)
        except ValueError:
            return False
    return True


def _redirect_allowed(redirect_uri: str, allowlist: tuple[str, ...]) -> bool:
    """HTTPS exact allowlist, or RFC 8252 http loopback. No query/fragment."""
    parsed = urlparse(redirect_uri)
    if parsed.query or parsed.fragment:
        return False
    if _is_loopback_redirect(redirect_uri):
        return True
    if parsed.scheme != "https":
        return False
    return redirect_uri in allowlist


def _normalize_scopes(scope_raw: str) -> tuple[list[str] | None, str | None]:
    """Only memory:full is accepted."""
    scopes = [s for s in scope_raw.replace(",", " ").split() if s]
    if not scopes:
        return [DEFAULT_SCOPE], None
    if set(scopes) != ALLOWED_SCOPES:
        return None, "invalid_scope"
    return scopes, None


def _oauth_session_cookie_value(request: Request) -> str | None:
    value = request.cookies.get(OAUTH_SESSION_COOKIE)
    return value if value else None


def _session_cookie_matches(request: Request, session_id: str) -> bool:
    cookie = _oauth_session_cookie_value(request)
    if not cookie or not session_id:
        return False
    return hmac.compare_digest(cookie, session_id)


def _set_oauth_session_cookie(response: Response, session_id: str) -> None:
    response.set_cookie(
        key=OAUTH_SESSION_COOKIE,
        value=session_id,
        max_age=CODE_TTL_SECONDS,
        httponly=True,
        secure=True,
        samesite="lax",
        path="/",
    )


def _clear_oauth_session_cookie(response: Response) -> None:
    response.delete_cookie(key=OAUTH_SESSION_COOKIE, path="/")


def _oauth_error(
    error: str,
    description: str,
    status: int = 400,
) -> JSONResponse:
    return JSONResponse(
        {"error": error, "error_description": description},
        status_code=status,
    )


def _consent_html(
    *,
    allowed_email: str,
    error: str | None,
    fields: dict[str, str],
    client_id: str = "",
    scope: str = "",
) -> str:
    del client_id, scope
    hidden = "".join(
        f'<input type="hidden" name="{html.escape(k)}" value="{html.escape(v)}">\n'
        for k, v in fields.items()
    )
    err_block = (
        f'<p style="color:#b91c1c">{html.escape(error)}</p>' if error else ""
    )
    email_value = html.escape(fields.get("email", allowed_email))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Authorize agentmemory</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 28rem; margin: 3rem auto; padding: 0 1rem; }}
    label {{ display: block; margin: 1rem 0 0.25rem; font-weight: 600; }}
    input[type=password], input[type=email] {{ width: 100%; padding: 0.5rem; box-sizing: border-box; }}
    button {{ margin-top: 1.25rem; padding: 0.6rem 1.2rem; cursor: pointer; }}
    .meta {{ color: #555; font-size: 0.9rem; margin: 0.75rem 0 0; line-height: 1.45; }}
  </style>
</head>
<body>
  <h1>Authorize access</h1>
  <p class="meta">This app wants to connect to your agentmemory so it can store and recall memories on your behalf.</p>
  {err_block}
  <form method="post" action="/authorize">
    {hidden}
    <label for="email">Email</label>
    <input id="email" name="email" type="email" value="{email_value}" autocomplete="email" required>
    <label for="password">agentmemory password</label>
    <input id="password" name="password" type="password" autocomplete="current-password" required>
    <button type="submit">Authorize</button>
  </form>
</body>
</html>"""


class OAuthAuthorizationServer:
    """OAuth AS: Google Sign-In or password consent."""

    def __init__(
        self,
        *,
        password_hash: str = "",
        password_pepper: str = "",
        allowed_email: str = "tonyzorin@gmail.com",
        google_client_id: str = "",
        google_client_secret: str = "",
        public_base_url: str = "https://mem.agentmemory.md",
        client_id: str = "agentmemory",
        redirect_allowlist: tuple[str, ...] = DEFAULT_REDIRECT_ALLOWLIST,
        code_ttl_seconds: int = CODE_TTL_SECONDS,
        access_token_ttl_seconds: int = ACCESS_TOKEN_TTL_SECONDS,
        refresh_token_ttl_seconds: int = REFRESH_TOKEN_TTL_SECONDS,
        redis_url: str | None = None,
        redis_key_prefix: str = "oauth:",
    ) -> None:
        if not allowed_email.strip():
            raise ValueError("OAuth allowed email is required")
        self._use_google = bool(
            google_client_id.strip() and google_client_secret.strip()
        )
        if not self._use_google and not password_hash.strip():
            raise ValueError(
                "OAuth requires Google client credentials or AGENTMEMORY_OAUTH_PASSWORD_HASH"
            )
        self._password_hash = password_hash.strip().lower()
        self._password_pepper = password_pepper
        self._allowed_email = allowed_email.strip().lower()
        self._google_client_id = google_client_id.strip()
        self._google_client_secret = google_client_secret.strip()
        base = public_base_url.rstrip("/")
        self._public_base_url = base
        self._google_callback_url = f"{base}/oauth/google/callback"
        self._client_id = client_id
        self._redirect_allowlist = redirect_allowlist
        self._code_ttl = code_ttl_seconds
        self._access_token_ttl = access_token_ttl_seconds
        self._refresh_token_ttl = refresh_token_ttl_seconds
        self.store = IssuedTokenStore(redis_url=redis_url, key_prefix=redis_key_prefix)
        self.verifier = OAuthIssuedTokenVerifier(self.store, pepper=password_pepper)
        self._password_rate_limiter = PasswordRateLimiter()

    def _check_password(self, password: str) -> bool:
        digest = hash_token(password, self._password_pepper)
        return hmac.compare_digest(digest, self._password_hash)

    def _check_email(self, email: str) -> bool:
        normalized = email.strip().lower()
        if not normalized:
            return False
        return hmac.compare_digest(normalized, self._allowed_email)

    def _validate_authorize_params(
        self, params: dict[str, str]
    ) -> tuple[dict[str, str] | None, str | None]:
        if params.get("response_type") != "code":
            return None, "unsupported_response_type"
        if params.get("client_id") != self._client_id:
            return None, "unauthorized_client"
        redirect_uri = params.get("redirect_uri", "")
        if not redirect_uri or not _redirect_allowed(
            redirect_uri, self._redirect_allowlist
        ):
            return None, "invalid_request"
        challenge = params.get("code_challenge", "")
        method = params.get("code_challenge_method", "")
        if not challenge or method != "S256":
            return None, "invalid_request"
        scopes, scope_err = _normalize_scopes(params.get("scope", DEFAULT_SCOPE))
        if scope_err:
            return None, scope_err
        validated = dict(params)
        validated["scope"] = " ".join(scopes or [DEFAULT_SCOPE])
        return validated, None

    def _redirect_with_auth_code(
        self,
        *,
        redirect_uri: str,
        code_challenge: str,
        code_challenge_method: str,
        scopes: list[str],
        client_state: str | None,
        subject: str,
    ) -> RedirectResponse:
        self.store.purge_expired()
        code = secrets.token_urlsafe(32)
        self.store.codes[code] = PendingAuthCode(
            code=code,
            client_id=self._client_id,
            redirect_uri=redirect_uri,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method,
            scopes=scopes,
            state=client_state,
            expires_at=time.time() + self._code_ttl,
            subject=subject,
        )
        query: dict[str, str] = {"code": code}
        if client_state:
            query["state"] = client_state
        sep = "&" if "?" in redirect_uri else "?"
        location = f"{redirect_uri}{sep}{urlencode(query)}"
        return RedirectResponse(location, status_code=302)

    def _request_origin(self, request: Request) -> str:
        """Origin the client actually called (LAN vs public)."""
        host = (request.headers.get("host") or "").strip()
        proto = (request.headers.get("x-forwarded-proto") or "").strip()
        if not proto:
            proto = request.url.scheme or "http"
        if host:
            return f"{proto}://{host}".rstrip("/")
        return self._public_base_url

    def _as_metadata(self, request: Request | None = None) -> dict[str, object]:
        public = self._public_base_url
        origin = self._request_origin(request) if request is not None else public
        return {
            "issuer": origin,
            "authorization_endpoint": f"{public}/authorize",
            "token_endpoint": f"{public}/token",
            "registration_endpoint": f"{public}/register",
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "response_types_supported": ["code"],
            "code_challenge_methods_supported": ["S256"],
            "token_endpoint_auth_methods_supported": ["none"],
            "scopes_supported": [DEFAULT_SCOPE],
        }

    def _resource_metadata(self, request: Request) -> dict[str, object]:
        origin = self._request_origin(request)
        return {
            "resource": f"{origin}/mcp",
            "authorization_servers": [origin],
            "bearer_methods_supported": ["header"],
            "scopes_supported": [DEFAULT_SCOPE],
        }

    def _token_response(
        self,
        access_token: str,
        refresh_token: str,
        expires_in: int,
        scopes: list[str],
    ) -> JSONResponse:
        return JSONResponse(
            {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "Bearer",
                "expires_in": expires_in,
                "scope": " ".join(scopes),
            }
        )

    def _issue_tokens(
        self,
        *,
        client_id: str,
        scopes: list[str],
        subject: str | None,
    ) -> tuple[str, str, int]:
        access_token = ACCESS_TOKEN_PREFIX + secrets.token_urlsafe(48)
        refresh_token = REFRESH_TOKEN_PREFIX + secrets.token_urlsafe(48)
        now = time.time()
        access_digest = hash_token(access_token, self._password_pepper)
        refresh_digest = hash_token(refresh_token, self._password_pepper)
        self.store.put_access(
            StoredAccessToken(
                token_hash=access_digest,
                client_id=client_id,
                scopes=scopes,
                expires_at=now + self._access_token_ttl,
                subject=subject,
            ),
            self._access_token_ttl,
        )
        self.store.put_refresh(
            StoredRefreshToken(
                token_hash=refresh_digest,
                client_id=client_id,
                scopes=scopes,
                expires_at=now + self._refresh_token_ttl,
                subject=subject,
                access_token_hash=access_digest,
            ),
            self._refresh_token_ttl,
        )
        return access_token, refresh_token, self._access_token_ttl

    def _issue_access_token(self, pending: PendingAuthCode) -> tuple[str, int]:
        """Backward-compatible wrapper used by older call sites."""
        access, _refresh, expires_in = self._issue_tokens(
            client_id=pending.client_id,
            scopes=pending.scopes,
            subject=pending.subject,
        )
        return access, expires_in

    def register_routes(self, mcp: fastmcp.FastMCP) -> None:
        from agentmemory.mcp.oauth_google import (
            _google_consent_html,
            create_oauth_session,
            register_google_routes,
        )

        server = self
        if self._use_google:
            register_google_routes(server, mcp)

        @mcp.custom_route("/authorize", methods=["GET", "POST"])
        async def authorize(request: Request) -> Response:
            if request.method == "GET":
                params = {k: v for k, v in request.query_params.items()}
                validated, err = server._validate_authorize_params(params)
                if err or validated is None:
                    return _oauth_error(err or "invalid_request", "Invalid authorize request", 400)
                fields = {
                    "response_type": validated["response_type"],
                    "client_id": validated["client_id"],
                    "redirect_uri": validated["redirect_uri"],
                    "code_challenge": validated["code_challenge"],
                    "code_challenge_method": validated["code_challenge_method"],
                    "state": validated.get("state", ""),
                    "scope": validated.get("scope", DEFAULT_SCOPE),
                }
                if server._use_google:
                    pending = create_oauth_session(server, validated)
                    response = HTMLResponse(
                        _google_consent_html(
                            login_url=f"/oauth/google/login?session={pending.session_id}",
                            error=None,
                        )
                    )
                    _set_oauth_session_cookie(response, pending.session_id)
                    return response
                return HTMLResponse(
                    _consent_html(
                        client_id=validated["client_id"],
                        scope=fields["scope"],
                        allowed_email=server._allowed_email,
                        error=None,
                        fields=fields,
                    )
                )

            form = await request.form()
            form_data = {k: str(v) for k, v in form.items()}
            if server._use_google:
                return _oauth_error(
                    "invalid_request",
                    "Use Sign in with Google on the authorize page",
                    400,
                )
            if server._password_rate_limiter.is_blocked(request):
                return _oauth_error(
                    "access_denied",
                    "Too many failed attempts. Try again later.",
                    429,
                )
            validated, err = server._validate_authorize_params(form_data)
            if err or validated is None:
                return HTMLResponse(
                    _consent_html(
                        client_id=form_data.get("client_id", server._client_id),
                        scope=form_data.get("scope", DEFAULT_SCOPE),
                        allowed_email=server._allowed_email,
                        error="Invalid authorization request.",
                        fields={
                            k: form_data.get(k, "")
                            for k in (
                                "response_type",
                                "client_id",
                                "redirect_uri",
                                "code_challenge",
                                "code_challenge_method",
                                "state",
                                "scope",
                                "email",
                            )
                        },
                    ),
                    status_code=400,
                )

            email = form_data.get("email", "")
            if not server._check_email(email):
                return HTMLResponse(
                    _consent_html(
                        client_id=validated["client_id"],
                        scope=validated.get("scope", DEFAULT_SCOPE),
                        allowed_email=server._allowed_email,
                        error=f"Access denied. Only {server._allowed_email} may authorize.",
                        fields={
                            k: validated.get(k, "")
                            for k in (
                                "response_type",
                                "client_id",
                                "redirect_uri",
                                "code_challenge",
                                "code_challenge_method",
                                "state",
                                "scope",
                            )
                        }
                        | {"email": email},
                    ),
                    status_code=403,
                )

            password = form_data.get("password", "")
            if not server._check_password(password):
                server._password_rate_limiter.record_failure(request)
                return HTMLResponse(
                    _consent_html(
                        client_id=validated["client_id"],
                        scope=validated.get("scope", DEFAULT_SCOPE),
                        allowed_email=server._allowed_email,
                        error="Incorrect password.",
                        fields={
                            k: validated.get(k, "")
                            for k in (
                                "response_type",
                                "client_id",
                                "redirect_uri",
                                "code_challenge",
                                "code_challenge_method",
                                "state",
                                "scope",
                            )
                        }
                        | {"email": email},
                    ),
                    status_code=401,
                )

            server._password_rate_limiter.reset(request)
            server.store.purge_expired()
            code = secrets.token_urlsafe(32)
            scope_raw = validated.get("scope", DEFAULT_SCOPE)
            scopes, _ = _normalize_scopes(scope_raw)
            assert scopes is not None

            server.store.codes[code] = PendingAuthCode(
                code=code,
                client_id=validated["client_id"],
                redirect_uri=validated["redirect_uri"],
                code_challenge=validated["code_challenge"],
                code_challenge_method=validated["code_challenge_method"],
                scopes=scopes,
                state=validated.get("state") or None,
                expires_at=time.time() + server._code_ttl,
                subject=email.strip().lower(),
            )

            query: dict[str, str] = {"code": code}
            if validated.get("state"):
                query["state"] = validated["state"]
            sep = "&" if "?" in validated["redirect_uri"] else "?"
            location = f"{validated['redirect_uri']}{sep}{urlencode(query)}"
            return RedirectResponse(location, status_code=302)

        @mcp.custom_route("/token", methods=["POST"])
        async def token_endpoint(request: Request) -> Response:
            content_type = request.headers.get("content-type", "")
            if "application/x-www-form-urlencoded" in content_type:
                form = await request.form()
                data = {k: str(v) for k, v in form.items()}
            else:
                try:
                    body = await request.json()
                    data = {k: str(v) for k, v in body.items()} if isinstance(body, dict) else {}
                except Exception:
                    data = {}

            grant_type = data.get("grant_type", "")
            client_id = data.get("client_id", "")
            if client_id and client_id != server._client_id:
                return _oauth_error("invalid_client", "Unknown client_id", 401)

            if grant_type == "refresh_token":
                raw_refresh = data.get("refresh_token", "")
                if not raw_refresh.startswith(REFRESH_TOKEN_PREFIX):
                    return _oauth_error("invalid_grant", "Invalid refresh token", 400)
                digest = hash_token(raw_refresh, server._password_pepper)
                stored = server.store.get_refresh(digest)
                if stored is None:
                    return _oauth_error("invalid_grant", "Invalid or expired refresh token", 400)
                server.store.delete_refresh(digest)
                if stored.access_token_hash:
                    server.store.delete_access(stored.access_token_hash)
                access_token, refresh_token, expires_in = server._issue_tokens(
                    client_id=stored.client_id,
                    scopes=stored.scopes,
                    subject=stored.subject,
                )
                return server._token_response(
                    access_token, refresh_token, expires_in, stored.scopes
                )

            if grant_type != "authorization_code":
                return _oauth_error(
                    "unsupported_grant_type",
                    "Only authorization_code and refresh_token are supported",
                )

            code = data.get("code", "")
            redirect_uri = data.get("redirect_uri", "")
            code_verifier = data.get("code_verifier", "")

            server.store.purge_expired()
            pending = server.store.codes.pop(code, None)
            if pending is None or pending.expires_at <= time.time():
                return _oauth_error("invalid_grant", "Invalid or expired authorization code", 400)

            if redirect_uri != pending.redirect_uri:
                return _oauth_error("invalid_grant", "redirect_uri mismatch", 400)

            if not code_verifier or not _verify_pkce(
                code_verifier,
                pending.code_challenge,
                pending.code_challenge_method,
            ):
                return _oauth_error("invalid_grant", "PKCE verification failed", 400)

            access_token, refresh_token, expires_in = server._issue_tokens(
                client_id=pending.client_id,
                scopes=pending.scopes,
                subject=pending.subject,
            )
            return server._token_response(
                access_token, refresh_token, expires_in, pending.scopes
            )

        @mcp.custom_route("/.well-known/oauth-authorization-server", methods=["GET"])
        async def oauth_as_metadata(request: Request) -> Response:
            return JSONResponse(server._as_metadata(request))

        @mcp.custom_route("/.well-known/oauth-protected-resource", methods=["GET"])
        async def oauth_resource_metadata(request: Request) -> Response:
            return JSONResponse(server._resource_metadata(request))

        @mcp.custom_route("/.well-known/oauth-protected-resource/mcp", methods=["GET"])
        async def oauth_resource_metadata_mcp(request: Request) -> Response:
            return JSONResponse(server._resource_metadata(request))

        @mcp.custom_route("/register", methods=["POST"])
        async def register_client(request: Request) -> Response:
            try:
                body = await request.json()
            except Exception:
                body = {}
            if not isinstance(body, dict):
                return _oauth_error("invalid_client_metadata", "JSON object required")
            raw_uris = body.get("redirect_uris", [])
            if isinstance(raw_uris, str):
                raw_uris = [raw_uris]
            if not isinstance(raw_uris, list) or not raw_uris:
                return _oauth_error("invalid_redirect_uri", "redirect_uris required")
            uris = [str(u) for u in raw_uris]
            for uri in uris:
                if not _redirect_allowed(uri, server._redirect_allowlist):
                    return _oauth_error("invalid_redirect_uri", "redirect_uri not allowed")
            return JSONResponse(
                {
                    "client_id": server._client_id,
                    "client_id_issued_at": int(time.time()),
                    "token_endpoint_auth_method": "none",
                    "grant_types": ["authorization_code", "refresh_token"],
                    "response_types": ["code"],
                    "redirect_uris": uris,
                    "scope": DEFAULT_SCOPE,
                },
                status_code=201,
            )

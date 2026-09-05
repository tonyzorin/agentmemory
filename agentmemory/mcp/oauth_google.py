"""Google Sign-In for OAuth consent (single allowed Google account)."""

from __future__ import annotations

import html
import secrets
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlencode

import httpx
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse, Response

if TYPE_CHECKING:
    from agentmemory.mcp.oauth import OAuthAuthorizationServer

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"
GOOGLE_SCOPES = "openid email profile"
SESSION_TTL_SECONDS = 600


@dataclass
class PendingOAuthSession:
    """PKCE authorize params while user completes Google login."""

    session_id: str
    client_id: str
    redirect_uri: str
    code_challenge: str
    code_challenge_method: str
    scopes: list[str]
    client_state: str | None
    expires_at: float


def _google_consent_html(
    *,
    login_url: str,
    error: str | None,
) -> str:
    err_block = (
        f'<p style="color:#b91c1c">{html.escape(error)}</p>' if error else ""
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Authorize agentmemory</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 28rem; margin: 3rem auto; padding: 0 1rem; text-align: center; }}
    .meta {{ color: #555; font-size: 0.9rem; margin: 1rem 0 1.5rem; line-height: 1.45; }}
    .google-btn {{
      display: inline-flex; align-items: center; gap: 0.75rem;
      padding: 0.65rem 1.25rem; border: 1px solid #dadce0; border-radius: 999px;
      background: #fff; color: #3c4043; font-size: 0.95rem; font-weight: 500;
      text-decoration: none; cursor: pointer;
    }}
    .google-btn:hover {{ background: #f8f9fa; }}
    .google-icon {{ width: 18px; height: 18px; }}
  </style>
</head>
<body>
  <h1>Authorize access</h1>
  <p class="meta">This app wants to connect to your agentmemory so it can store and recall memories on your behalf.</p>
  {err_block}
  <a class="google-btn" href="{html.escape(login_url)}">
    <svg class="google-icon" viewBox="0 0 48 48" aria-hidden="true">
      <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/>
      <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6.01c4.51-4.18 7.09-10.36 7.09-17.66z"/>
      <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"/>
      <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6.01c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/>
    </svg>
    Sign in with Google
  </a>
</body>
</html>"""


def _drop_session(server: OAuthAuthorizationServer, session_id: str) -> None:
    server.store.oauth_sessions.pop(session_id, None)


def register_google_routes(server: OAuthAuthorizationServer, mcp) -> None:
    """Register /oauth/google/login and /oauth/google/callback."""
    from agentmemory.mcp.oauth import (
        _clear_oauth_session_cookie,
        _session_cookie_matches,
    )

    @mcp.custom_route("/oauth/google/login", methods=["GET"])
    async def google_login(request: Request) -> Response:
        session_id = request.query_params.get("session", "")
        if not _session_cookie_matches(request, session_id):
            return HTMLResponse("Invalid OAuth session.", status_code=403)

        pending = server.store.oauth_sessions.get(session_id)
        server.store.purge_expired()
        if not pending or pending.expires_at <= time.time():
            return HTMLResponse("Session expired. Restart authorization.", status_code=400)

        params = {
            "client_id": server._google_client_id,
            "redirect_uri": server._google_callback_url,
            "response_type": "code",
            "scope": GOOGLE_SCOPES,
            "state": session_id,
            "access_type": "online",
            "prompt": "select_account",
        }
        return RedirectResponse(f"{GOOGLE_AUTH_URL}?{urlencode(params)}", status_code=302)

    @mcp.custom_route("/oauth/google/callback", methods=["GET"])
    async def google_callback(request: Request) -> Response:
        error = request.query_params.get("error")
        if error:
            desc = request.query_params.get("error_description", error)
            return HTMLResponse(f"Google sign-in failed: {html.escape(desc)}", status_code=400)

        session_id = request.query_params.get("state", "")
        if not _session_cookie_matches(request, session_id):
            _drop_session(server, session_id)
            return HTMLResponse("Invalid OAuth session.", status_code=403)

        google_code = request.query_params.get("code", "")
        pending = server.store.oauth_sessions.get(session_id)
        server.store.purge_expired()
        if not pending or pending.expires_at <= time.time():
            return HTMLResponse("Session expired. Restart authorization.", status_code=400)
        if not google_code:
            return HTMLResponse("Missing Google authorization code.", status_code=400)

        async with httpx.AsyncClient(timeout=15.0) as client:
            token_resp = await client.post(
                GOOGLE_TOKEN_URL,
                data={
                    "code": google_code,
                    "client_id": server._google_client_id,
                    "client_secret": server._google_client_secret,
                    "redirect_uri": server._google_callback_url,
                    "grant_type": "authorization_code",
                },
                headers={"Accept": "application/json"},
            )
            if token_resp.status_code != 200:
                return HTMLResponse("Google token exchange failed.", status_code=400)
            access_token = token_resp.json().get("access_token")
            if not access_token:
                return HTMLResponse("Google token response missing access_token.", status_code=400)

            user_resp = await client.get(
                GOOGLE_USERINFO_URL,
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if user_resp.status_code != 200:
                return HTMLResponse("Failed to fetch Google profile.", status_code=400)
            userinfo = user_resp.json()
            email = (userinfo.get("email") or "").strip().lower()
            email_verified = userinfo.get("email_verified")

        if not email_verified:
            return HTMLResponse(
                _google_consent_html(
                    login_url=f"/oauth/google/login?session={session_id}",
                    error="Google account email is not verified.",
                ),
                status_code=403,
            )

        if not server._check_email(email):
            return HTMLResponse(
                _google_consent_html(
                    login_url=f"/oauth/google/login?session={session_id}",
                    error=f"Access denied. Sign in with {server._allowed_email}.",
                ),
                status_code=403,
            )

        server.store.oauth_sessions.pop(session_id, None)
        response = server._redirect_with_auth_code(
            redirect_uri=pending.redirect_uri,
            code_challenge=pending.code_challenge,
            code_challenge_method=pending.code_challenge_method,
            scopes=pending.scopes,
            client_state=pending.client_state,
            subject=email,
        )
        _clear_oauth_session_cookie(response)
        return response


def create_oauth_session(
    server: OAuthAuthorizationServer,
    validated: dict[str, str],
) -> PendingOAuthSession:
    from agentmemory.mcp.oauth import DEFAULT_SCOPE, _normalize_scopes

    scope_raw = validated.get("scope", DEFAULT_SCOPE)
    scopes, _ = _normalize_scopes(scope_raw)
    assert scopes is not None
    session_id = secrets.token_urlsafe(32)
    pending = PendingOAuthSession(
        session_id=session_id,
        client_id=validated["client_id"],
        redirect_uri=validated["redirect_uri"],
        code_challenge=validated["code_challenge"],
        code_challenge_method=validated["code_challenge_method"],
        scopes=scopes,
        client_state=validated.get("state") or None,
        expires_at=time.time() + SESSION_TTL_SECONDS,
    )
    server.store.oauth_sessions[session_id] = pending
    return pending

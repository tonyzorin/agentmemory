"""Shared token hashing helpers for Bearer and OAuth passwords."""

from __future__ import annotations

import hashlib
import hmac
import secrets


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

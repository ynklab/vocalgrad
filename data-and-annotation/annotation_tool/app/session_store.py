from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from .config import SESSION_MAX_AGE_SECONDS, SESSION_SIGNING_SECRET, SESSION_TOKEN_BYTES


@dataclass(frozen=True)
class SessionRecord:
    token: str
    user_id: str
    role: str
    created_at: str


_REVOKED_TOKENS: set[str] = set()


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode((data + padding).encode("ascii"))


def _sign(payload_b64: str) -> str:
    digest = hmac.new(
        SESSION_SIGNING_SECRET.encode("utf-8"),
        payload_b64.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return _b64encode(digest)


def _build_signed_token(*, user_id: str, role: str, created_at: str) -> str:
    payload = {
        "user_id": user_id,
        "role": role,
        "created_at": created_at,
        "iat": int(time.time()),
        "nonce": secrets.token_urlsafe(SESSION_TOKEN_BYTES),
    }
    payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    payload_b64 = _b64encode(payload_json.encode("utf-8"))
    signature_b64 = _sign(payload_b64)
    return f"{payload_b64}.{signature_b64}"


def _verify_signed_token(token: str) -> SessionRecord | None:
    try:
        payload_b64, signature_b64 = token.split(".", 1)
    except ValueError:
        return None

    expected_sig = _sign(payload_b64)
    if not hmac.compare_digest(signature_b64, expected_sig):
        return None

    try:
        payload = json.loads(_b64decode(payload_b64).decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return None

    issued_at = int(payload.get("iat", 0))
    if issued_at <= 0:
        return None
    if (int(time.time()) - issued_at) > SESSION_MAX_AGE_SECONDS:
        return None

    user_id = str(payload.get("user_id", "")).strip()
    role = str(payload.get("role", "")).strip()
    created_at = str(payload.get("created_at", "")).strip()
    if not user_id or not role or not created_at:
        return None

    return SessionRecord(
        token=token,
        user_id=user_id,
        role=role,
        created_at=created_at,
    )


def create_session(*, user_id: str, role: str) -> SessionRecord:
    created_at = datetime.now(timezone.utc).isoformat()
    token = _build_signed_token(user_id=user_id, role=role, created_at=created_at)
    return SessionRecord(
        token=token,
        user_id=user_id,
        role=role,
        created_at=created_at,
    )


def get_session(token: str) -> SessionRecord | None:
    if token in _REVOKED_TOKENS:
        return None
    return _verify_signed_token(token)


def clear_session(token: str) -> None:
    _REVOKED_TOKENS.add(token)

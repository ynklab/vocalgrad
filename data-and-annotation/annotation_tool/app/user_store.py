from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass
from pathlib import Path

from .config import PASSWORD_HASH_ITERATIONS, USERS_PATH

VALID_ROLES = {"annotator", "test", "admin"}


@dataclass(frozen=True)
class UserRecord:
    user_id: str
    role: str
    password_hash: str
    assigned_categories: tuple[str, ...]
    category_order: tuple[str, ...]


def hash_password(password: str, *, iterations: int = PASSWORD_HASH_ITERATIONS) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return (
        f"pbkdf2_sha256${iterations}$"
        f"{base64.urlsafe_b64encode(salt).decode('ascii')}$"
        f"{base64.urlsafe_b64encode(digest).decode('ascii')}"
    )


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations_text, salt_text, digest_text = encoded.split("$", 3)
    except ValueError:
        return False
    if algorithm != "pbkdf2_sha256":
        return False

    iterations = int(iterations_text)
    salt = base64.urlsafe_b64decode(salt_text.encode("ascii"))
    expected = base64.urlsafe_b64decode(digest_text.encode("ascii"))
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(actual, expected)


def _read_users_path(path: Path = USERS_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Users file not found: {path}. Run scripts/init_annotation_users.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_users(path: Path = USERS_PATH) -> dict[str, UserRecord]:
    payload = _read_users_path(path)
    users = payload.get("users", [])
    if not isinstance(users, list) or not users:
        raise ValueError(f"Users file has no users: {path}")

    out: dict[str, UserRecord] = {}
    for entry in users:
        user_id = str(entry["user_id"]).strip()
        role = str(entry["role"]).strip()
        password_hash = str(entry["password_hash"]).strip()
        assigned_categories = tuple(str(v).strip() for v in entry.get("assigned_categories", []))
        category_order = tuple(
            str(v).strip()
            for v in entry.get("category_order", assigned_categories)
        )

        if not user_id:
            raise ValueError(f"Users file contains empty user_id: {path}")
        if role not in VALID_ROLES:
            raise ValueError(f"Invalid role {role!r} for user {user_id!r}")
        if role in {"annotator", "test"} and len(category_order) != 5:
            raise ValueError(f"User {user_id!r} must have exactly 5 categories.")

        out[user_id] = UserRecord(
            user_id=user_id,
            role=role,
            password_hash=password_hash,
            assigned_categories=assigned_categories,
            category_order=category_order,
        )
    return out


def authenticate_user(user_id: str, password: str) -> UserRecord | None:
    users = load_users()
    user = users.get(user_id.strip())
    if user is None:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user

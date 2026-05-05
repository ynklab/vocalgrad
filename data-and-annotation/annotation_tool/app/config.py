from __future__ import annotations

import os
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent
TOOL_ROOT = APP_ROOT.parent
REPO_ROOT = TOOL_ROOT.parent

SUPPORTED_CATEGORIES = (
    "volume",
    "speaking_speed",
    "voice_pitch",
    "background_noise",
    "audio_distortion",
    "audio_roughness",
    "voice_clarity",
    "voice_vibration",
    "echo",
)

SHARED_MANIFEST_FILENAME = "shared_annotation_50.csv"
EXPECTED_NUM_ITEMS = 50
MANIFESTS_ROOT = TOOL_ROOT / "data" / "manifests"
RESULTS_ROOT = TOOL_ROOT / "data" / "results"
USERS_ROOT = TOOL_ROOT / "data" / "users"
USERS_PATH = USERS_ROOT / "users.json"
CREDENTIALS_PATH = USERS_ROOT / "credentials.csv"

# Restrict audio serving to processed artifacts only.
AUDIO_BASE_DIRS = [
    REPO_ROOT / "data" / "processed",
]

SESSION_TOKEN_BYTES = 24
SESSION_MAX_AGE_SECONDS = int(
    os.getenv("VOCALGRAD_SESSION_MAX_AGE_SECONDS", str(60 * 60 * 24 * 14))
)
SESSION_SIGNING_SECRET = os.getenv("VOCALGRAD_SESSION_SECRET", "dev-insecure-change-me")
PASSWORD_HASH_ITERATIONS = 240_000


def manifest_path_for_category(category: str) -> Path:
    if category not in SUPPORTED_CATEGORIES:
        raise ValueError(f"Unsupported category: {category}")
    return MANIFESTS_ROOT / category / SHARED_MANIFEST_FILENAME

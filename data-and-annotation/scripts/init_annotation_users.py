from __future__ import annotations

import argparse
import csv
import json
import secrets
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from annotation_tool.app.config import CREDENTIALS_PATH, USERS_PATH, USERS_ROOT  # noqa: E402
from annotation_tool.app.user_store import hash_password  # noqa: E402

CATEGORIES = [
    "volume",
    "speaking_speed",
    "voice_pitch",
    "background_noise",
    "audio_distortion",
    "audio_roughness",
    "voice_clarity",
    "voice_vibration",
    "echo",
]

ANNOTATOR_ASSIGNMENTS = {
    "annotator_01": [
        "volume",
        "speaking_speed",
        "voice_pitch",
        "background_noise",
        "audio_distortion",
    ],
    "annotator_02": [
        "audio_roughness",
        "voice_clarity",
        "voice_vibration",
        "echo",
        "volume",
    ],
    "annotator_03": [
        "speaking_speed",
        "voice_pitch",
        "background_noise",
        "audio_distortion",
        "audio_roughness",
    ],
    "annotator_04": [
        "voice_clarity",
        "voice_vibration",
        "echo",
    ],
    "annotator_05": [
        "speaking_speed",
        "background_noise",
        "audio_distortion",
        "voice_vibration",
        "echo",
    ],
    "annotator_06": [
        "volume",
        "voice_pitch",
        "audio_roughness",
        "voice_clarity",
    ],
}


def random_password(length: int = 14) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789"
    return "".join(secrets.choice(alphabet) for _ in range(length))


def build_records() -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    test_categories = sorted(secrets.SystemRandom().sample(CATEGORIES, 5))

    credential_rows: list[dict[str, str]] = []
    user_rows: list[dict[str, object]] = []

    for user_id, categories in ANNOTATOR_ASSIGNMENTS.items():
        password = random_password()
        credential_rows.append(
            {
                "user_id": user_id,
                "role": "annotator",
                "password": password,
                "assigned_categories": ",".join(categories),
            }
        )
        user_rows.append(
            {
                "user_id": user_id,
                "role": "annotator",
                "password_hash": hash_password(password),
                "assigned_categories": categories,
                "category_order": categories,
            }
        )

    test_password = random_password()
    credential_rows.append(
        {
            "user_id": "test_user",
            "role": "test",
            "password": test_password,
            "assigned_categories": ",".join(test_categories),
        }
    )
    user_rows.append(
        {
            "user_id": "test_user",
            "role": "test",
            "password_hash": hash_password(test_password),
            "assigned_categories": test_categories,
            "category_order": test_categories,
        }
    )

    admin_password = random_password()
    credential_rows.append(
        {
            "user_id": "admin",
            "role": "admin",
            "password": admin_password,
            "assigned_categories": "",
        }
    )
    user_rows.append(
        {
            "user_id": "admin",
            "role": "admin",
            "password_hash": hash_password(admin_password),
            "assigned_categories": [],
            "category_order": [],
        }
    )

    return user_rows, credential_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize local users for the annotation tool.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing users.json and credentials.csv if they already exist.",
    )
    args = parser.parse_args()

    if (USERS_PATH.exists() or CREDENTIALS_PATH.exists()) and not args.overwrite:
        raise SystemExit(
            "Users already initialized. Use --overwrite to regenerate users and passwords."
        )

    USERS_ROOT.mkdir(parents=True, exist_ok=True)
    user_rows, credential_rows = build_records()

    USERS_PATH.write_text(json.dumps({"users": user_rows}, indent=2), encoding="utf-8")

    with CREDENTIALS_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["user_id", "role", "password", "assigned_categories"],
        )
        writer.writeheader()
        writer.writerows(credential_rows)

    print(f"Wrote {USERS_PATH}")
    print(f"Wrote {CREDENTIALS_PATH}")


if __name__ == "__main__":
    main()

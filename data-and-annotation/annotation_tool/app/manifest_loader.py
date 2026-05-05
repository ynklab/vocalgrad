from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .config import (
    AUDIO_BASE_DIRS,
    EXPECTED_NUM_ITEMS,
    REPO_ROOT,
    SUPPORTED_CATEGORIES,
    manifest_path_for_category,
)

REQUIRED_COLUMNS = [
    "item_id",
    "clip_id",
    "audio_path",
    "category",
    "attribute",
    "difficulty",
    "trajectory",
    "label",
]
VALID_LABELS = {"increase", "decrease"}


@dataclass(frozen=True)
class ManifestItem:
    item_id: str
    clip_id: str
    audio_path: str
    category: str
    attribute: str
    difficulty: str
    trajectory: str
    label: str

    def audio_file(self) -> Path:
        return (REPO_ROOT / self.audio_path).resolve()

    def audio_url(self) -> str:
        return f"/audio/{self.audio_path}"


def _validate_audio_path(audio_path: str) -> None:
    resolved = (REPO_ROOT / audio_path).resolve()
    if not resolved.is_file():
        raise ValueError(f"Audio path does not exist: {audio_path}")

    if not any(
        base.resolve() in resolved.parents or resolved == base.resolve()
        for base in AUDIO_BASE_DIRS
    ):
        raise ValueError(f"Audio path is outside allowed directories: {audio_path}")


@lru_cache(maxsize=None)
def load_manifest(category: str) -> tuple[ManifestItem, ...]:
    if category not in SUPPORTED_CATEGORIES:
        raise ValueError(f"Unsupported category: {category}")

    path = manifest_path_for_category(category)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"Manifest is empty: {path}")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in rows[0]]
    if missing_columns:
        raise ValueError(f"Manifest missing required columns: {missing_columns}")

    if len(rows) != EXPECTED_NUM_ITEMS:
        raise ValueError(
            f"Manifest must contain exactly {EXPECTED_NUM_ITEMS} rows, found {len(rows)}: {path}"
        )

    items: list[ManifestItem] = []
    seen_item_ids: set[str] = set()
    for row in rows:
        item_id = row["item_id"].strip()
        if not item_id:
            raise ValueError(f"Manifest contains empty item_id: {path}")
        if item_id in seen_item_ids:
            raise ValueError(f"Duplicate item_id in manifest {path}: {item_id}")
        seen_item_ids.add(item_id)

        label = row["label"].strip().lower()
        if label not in VALID_LABELS:
            raise ValueError(f"Invalid label for item_id={item_id}: {row['label']}")

        audio_path = row["audio_path"].strip()
        _validate_audio_path(audio_path)

        items.append(
            ManifestItem(
                item_id=item_id,
                clip_id=row["clip_id"].strip(),
                audio_path=audio_path,
                category=row["category"].strip(),
                attribute=row["attribute"].strip(),
                difficulty=row["difficulty"].strip(),
                trajectory=row["trajectory"].strip(),
                label=label,
            )
        )

    return tuple(items)


def manifest_by_item_id(category: str) -> dict[str, ManifestItem]:
    return {item.item_id: item for item in load_manifest(category)}


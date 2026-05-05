from __future__ import annotations

import csv
import hashlib
import random
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"
OUTPUT_ROOT = REPO_ROOT / "annotation_tool" / "data" / "manifests"
NUM_SHARED_ITEMS = 50
MANIFEST_FILENAME = f"shared_annotation_{NUM_SHARED_ITEMS}.csv"

CATEGORY_SPECS = {
    "volume": {
        "manifest_path": PROCESSED_ROOT / "test" / "volume" / "metadata" / "volume_manifest.csv",
        "attribute": "volume",
    },
    "speaking_speed": {
        "manifest_path": (
            PROCESSED_ROOT
            / "test"
            / "speaking_speed"
            / "metadata"
            / "speaking_speed_manifest.csv"
        ),
        "attribute": "speaking speed",
    },
    "voice_pitch": {
        "manifest_path": (
            PROCESSED_ROOT
            / "test"
            / "voice_pitch"
            / "metadata"
            / "voice_pitch_manifest.csv"
        ),
        "attribute": "voice pitch",
    },
    "background_noise": {
        "manifest_path": (
            PROCESSED_ROOT
            / "test"
            / "background_noise"
            / "metadata"
            / "background_noise_manifest.csv"
        ),
        "attribute": "background noise",
    },
    "audio_distortion": {
        "manifest_path": (
            PROCESSED_ROOT
            / "test"
            / "audio_distortion"
            / "metadata"
            / "audio_distortion_manifest.csv"
        ),
        "attribute": "audio distortion",
    },
    "audio_roughness": {
        "manifest_path": (
            PROCESSED_ROOT
            / "test"
            / "audio_roughness"
            / "metadata"
            / "audio_roughness_manifest.csv"
        ),
        "attribute": "audio roughness",
    },
    "voice_clarity": {
        "manifest_path": (
            PROCESSED_ROOT
            / "test"
            / "voice_clarity"
            / "metadata"
            / "voice_clarity_manifest.csv"
        ),
        "attribute": "voice clarity",
    },
    "voice_vibration": {
        "manifest_path": (
            PROCESSED_ROOT
            / "test"
            / "voice_vibration"
            / "metadata"
            / "voice_vibration_manifest.csv"
        ),
        "attribute": "voice vibration",
    },
    "echo": {
        "manifest_path": PROCESSED_ROOT / "test" / "echo" / "metadata" / "echo_manifest.csv",
        "attribute": "echo",
    },
}

TIER_TRAJECTORY_ORDER = [
    ("low", "linear"),
    ("mid", "quad_first_flat"),
    ("high", "quad_last_flat"),
    ("low", "jump"),
    ("mid", "linear"),
    ("high", "quad_first_flat"),
    ("low", "quad_last_flat"),
    ("mid", "jump"),
    ("high", "linear"),
    ("low", "quad_first_flat"),
    ("mid", "quad_last_flat"),
    ("high", "jump"),
]

OUTPUT_COLUMNS = [
    "item_id",
    "clip_id",
    "audio_path",
    "category",
    "attribute",
    "difficulty",
    "trajectory",
    "label",
]
MANIFEST_SHUFFLE_SEED = "vocalgrad-annotation-shared-order-v1"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def select_shared_subset(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    source_ids_by_speaker: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        condition = (row["tier"], row["curve"], row["direction"])
        source_test_item_id = row["source_test_item_id"]
        grouped[condition][source_test_item_id] = row
        source_ids_by_speaker[row["speaker_id"]].add(source_test_item_id)

    if len(source_ids_by_speaker) != NUM_SHARED_ITEMS:
        raise ValueError(
            "Expected one selected speaker per shared item. "
            f"Found {len(source_ids_by_speaker)} speakers for target size {NUM_SHARED_ITEMS}."
        )

    ordered_sources = [
        sorted(source_ids_by_speaker[speaker_id])[0]
        for speaker_id in sorted(source_ids_by_speaker)
    ]

    selected: list[dict[str, str]] = []
    for index, source_test_item_id in enumerate(ordered_sources):
        tier, trajectory = TIER_TRAJECTORY_ORDER[index % len(TIER_TRAJECTORY_ORDER)]
        direction = "up" if index % 2 == 0 else "down"
        condition = (tier, trajectory, direction)
        row = grouped[condition].get(source_test_item_id)
        if row is None:
            raise ValueError(
                "Missing condition "
                f"{condition} for source_test_item_id={source_test_item_id}."
            )
        selected.append(row)

    return selected


def convert_rows(
    rows: list[dict[str, str]],
    *,
    category: str,
    attribute: str,
) -> list[dict[str, str]]:
    converted: list[dict[str, str]] = []
    for row in rows:
        converted.append(
            {
                "item_id": row["benchmark_clip_id"],
                "clip_id": row["source_test_item_id"],
                "audio_path": row["output_audio_path"],
                "category": category,
                "attribute": attribute,
                "difficulty": row["tier"],
                "trajectory": row["curve"],
                "label": annotation_label_for_row(category=category, row=row),
            }
        )
    return converted


def annotation_label_for_row(category: str, row: dict[str, str]) -> str:
    direction = row["direction"]
    return "increase" if direction == "up" else "decrease"


def write_manifest(category: str, rows: list[dict[str, str]]) -> Path:
    out_dir = OUTPUT_ROOT / category
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / MANIFEST_FILENAME
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def shuffled_rows(
    rows: list[dict[str, str]],
    *,
    category: str,
) -> list[dict[str, str]]:
    seed_material = f"{MANIFEST_SHUFFLE_SEED}:{category}".encode("utf-8")
    seed = int(hashlib.sha256(seed_material).hexdigest()[:16], 16)
    rng = random.Random(seed)
    out = list(rows)
    rng.shuffle(out)
    return out


def main() -> None:
    for category, spec in CATEGORY_SPECS.items():
        raw_rows = load_rows(spec["manifest_path"])
        selected_rows = select_shared_subset(raw_rows)
        output_rows = convert_rows(
            selected_rows,
            category=category,
            attribute=spec["attribute"],
        )
        output_rows = shuffled_rows(output_rows, category=category)
        out_path = write_manifest(category, output_rows)
        print(f"Wrote {len(output_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()

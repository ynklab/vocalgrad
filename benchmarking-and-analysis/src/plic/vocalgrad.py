from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


DEFAULT_VOCALGRAD_DATASET_ROOT = Path("datasets/vocalgrad/test")
DEFAULT_VOCALGRAD_SOURCE_ROOT = DEFAULT_VOCALGRAD_DATASET_ROOT / "source_clips"


@dataclass
class VocalGradSample:
    category: str
    benchmark_clip_id: str
    audio_path: Path
    gold_label: str
    metadata: dict[str, str]


def _manifest_path(dataset_root: Path, category: str) -> Path:
    return dataset_root / category / "metadata" / f"{category}_manifest.csv"


def _resolve_audio_path(dataset_root: Path, row: dict[str, str]) -> Path:
    category = row["category"]
    direction = row["direction"]
    benchmark_clip_id = row["benchmark_clip_id"]
    return dataset_root / category / "audio" / direction / f"{benchmark_clip_id}.wav"


def list_vocalgrad_categories(dataset_root: Path) -> list[str]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"VocalGrad dataset root not found: {dataset_root}")

    categories: list[str] = []
    for category_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        category = category_dir.name
        manifest_path = _manifest_path(dataset_root, category)
        if manifest_path.exists():
            categories.append(category)
    return categories


def load_local_vocalgrad_samples(
    dataset_root: Path,
    category: str = "volume",
) -> list[VocalGradSample]:
    manifest_path = _manifest_path(dataset_root, category)
    if not manifest_path.exists():
        raise FileNotFoundError(f"VocalGrad manifest not found: {manifest_path}")

    samples: list[VocalGradSample] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            direction = row["direction"].strip().lower()
            if direction not in {"up", "down"}:
                continue

            audio_path = _resolve_audio_path(dataset_root, row)
            if not audio_path.exists():
                raise FileNotFoundError(
                    "Expected VocalGrad audio file not found: "
                    f"{audio_path} (benchmark_clip_id={row['benchmark_clip_id']})"
                )

            samples.append(
                VocalGradSample(
                    category=category,
                    benchmark_clip_id=row["benchmark_clip_id"],
                    audio_path=audio_path,
                    gold_label="increase" if direction == "up" else "decrease",
                    metadata=row,
                )
            )

    return samples

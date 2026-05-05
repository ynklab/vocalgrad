from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

ACTIVE_CATEGORIES = [
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

ABLATION_CATEGORIES = [
    "voice_pitch",
    "volume",
    "background_noise",
    "voice_vibration",
]

EXPECTED_TEST_SOURCES = 100
EXPECTED_TRAIN_SOURCES = 500
EXPECTED_TEST_SPEAKERS = 50
EXPECTED_TRAIN_SPEAKERS = 57
EXPECTED_TEST_ROWS_PER_CATEGORY = 2400
EXPECTED_TRAIN_ROWS_PER_CATEGORY = 12000
EXPECTED_ABLATION_ROWS_PER_CATEGORY = 240
EXPECTED_PUBLIC_ROWS = 21600


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate VocalGrad dataset metadata.")
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/processed"),
        help="Root directory containing generated benchmark outputs.",
    )
    parser.add_argument(
        "--public-root",
        type=Path,
        default=Path("data/vocalgrad_public"),
        help="Root directory containing the packaged public dataset.",
    )
    parser.add_argument(
        "--require-processed",
        action="store_true",
        help="Fail if generated processed manifests are missing.",
    )
    parser.add_argument(
        "--require-public",
        action="store_true",
        help="Fail if the packaged public metadata is missing.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fail(message: str) -> None:
    raise SystemExit(f"[FAIL] {message}")


def check_equal(name: str, actual: object, expected: object) -> None:
    if actual != expected:
        fail(f"{name}: expected {expected!r}, got {actual!r}")
    print(f"[OK] {name}: {actual}")


def check_source_metadata() -> None:
    test_source_path = Path("data/metadata/selection/test_source_clips_100.csv")
    train_source_path = Path("data/metadata/selection/train_source_clips_500.csv")
    train_report_path = Path("data/metadata/selection/train_selection_report.json")

    test_rows = read_csv_rows(test_source_path)
    train_rows = read_csv_rows(train_source_path)

    check_equal("test source rows", len(test_rows), EXPECTED_TEST_SOURCES)
    check_equal("train source rows", len(train_rows), EXPECTED_TRAIN_SOURCES)
    check_equal(
        "test speakers",
        len({row["speaker_id"] for row in test_rows}),
        EXPECTED_TEST_SPEAKERS,
    )
    check_equal(
        "train speakers",
        len({row["speaker_id"] for row in train_rows}),
        EXPECTED_TRAIN_SPEAKERS,
    )

    test_speakers = {row["speaker_id"] for row in test_rows}
    train_speakers = {row["speaker_id"] for row in train_rows}
    check_equal("train/test speaker overlap", len(test_speakers & train_speakers), 0)

    test_texts = {" ".join(row["text"].split()) for row in test_rows}
    train_texts = [" ".join(row["text"].split()) for row in train_rows]
    check_equal("train/test text overlap", sum(text in test_texts for text in train_texts), 0)
    check_equal("duplicate train texts", len(train_texts) - len(set(train_texts)), 0)

    test_missing_onset = sum(
        not row.get("onset_sec") or not row.get("offset_sec") for row in test_rows
    )
    train_missing_onset = sum(
        not row.get("onset_sec") or not row.get("offset_sec") for row in train_rows
    )
    check_equal("test rows missing onset/offset", test_missing_onset, 0)
    check_equal("train rows missing onset/offset", train_missing_onset, 0)

    clip_counts = Counter(row["speaker_id"] for row in train_rows)
    distribution = dict(sorted(Counter(clip_counts.values()).items()))
    check_equal("train per-speaker clip-count distribution", distribution, {8: 13, 9: 44})

    with train_report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)
    check_equal("train report speakers", report["train_speakers"], EXPECTED_TRAIN_SPEAKERS)
    check_equal("train report clips", report["num_train_clips"], EXPECTED_TRAIN_SOURCES)


def check_manifest(path: Path, expected_rows: int, require: bool) -> None:
    if not path.exists():
        if require:
            fail(f"missing manifest: {path}")
        print(f"[SKIP] missing manifest: {path}")
        return

    rows = read_csv_rows(path)
    check_equal(str(path), len(rows), expected_rows)
    directions = Counter(row.get("direction", "") for row in rows)
    if directions:
        expected_per_direction = expected_rows // 2
        check_equal(f"{path} up rows", directions.get("up", 0), expected_per_direction)
        check_equal(f"{path} down rows", directions.get("down", 0), expected_per_direction)


def check_processed_manifests(processed_root: Path, require: bool) -> None:
    for category in ACTIVE_CATEGORIES:
        check_manifest(
            processed_root / "test" / category / "metadata" / f"{category}_manifest.csv",
            EXPECTED_TEST_ROWS_PER_CATEGORY,
            require,
        )
        check_manifest(
            processed_root / "train" / category / "metadata" / f"{category}_manifest.csv",
            EXPECTED_TRAIN_ROWS_PER_CATEGORY,
            require,
        )

    for category in ABLATION_CATEGORIES:
        check_manifest(
            processed_root / "ablation_beep" / category / "metadata" / f"{category}_manifest.csv",
            EXPECTED_ABLATION_ROWS_PER_CATEGORY,
            require,
        )


def check_public_dataset(public_root: Path, require: bool) -> None:
    metadata_path = public_root / "metadata.csv"
    if not metadata_path.exists():
        if require:
            fail(f"missing public metadata: {metadata_path}")
        print(f"[SKIP] missing public metadata: {metadata_path}")
        return

    rows = read_csv_rows(metadata_path)
    check_equal("public metadata rows", len(rows), EXPECTED_PUBLIC_ROWS)
    check_equal(
        "public categories",
        sorted({row["category"] for row in rows}),
        sorted(ACTIVE_CATEGORIES),
    )


def main() -> None:
    args = parse_args()
    check_source_metadata()
    check_processed_manifests(args.processed_root, require=args.require_processed)
    check_public_dataset(args.public_root, require=args.require_public)
    print("[OK] VocalGrad validation completed")


if __name__ == "__main__":
    main()

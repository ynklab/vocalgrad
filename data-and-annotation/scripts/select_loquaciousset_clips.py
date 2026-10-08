#!/usr/bin/env python3
"""Select gender-balanced LoquaciousSet clips from the dev and test splits.

The LoquaciousSet dev split is used as VocalGrad extension train data, while the
LoquaciousSet test split remains test data. Metadata is read from the Hugging
Face Dataset Viewer API, so only the selected audio files are downloaded.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

DATASET = "speechbrain/LoquaciousSet"
CONFIG = "clean"
ROWS_API = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100
REQUEST_INTERVAL_SEC = 0.8
USER_AGENT = "vocalgrad-loquaciousset-selector/1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/selected/loquaciousset"),
    )
    parser.add_argument("--num-train-clips", type=int, default=500)
    parser.add_argument("--num-test-clips", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-duration-sec", type=float, default=0.0)
    parser.add_argument("--max-duration-sec", type=float, default=float("inf"))
    parser.add_argument(
        "--max-clips-per-speaker",
        type=int,
        default=None,
        help="Optional cap that increases speaker diversity within each output split.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Write manifests without downloading the selected WAV files.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def fetch_json(url: str, *, attempts: int = 8) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if attempt + 1 == attempts:
                raise
            if error.code == 429:
                retry_after = error.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else min(15 * (attempt + 1), 60)
            elif 500 <= error.code < 600:
                delay = min(2**attempt, 30)
            else:
                raise
            time.sleep(delay)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt + 1 == attempts:
                raise
            time.sleep(min(2**attempt, 30))
    raise AssertionError("unreachable")


def canonical_gender(value: Any) -> str:
    label = str(value or "").strip().lower()
    if label in {"f", "female", "feminine", "female_feminine"}:
        return "female"
    if label in {"m", "male", "masculine", "male_masculine"}:
        return "male"
    return "unknown"


def fetch_split_metadata(split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    total: int | None = None

    while total is None or offset < total:
        query = urllib.parse.urlencode(
            {
                "dataset": DATASET,
                "config": CONFIG,
                "split": split,
                "offset": offset,
                "length": PAGE_SIZE,
            }
        )
        payload = fetch_json(f"{ROWS_API}?{query}")
        total = int(payload["num_rows_total"])
        for item in payload["rows"]:
            row = item["row"]
            audio = row.get("wav") or []
            if not audio or not audio[0].get("src"):
                continue
            rows.append(
                {
                    "source_row_index": int(item["row_idx"]),
                    "clip_id": str(row["ID"]),
                    "speaker_id": str(row.get("spk_id") or ""),
                    "gender": canonical_gender(row.get("sex")),
                    "duration_sec": float(row["duration"]),
                    "text": str(row["text"]),
                    "audio_url": str(audio[0]["src"]),
                }
            )
        offset += len(payload["rows"])
        if not payload["rows"] and offset < total:
            raise RuntimeError(f"Dataset API returned no {split} rows at offset {offset}.")
        if offset < total:
            time.sleep(REQUEST_INTERVAL_SEC)

    return rows


def balanced_targets(total: int, labels: list[str], rng: random.Random) -> dict[str, int]:
    if not labels:
        raise ValueError("No gender labels are available.")
    base, remainder = divmod(total, len(labels))
    shuffled = list(sorted(labels))
    rng.shuffle(shuffled)
    targets = {label: base for label in labels}
    for label in shuffled[:remainder]:
        targets[label] += 1
    return targets


def select_balanced_clips(
    rows: list[dict[str, Any]],
    *,
    count: int,
    min_duration_sec: float,
    max_duration_sec: float,
    max_clips_per_speaker: int | None,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("Selection count must be positive.")
    if max_clips_per_speaker is not None and max_clips_per_speaker <= 0:
        raise ValueError("--max-clips-per-speaker must be positive.")

    eligible = [
        row
        for row in rows
        if min_duration_sec <= row["duration_sec"] <= max_duration_sec
        and row["gender"] in {"female", "male"}
    ]
    labels = sorted({row["gender"] for row in eligible})
    targets = balanced_targets(count, labels, rng)
    selected: list[dict[str, Any]] = []

    for label in labels:
        candidates = [row for row in eligible if row["gender"] == label]
        rng.shuffle(candidates)
        speaker_counts: Counter[str] = Counter()
        picks: list[dict[str, Any]] = []
        for row in candidates:
            speaker_id = row["speaker_id"]
            if (
                max_clips_per_speaker is not None
                and speaker_counts[speaker_id] >= max_clips_per_speaker
            ):
                continue
            picks.append(row)
            speaker_counts[speaker_id] += 1
            if len(picks) == targets[label]:
                break
        if len(picks) != targets[label]:
            raise ValueError(
                f"Cannot select {targets[label]} {label!r} clips; only {len(picks)} "
                "meet the duration and per-speaker constraints."
            )
        selected.extend(picks)

    rng.shuffle(selected)
    return selected


def safe_filename(clip_id: str) -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in clip_id)
    if not safe:
        raise ValueError(f"Clip ID cannot be converted to a filename: {clip_id!r}")
    return f"{safe}.wav"


def download_file(url: str, destination: Path, *, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                with temporary.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
            temporary.replace(destination)
            return
        except urllib.error.HTTPError as error:
            temporary.unlink(missing_ok=True)
            if attempt == 4 or (error.code != 429 and not 500 <= error.code < 600):
                raise
            retry_after = error.headers.get("Retry-After")
            time.sleep(float(retry_after) if retry_after else min(5 * (attempt + 1), 30))
        except (urllib.error.URLError, TimeoutError):
            temporary.unlink(missing_ok=True)
            if attempt == 4:
                raise
            time.sleep(min(2**attempt, 30))


def write_manifest(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    source_split: str,
    output_split: str,
    audio_dir: Path,
) -> None:
    item_id_field = f"{output_split}_item_id"
    fields = [
        item_id_field,
        "clip_id",
        "speaker_id",
        "gender",
        "duration_sec",
        "text",
        "source_dataset",
        "source_config",
        "source_split",
        "source_row_index",
        "split",
        "audio_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    item_id_field: f"loquaciousset_{output_split}_{index:04d}",
                    "clip_id": row["clip_id"],
                    "speaker_id": row["speaker_id"],
                    "gender": row["gender"],
                    "duration_sec": f"{row['duration_sec']:.6f}",
                    "text": row["text"],
                    "source_dataset": DATASET,
                    "source_config": CONFIG,
                    "source_split": source_split,
                    "source_row_index": row["source_row_index"],
                    "split": output_split,
                    "audio_path": str(audio_dir / safe_filename(row["clip_id"])),
                }
            )


def print_summary(name: str, rows: list[dict[str, Any]]) -> None:
    genders = dict(sorted(Counter(row["gender"] for row in rows).items()))
    speakers = {row["speaker_id"] for row in rows}
    durations = [row["duration_sec"] for row in rows]
    print(
        f"{name}: clips={len(rows)}, speakers={len(speakers)}, "
        f"gender={genders}, duration_sec={sum(durations):.1f}"
    )


def main() -> None:
    args = parse_args()
    if args.min_duration_sec < 0 or args.max_duration_sec < args.min_duration_sec:
        raise ValueError("Invalid duration range.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    split_specs = [
        ("test", "test", args.num_test_clips),
    ]

    for split_index, (source_split, output_split, count) in enumerate(split_specs):
        rows = fetch_split_metadata(source_split)
        rng = random.Random(args.seed + 1)  # preserve the original test seed offset
        selected = select_balanced_clips(
            rows,
            count=count,
            min_duration_sec=args.min_duration_sec,
            max_duration_sec=args.max_duration_sec,
            max_clips_per_speaker=args.max_clips_per_speaker,
            rng=rng,
        )
        audio_dir = output_dir / output_split / "audio"
        if not args.metadata_only:
            for index, row in enumerate(selected, start=1):
                destination = audio_dir / safe_filename(row["clip_id"])
                download_file(row["audio_url"], destination, overwrite=args.overwrite)
                if index % 25 == 0 or index == len(selected):
                    print(f"{output_split}: downloaded {index}/{len(selected)}")

        manifest_path = output_dir / f"{output_split}_source_clips_{count}.csv"
        write_manifest(
            manifest_path,
            selected,
            source_split=source_split,
            output_split=output_split,
            audio_dir=audio_dir,
        )
        print_summary(output_split, selected)
        print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()

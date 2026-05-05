#!/usr/bin/env python3
"""Select VocalGrad train source clips from non-test VCTK speakers.

Protocol:
- collect speakers that are not used for the test split
- keep only clips with duration >= threshold
- exclude clips whose transcript text overlaps with the test-source texts
- sample 500 train source clips as evenly as possible across train speakers
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

from select_vocalgrad_splits import (
    count_by,
    discover_sentences_per_speaker,
    load_speaker_info,
    read_transcript_text,
    read_wav_duration_sec,
    write_speaker_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vctk-root", type=Path, default=Path("data/original/VCTK-Corpus"))
    parser.add_argument("--selection-dir", type=Path, default=Path("data/metadata/selection"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-train-clips", type=int, default=500)
    parser.add_argument("--min-clip-duration-sec", type=float, default=3.0)
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def build_train_speaker_ids(
    all_speaker_ids: Iterable[str],
    test_speaker_ids: set[str],
) -> list[str]:
    return sorted(
        [speaker_id for speaker_id in all_speaker_ids if speaker_id not in test_speaker_ids]
    )


def per_speaker_targets(
    speaker_ids: list[str],
    total_clips: int,
    rng: random.Random,
) -> dict[str, int]:
    if not speaker_ids:
        raise ValueError("No train speakers available.")

    base = total_clips // len(speaker_ids)
    remainder = total_clips % len(speaker_ids)
    shuffled = list(speaker_ids)
    rng.shuffle(shuffled)

    targets = {speaker_id: base for speaker_id in speaker_ids}
    for speaker_id in shuffled[:remainder]:
        targets[speaker_id] += 1
    return targets


def collect_candidate_clips(
    *,
    speaker_ids: list[str],
    sentences_per_speaker: dict[str, set[str]],
    vctk_root: Path,
    min_clip_duration_sec: float,
    excluded_texts: set[str],
) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}

    for speaker_id in speaker_ids:
        candidates: list[dict[str, str]] = []
        for sentence_id in sorted(sentences_per_speaker.get(speaker_id, set())):
            clip_id = f"{speaker_id}_{sentence_id}"
            wav_path = vctk_root / "wav48" / speaker_id / f"{clip_id}.wav"
            txt_path = vctk_root / "txt" / speaker_id / f"{clip_id}.txt"
            if not wav_path.exists() or not txt_path.exists():
                continue

            text = read_transcript_text(txt_path)
            if normalize_text(text) in excluded_texts:
                continue

            duration_sec = read_wav_duration_sec(wav_path)
            if duration_sec < min_clip_duration_sec:
                continue

            candidates.append(
                {
                    "speaker_id": speaker_id,
                    "sentence_id": sentence_id,
                    "clip_id": clip_id,
                    "audio_duration_sec": f"{duration_sec:.6f}",
                    "audio_path": str(wav_path),
                    "text_path": str(txt_path),
                    "text": text,
                }
            )

        out[speaker_id] = candidates

    return out


def select_train_clips(
    *,
    speaker_ids: list[str],
    targets: dict[str, int],
    candidate_map: dict[str, list[dict[str, str]]],
    excluded_texts: set[str],
    rng: random.Random,
) -> list[dict[str, str]]:
    used_texts = set(excluded_texts)
    selected_by_speaker: dict[str, list[dict[str, str]]] = {
        speaker_id: [] for speaker_id in speaker_ids
    }

    for speaker_id in speaker_ids:
        candidates = list(candidate_map[speaker_id])
        rng.shuffle(candidates)
        candidate_map[speaker_id] = candidates
        unique_text_count = len({normalize_text(item["text"]) for item in candidates})
        if unique_text_count < targets[speaker_id]:
            raise ValueError(
                f"Speaker {speaker_id} has only {unique_text_count} eligible unique-text clips, "
                f"but target is {targets[speaker_id]}."
            )

    progress = True
    while progress:
        progress = False
        for speaker_id in speaker_ids:
            if len(selected_by_speaker[speaker_id]) >= targets[speaker_id]:
                continue

            candidates = candidate_map[speaker_id]
            pick_index = None
            for idx, item in enumerate(candidates):
                norm_text = normalize_text(item["text"])
                if norm_text in used_texts:
                    continue
                pick_index = idx
                break

            if pick_index is None:
                continue

            item = candidates.pop(pick_index)
            used_texts.add(normalize_text(item["text"]))
            selected_by_speaker[speaker_id].append(item)
            progress = True

        if all(
            len(selected_by_speaker[speaker_id]) >= targets[speaker_id]
            for speaker_id in speaker_ids
        ):
            break

    missing = {
        speaker_id: targets[speaker_id] - len(selected_by_speaker[speaker_id])
        for speaker_id in speaker_ids
        if len(selected_by_speaker[speaker_id]) < targets[speaker_id]
    }
    if missing:
        raise ValueError(f"Failed to satisfy per-speaker targets: {missing}")

    rows: list[dict[str, str]] = []
    for speaker_id in sorted(speaker_ids):
        items = sorted(selected_by_speaker[speaker_id], key=lambda item: item["clip_id"])
        for idx, item in enumerate(items, start=1):
            rows.append(
                {
                    "train_item_id": f"{speaker_id}_{idx:02d}",
                    "clip_id": item["clip_id"],
                    "speaker_id": speaker_id,
                    "sentence_id": item["sentence_id"],
                    "audio_duration_sec": item["audio_duration_sec"],
                    "audio_path": item["audio_path"],
                    "text_path": item["text_path"],
                    "text": item["text"],
                    "split": "train_source",
                }
            )

    return rows


def write_train_source_clips(path: Path, clip_rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "train_item_id",
                "clip_id",
                "speaker_id",
                "sentence_id",
                "audio_duration_sec",
                "audio_path",
                "text_path",
                "text",
                "split",
            ],
        )
        writer.writeheader()
        writer.writerows(clip_rows)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    selection_dir = args.selection_dir
    test_speakers_path = selection_dir / "test_speakers_50.csv"
    test_source_path = selection_dir / "test_source_clips_100.csv"

    txt_root = args.vctk_root / "txt"
    wav_root = args.vctk_root / "wav48"
    speaker_info_path = args.vctk_root / "speaker-info.txt"

    speaker_map = load_speaker_info(speaker_info_path)
    sentences_per_speaker = discover_sentences_per_speaker(txt_root, wav_root)

    test_speaker_rows = read_csv_rows(test_speakers_path)
    test_source_rows = read_csv_rows(test_source_path)

    test_speaker_ids = {row["speaker_id"] for row in test_speaker_rows}
    excluded_texts = {normalize_text(row["text"]) for row in test_source_rows}

    candidates = sorted(set(speaker_map) & set(sentences_per_speaker))
    train_speaker_ids = build_train_speaker_ids(candidates, test_speaker_ids)
    targets = per_speaker_targets(train_speaker_ids, args.num_train_clips, rng)

    candidate_map = collect_candidate_clips(
        speaker_ids=train_speaker_ids,
        sentences_per_speaker=sentences_per_speaker,
        vctk_root=args.vctk_root,
        min_clip_duration_sec=args.min_clip_duration_sec,
        excluded_texts=excluded_texts,
    )

    train_clip_rows = select_train_clips(
        speaker_ids=train_speaker_ids,
        targets=targets,
        candidate_map=candidate_map,
        excluded_texts=excluded_texts,
        rng=rng,
    )

    selection_dir.mkdir(parents=True, exist_ok=True)
    train_speakers_path = selection_dir / f"train_speakers_{len(train_speaker_ids)}.csv"
    train_source_path = selection_dir / f"train_source_clips_{len(train_clip_rows)}.csv"
    train_report_path = selection_dir / "train_selection_report.json"

    write_speaker_csv(train_speakers_path, train_speaker_ids, speaker_map)
    write_train_source_clips(train_source_path, train_clip_rows)

    train_texts = [normalize_text(row["text"]) for row in train_clip_rows]
    durations = [float(row["audio_duration_sec"]) for row in train_clip_rows]
    per_speaker_counts = Counter(row["speaker_id"] for row in train_clip_rows)

    report = {
        "seed": args.seed,
        "vctk_root": str(args.vctk_root),
        "candidate_speakers": len(candidates),
        "test_speakers": len(test_speaker_ids),
        "train_speakers": len(train_speaker_ids),
        "num_train_clips": len(train_clip_rows),
        "min_clip_duration_sec": args.min_clip_duration_sec,
        "excluded_test_text_count": len(excluded_texts),
        "duplicate_text_overlap_with_test": sum(
            1 for text in train_texts if text in excluded_texts
        ),
        "duplicate_texts_within_train": len(train_texts) - len(set(train_texts)),
        "min_audio_duration_sec_in_train_source": min(durations) if durations else None,
        "max_audio_duration_sec_in_train_source": max(durations) if durations else None,
        "speaker_clip_count_distribution": count_by([str(v) for v in per_speaker_counts.values()]),
        "gender_counts": count_by([speaker_map[s].gender for s in train_speaker_ids]),
        "accent_counts": count_by([speaker_map[s].accent_group for s in train_speaker_ids]),
        "output_files": {
            "train_speakers": str(train_speakers_path),
            "train_source_clips": str(train_source_path),
        },
    }

    with train_report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

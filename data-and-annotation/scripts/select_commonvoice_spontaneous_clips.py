#!/usr/bin/env python3
"""Select multilingual Common Voice Spontaneous Speech source clips.

For each locale, this script selects 100 train and 20 test clips shorter than
30 seconds. Speakers (``client_id``) are kept disjoint between train and test.
Selected MP3 files are converted to 16 kHz mono PCM WAV for the VocalGrad
generation scripts.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import librosa
import soundfile as sf

DEFAULT_SOURCE_ROOT = Path("data/original/CommonVoice-Spontaneous")
DEFAULT_OUTPUT_ROOT = Path("data/selected/commonvoice_spontaneous")
LOCALES = ("cdo", "cgg", "kbd", "qxp", "shi")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--num-train-per-language", type=int, default=100)
    parser.add_argument("--num-test-per-language", type=int, default=20)
    parser.add_argument("--max-duration-sec", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def find_locale_root(source_root: Path, locale: str) -> Path:
    matches = sorted(source_root.glob(f"sps-corpus-*-{locale}"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one source directory for {locale!r}, found {len(matches)}."
        )
    return matches[0]


def load_candidates(
    source_root: Path,
    locale: str,
    max_duration_sec: float,
) -> list[dict[str, Any]]:
    locale_root = find_locale_root(source_root, locale)
    tsv_path = locale_root / f"ss-corpus-{locale}.tsv"
    with tsv_path.open("r", encoding="utf-8") as file:
        source_rows = list(csv.DictReader(file, delimiter="\t"))

    # Some releases contain multiple metadata rows for one physical audio file.
    by_audio_file: dict[str, dict[str, Any]] = {}
    for row in source_rows:
        duration_sec = float(row["duration_ms"]) / 1000.0
        if duration_sec >= max_duration_sec:
            continue
        audio_path = locale_root / "audios" / row["audio_file"]
        if not audio_path.exists():
            continue
        by_audio_file.setdefault(
            row["audio_file"],
            {
                **row,
                "locale": locale,
                "duration_sec": duration_sec,
                "source_audio_path": str(audio_path),
                "source_tsv_path": str(tsv_path),
            },
        )
    return list(by_audio_file.values())


def choose_speaker_partition(
    candidates: list[dict[str, Any]],
    *,
    num_train: int,
    num_test: int,
    rng: random.Random,
    attempts: int = 100_000,
) -> tuple[set[str], set[str]]:
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_speaker[row["client_id"]].append(row)
    speakers = sorted(by_speaker)
    if len(speakers) < 2:
        raise ValueError("At least two speakers are required for speaker-disjoint splits.")

    test_probability = num_test / (num_train + num_test)
    for _ in range(attempts):
        test_speakers = {speaker for speaker in speakers if rng.random() < test_probability}
        if not test_speakers or len(test_speakers) == len(speakers):
            continue
        train_speakers = set(speakers) - test_speakers
        test_capacity = sum(len(by_speaker[speaker]) for speaker in test_speakers)
        train_capacity = sum(len(by_speaker[speaker]) for speaker in train_speakers)
        if test_capacity >= num_test and train_capacity >= num_train:
            return train_speakers, test_speakers

    raise ValueError(
        f"Could not find a speaker-disjoint partition with train={num_train}, test={num_test}."
    )


def select_locale(
    candidates: list[dict[str, Any]],
    *,
    num_train: int,
    num_test: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_speakers, test_speakers = choose_speaker_partition(
        candidates,
        num_train=num_train,
        num_test=num_test,
        rng=rng,
    )
    train_candidates = [row for row in candidates if row["client_id"] in train_speakers]
    test_candidates = [row for row in candidates if row["client_id"] in test_speakers]
    return rng.sample(train_candidates, num_train), rng.sample(test_candidates, num_test)


def convert_to_wav(source: Path, destination: Path, *, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    audio, _ = librosa.load(source, sr=16_000, mono=True)
    temporary = destination.with_suffix(".part.wav")
    try:
        sf.write(temporary, audio, 16_000, subtype="PCM_16", format="WAV")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def output_row(
    row: dict[str, Any],
    *,
    split: str,
    item_index: int,
    output_root: Path,
) -> dict[str, str]:
    locale = row["locale"]
    clip_id = Path(row["audio_file"]).stem
    audio_path = output_root / split / "audio" / locale / f"{clip_id}.wav"
    return {
        f"{split}_item_id": f"cvss_{split}_{item_index:04d}",
        "clip_id": clip_id,
        "speaker_id": f"{locale}:{row['client_id']}",
        "language": row["language"],
        "locale": locale,
        "gender": row["gender"],
        "age": row["age"],
        "duration_sec": f"{row['duration_sec']:.6f}",
        "prompt": row["prompt"],
        "text": row["transcription"],
        "audio_path": str(audio_path),
        "source_audio_path": row["source_audio_path"],
        "source_tsv_path": row["source_tsv_path"],
        "source_audio_id": row["audio_id"],
        "source_prompt_id": row["prompt_id"],
        "source_split": row["split"],
        "split": split,
    }


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.max_duration_sec <= 0:
        raise ValueError("--max-duration-sec must be positive.")

    selected: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    for locale_index, locale in enumerate(LOCALES):
        candidates = load_candidates(args.source_root, locale, args.max_duration_sec)
        rng = random.Random(args.seed + locale_index)
        train_rows, test_rows = select_locale(
            candidates,
            num_train=args.num_train_per_language,
            num_test=args.num_test_per_language,
            rng=rng,
        )
        selected["train"].extend(train_rows)
        selected["test"].extend(test_rows)
        print(
            f"{locale}: candidates={len(candidates)}, "
            f"train_speakers={len({row['client_id'] for row in train_rows})}, "
            f"test_speakers={len({row['client_id'] for row in test_rows})}"
        )

    for split in ("test",):
        rows = selected[split]
        random.Random(args.seed + (10 if split == "train" else 11)).shuffle(rows)
        manifest_rows: list[dict[str, str]] = []
        for index, row in enumerate(rows, start=1):
            manifest_row = output_row(
                row,
                split=split,
                item_index=index,
                output_root=args.output_root,
            )
            convert_to_wav(
                Path(row["source_audio_path"]),
                Path(manifest_row["audio_path"]),
                overwrite=args.overwrite,
            )
            manifest_rows.append(manifest_row)
            if index % 50 == 0:
                print(f"{split}: converted {index}/{len(rows)}")

        manifest_path = args.output_root / f"{split}_source_clips_{len(rows)}.csv"
        write_manifest(manifest_path, manifest_rows)
        language_counts = dict(sorted(Counter(row["locale"] for row in rows).items()))
        print(f"Wrote {manifest_path}: {language_counts}")


if __name__ == "__main__":
    main()

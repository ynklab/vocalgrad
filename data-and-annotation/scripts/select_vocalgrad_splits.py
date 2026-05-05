#!/usr/bin/env python3
"""Select VocalGrad test source clips from local VCTK data.

Current protocol:
- pick 50 speakers with balanced gender/accent
- for each selected speaker, pick 2 clips with duration >= threshold
- store clip-level metadata including transcript text
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import wave
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Speaker:
    speaker_id: str
    age: str
    gender: str
    accent_raw: str
    accent_group: str
    region: str


def canonical_accent(accent_raw: str) -> str:
    lower = accent_raw.strip().lower()
    if "american" in lower:
        return "American"
    if "english" in lower:
        return "English"
    if "scottish" in lower:
        return "Scottish"
    return "Others"


def load_speaker_info(path: Path) -> dict[str, Speaker]:
    speakers: dict[str, Speaker] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("ID"):
                continue
            tokens = line.split()
            if len(tokens) < 4:
                continue
            speaker_num, age, gender, accent_raw = tokens[:4]
            region = " ".join(tokens[4:])
            speaker_id = f"p{int(speaker_num):03d}"
            speakers[speaker_id] = Speaker(
                speaker_id=speaker_id,
                age=age,
                gender=gender,
                accent_raw=accent_raw,
                accent_group=canonical_accent(accent_raw),
                region=region,
            )
    return speakers


def discover_sentences_per_speaker(txt_root: Path, wav_root: Path) -> dict[str, set[str]]:
    sentences: dict[str, set[str]] = {}
    txt_speakers = {p.name for p in txt_root.glob("p*") if p.is_dir()}
    wav_speakers = {p.name for p in wav_root.glob("p*") if p.is_dir()}

    for speaker_id in sorted(txt_speakers & wav_speakers):
        txt_dir = txt_root / speaker_id
        wav_dir = wav_root / speaker_id
        per_speaker: set[str] = set()
        for txt_path in txt_dir.glob("*.txt"):
            stem = txt_path.stem
            wav_path = wav_dir / f"{stem}.wav"
            if not wav_path.exists() or "_" not in stem:
                continue
            sentence_id = stem.split("_", maxsplit=1)[1]
            per_speaker.add(sentence_id)
        if per_speaker:
            sentences[speaker_id] = per_speaker
    return sentences


def make_targets(total: int, labels: list[str]) -> dict[str, int]:
    labels = sorted(labels)
    base = total // len(labels)
    rem = total % len(labels)
    out = {label: base for label in labels}
    for label in labels[:rem]:
        out[label] += 1
    return out


def sample_score(
    speaker_ids: list[str],
    speaker_map: dict[str, Speaker],
    gender_targets: dict[str, int],
    accent_targets: dict[str, int],
) -> int:
    genders = Counter(speaker_map[s].gender for s in speaker_ids)
    accents = Counter(speaker_map[s].accent_group for s in speaker_ids)

    gender_score = sum(abs(genders[g] - gender_targets[g]) for g in gender_targets)
    accent_score = sum(abs(accents[a] - accent_targets[a]) for a in accent_targets)
    return 10 * gender_score + accent_score


def pick_balanced_speakers(
    candidates: list[str],
    speaker_map: dict[str, Speaker],
    n_speakers: int,
    rng: random.Random,
    search_iters: int,
) -> list[str]:
    genders = sorted(
        {speaker_map[s].gender for s in candidates if speaker_map[s].gender in {"F", "M"}}
    )
    if not genders:
        genders = sorted({speaker_map[s].gender for s in candidates})
    accents = sorted({speaker_map[s].accent_group for s in candidates})

    gender_targets = make_targets(n_speakers, genders)
    accent_targets = make_targets(n_speakers, accents)

    best: list[str] | None = None
    best_score: int | None = None

    for _ in range(search_iters):
        sampled = rng.sample(candidates, n_speakers)
        score = sample_score(sampled, speaker_map, gender_targets, accent_targets)
        if best_score is None or score < best_score:
            best = sampled
            best_score = score
            if score == 0:
                break

    if best is None:
        raise RuntimeError("Failed to select speakers.")
    return sorted(best)


def read_wav_duration_sec(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        framerate = wf.getframerate()
        nframes = wf.getnframes()
    return nframes / framerate


def read_transcript_text(path: Path) -> str:
    raw = path.read_text(encoding="utf-8")
    return " ".join(raw.split())


def write_speaker_csv(path: Path, speaker_ids: list[str], speaker_map: dict[str, Speaker]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "speaker_id",
                "age",
                "gender",
                "accent_raw",
                "accent_group",
                "region",
            ],
        )
        writer.writeheader()
        for speaker_id in sorted(speaker_ids):
            s = speaker_map[speaker_id]
            writer.writerow(
                {
                    "speaker_id": s.speaker_id,
                    "age": s.age,
                    "gender": s.gender,
                    "accent_raw": s.accent_raw,
                    "accent_group": s.accent_group,
                    "region": s.region,
                }
            )


def select_source_clips(
    selected_speakers: list[str],
    sentences_per_speaker: dict[str, set[str]],
    clips_per_speaker: int,
    min_clip_duration_sec: float,
    exclude_clip_ids: set[str],
    vctk_root: Path,
    rng: random.Random,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for speaker_id in sorted(selected_speakers):
        candidates: list[dict[str, str]] = []
        for sentence_id in sorted(sentences_per_speaker[speaker_id]):
            clip_id = f"{speaker_id}_{sentence_id}"
            if clip_id in exclude_clip_ids:
                continue

            wav_path = vctk_root / "wav48" / speaker_id / f"{clip_id}.wav"
            txt_path = vctk_root / "txt" / speaker_id / f"{clip_id}.txt"
            if not wav_path.exists() or not txt_path.exists():
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
                    "text": read_transcript_text(txt_path),
                }
            )

        if len(candidates) < clips_per_speaker:
            raise ValueError(
                f"Speaker {speaker_id} has only {len(candidates)} clips "
                f"with duration >= {min_clip_duration_sec:.3f}s."
            )

        selected_for_speaker = rng.sample(candidates, clips_per_speaker)
        selected_for_speaker = sorted(selected_for_speaker, key=lambda x: x["clip_id"])

        for idx, item in enumerate(selected_for_speaker, start=1):
            rows.append(
                {
                    "test_item_id": f"{speaker_id}_{idx:02d}",
                    "clip_id": item["clip_id"],
                    "speaker_id": speaker_id,
                    "sentence_id": item["sentence_id"],
                    "audio_duration_sec": item["audio_duration_sec"],
                    "audio_path": item["audio_path"],
                    "text_path": item["text_path"],
                    "text": item["text"],
                    "split": "test_source",
                }
            )

    return rows


def write_selected_sentences_csv(path: Path, clip_rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "speaker_id",
                "sentence_id",
                "clip_id",
                "audio_duration_sec",
                "text_path",
            ],
        )
        writer.writeheader()
        for row in clip_rows:
            writer.writerow(
                {
                    "speaker_id": row["speaker_id"],
                    "sentence_id": row["sentence_id"],
                    "clip_id": row["clip_id"],
                    "audio_duration_sec": row["audio_duration_sec"],
                    "text_path": row["text_path"],
                }
            )


def write_test_source_clips(path: Path, clip_rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "test_item_id",
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


def count_by(items: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(items).items()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vctk-root", type=Path, default=Path("data/original/VCTK-Corpus"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/metadata/selection"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-speakers", type=int, default=50)
    parser.add_argument("--clips-per-speaker", type=int, default=2)
    parser.add_argument("--search-iters", type=int, default=20000)
    parser.add_argument(
        "--min-clip-duration-sec",
        type=float,
        default=3.0,
        help="Keep only clips with duration >= this threshold.",
    )
    parser.add_argument(
        "--exclude-clip-id",
        action="append",
        default=[],
        help="Clip ID to exclude from source clip sampling (repeatable). Example: p360_140",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rng = random.Random(args.seed)

    txt_root = args.vctk_root / "txt"
    wav_root = args.vctk_root / "wav48"
    speaker_info_path = args.vctk_root / "speaker-info.txt"

    speaker_map = load_speaker_info(speaker_info_path)
    sentences_per_speaker = discover_sentences_per_speaker(txt_root, wav_root)

    candidates = sorted(set(speaker_map) & set(sentences_per_speaker))
    if len(candidates) < args.test_speakers:
        raise ValueError(
            "Need at least "
            f"{args.test_speakers} speakers, but only {len(candidates)} are available."
        )

    selected_speakers = pick_balanced_speakers(
        candidates=candidates,
        speaker_map=speaker_map,
        n_speakers=args.test_speakers,
        rng=rng,
        search_iters=args.search_iters,
    )

    clip_rows = select_source_clips(
        selected_speakers=selected_speakers,
        sentences_per_speaker=sentences_per_speaker,
        clips_per_speaker=args.clips_per_speaker,
        min_clip_duration_sec=args.min_clip_duration_sec,
        exclude_clip_ids=set(args.exclude_clip_id),
        vctk_root=args.vctk_root,
        rng=rng,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    speaker_path = args.out_dir / f"test_speakers_{len(selected_speakers)}.csv"
    clip_count = len(selected_speakers) * args.clips_per_speaker
    sentence_path = args.out_dir / f"selected_sentences_{clip_count}.csv"
    clips_path = args.out_dir / f"test_source_clips_{clip_count}.csv"

    write_speaker_csv(speaker_path, selected_speakers, speaker_map)
    write_selected_sentences_csv(sentence_path, clip_rows)
    write_test_source_clips(clips_path, clip_rows)

    sentence_freq = Counter(row["sentence_id"] for row in clip_rows)
    durations = [float(row["audio_duration_sec"]) for row in clip_rows]

    report = {
        "seed": args.seed,
        "vctk_root": str(args.vctk_root),
        "candidate_speakers": len(candidates),
        "test_speakers": len(selected_speakers),
        "clips_per_speaker": args.clips_per_speaker,
        "test_source_clips": len(clip_rows),
        "min_clip_duration_sec": args.min_clip_duration_sec,
        "excluded_clip_ids": sorted(set(args.exclude_clip_id)),
        "duplicate_sentence_ids_in_test_source": sum(
            1 for freq in sentence_freq.values() if freq > 1
        ),
        "max_sentence_frequency_in_test_source": (
            max(sentence_freq.values()) if sentence_freq else 0
        ),
        "min_audio_duration_sec_in_test_source": min(durations) if durations else None,
        "max_audio_duration_sec_in_test_source": max(durations) if durations else None,
        "gender_counts": count_by([speaker_map[s].gender for s in selected_speakers]),
        "accent_counts": count_by([speaker_map[s].accent_group for s in selected_speakers]),
        "output_files": {
            "test_speakers": str(speaker_path),
            "selected_sentences": str(sentence_path),
            "test_source_clips": str(clips_path),
        },
    }

    report_path = args.out_dir / "selection_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

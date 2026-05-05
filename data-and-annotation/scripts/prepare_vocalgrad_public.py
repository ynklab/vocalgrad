from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

DEFAULT_PROCESSED_ROOT = Path("data/processed")
DEFAULT_OUTPUT_ROOT = Path("data/vocalgrad_public")

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

ATTRIBUTE_BY_CATEGORY = {
    "volume": "volume",
    "speaking_speed": "speaking speed",
    "voice_pitch": "voice pitch",
    "background_noise": "background noise",
    "audio_distortion": "audio distortion",
    "audio_roughness": "audio roughness",
    "voice_clarity": "voice clarity",
    "voice_vibration": "voice vibration",
    "echo": "echo",
}

METADATA_COLUMNS = [
    "file_name",
    "category",
    "benchmark_clip_id",
    "source_text",
    "attribute",
    "prompt",
    "label",
    "direction",
    "curve",
    "tier",
    "source_clip_id",
    "speaker_id",
    "sentence_id",
    "duration_sec",
    "onset_sec",
    "offset_sec",
    "split",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the public test-only VocalGrad dataset for Hugging Face.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=DEFAULT_PROCESSED_ROOT,
        help="Root of the full processed VocalGrad dataset.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory to create for the public dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the output directory before rebuilding it.",
    )
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def label_for_direction(direction: str) -> str:
    if direction == "up":
        return "increase"
    if direction == "down":
        return "decrease"
    raise ValueError(f"Unknown direction: {direction}")


def flat_audio_name(category: str, filename: str) -> str:
    return f"{category}__{filename}"


def write_readme(output_root: Path) -> None:
    readme = """---
license: cc-by-4.0
configs:
  - config_name: default
    drop_labels: true
---

# VocalGrad

VocalGrad is an audio benchmark for evaluating whether a model can detect the
direction of gradual perceptual change in speech. This public release contains
the test split only.

Each example contains one audio clip and one target attribute. The task is to
answer whether that attribute increases or decreases over time.

## Task

Given an audio clip and an attribute name, predict one of two labels:

- `increase`
- `decrease`

The ground-truth label is derived from the metadata field `direction`:

- `direction=up` -> `increase`
- `direction=down` -> `decrease`

## Public Release Contents

- split: `test`
- categories: `9`
- source clips per category: `100`
- combinations per source clip: `24`
- factorization: `3 tiers x 4 curves x 2 directions`
- generated clips per category: `2400`
- total audio clips: `21600`

## Hugging Face Audio Layout

This repository follows the Hugging Face audio dataset convention:

```text
README.md
metadata.csv
audio/
metadata/summary.json
```

- `metadata.csv` is located at the repository root.
- `metadata.csv` contains a `file_name` column.
- `file_name` contains full relative paths to benchmark clips in the flat
  `audio/` directory.
- `audio/` contains only benchmark clips referenced by `metadata.csv`.

## Metadata Columns

Common columns include:

- `file_name`
- `category`
- `benchmark_clip_id`
- `source_text`
- `attribute`
- `prompt`
- `label`
- `direction`
- `curve`
- `tier`
- `source_clip_id`
- `speaker_id`
- `sentence_id`
- `duration_sec`
- `onset_sec`
- `offset_sec`
- `split`

## Categories

- `volume`
- `speaking_speed`
- `voice_pitch`
- `background_noise`
- `audio_distortion`
- `audio_roughness`
- `voice_clarity`
- `voice_vibration`
- `echo`

## Prompting

Use the following prompt structure:

```text
Does the {attribute} increase or decrease over time?

Answer with only one word: "increase" or "decrease".
```

Current category-to-attribute mapping:

- `volume` -> `volume`
- `speaking_speed` -> `speaking speed`
- `voice_pitch` -> `voice pitch`
- `background_noise` -> `background noise`
- `audio_distortion` -> `audio distortion`
- `audio_roughness` -> `audio roughness`
- `voice_clarity` -> `voice clarity`
- `voice_vibration` -> `voice vibration`
- `echo` -> `echo`

## Evaluation

Normalize model output to lowercase text. Accepted labels are:

- `increase`
- `decrease`

Any other output should be treated as invalid and counted as incorrect.

Main metric:

```text
accuracy = (# correct predictions) / (# total samples)
```

Recommended breakdowns:

- overall
- per category
- per tier
- per curve
- per direction

## License

VocalGrad is released under the Creative Commons Attribution 4.0 International
License (CC BY 4.0).

The source speech clips are derived from the CSTR VCTK Corpus version 0.92,
which is distributed under the Creative Commons Attribution 4.0 International
Public License.

## Source Dataset Credit

VocalGrad uses source speech from the CSTR VCTK Corpus:

Yamagishi, Junichi; Veaux, Christophe; MacDonald, Kirsten. (2019). CSTR VCTK
Corpus: English Multi-speaker Corpus for CSTR Voice Cloning Toolkit (version
0.92), [sound]. University of Edinburgh. The Centre for Speech Technology
Research (CSTR). https://doi.org/10.7488/ds/2645.

Original dataset record:
https://datashare.ed.ac.uk/handle/10283/3443

License:
Creative Commons Attribution 4.0 International
https://creativecommons.org/licenses/by/4.0/

## Modifications

The audio clips in VocalGrad are transformed derivatives of VCTK source clips.
VocalGrad applies controlled time-varying transformations to create benchmark
examples for detecting whether a target perceptual attribute increases or
decreases over time. The included metadata identifies the source clip ID and
the transformation condition for each derived clip.
"""
    (output_root / "README.md").write_text(readme)


def main() -> None:
    args = parse_args()
    processed_root = args.processed_root
    output_root = args.output_root

    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    metadata_rows: list[dict[str, str]] = []
    category_counts: dict[str, int] = {}

    for category in CATEGORIES:
        manifest_path = processed_root / "test" / category / "metadata" / f"{category}_manifest.csv"
        rows = read_manifest(manifest_path)
        if len(rows) != 2400:
            raise SystemExit(f"Expected 2400 rows for {category}, found {len(rows)}")

        category_counts[category] = len(rows)
        for row in rows:
            input_audio = Path(row["output_audio_path"])
            if not input_audio.is_file():
                raise FileNotFoundError(input_audio)

            audio_dest = output_root / "audio" / flat_audio_name(category, input_audio.name)
            audio_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_audio, audio_dest)
            relative_audio_path = audio_dest.relative_to(output_root).as_posix()

            attribute = ATTRIBUTE_BY_CATEGORY[category]
            metadata_rows.append(
                {
                    "file_name": relative_audio_path,
                    "category": category,
                    "benchmark_clip_id": row["benchmark_clip_id"],
                    "source_text": row["source_text"],
                    "attribute": attribute,
                    "prompt": (
                        f"Does the {attribute} increase or decrease over time? "
                        'Answer with only one word: "increase" or "decrease".'
                    ),
                    "label": label_for_direction(row["direction"]),
                    "direction": row["direction"],
                    "curve": row["curve"],
                    "tier": row["tier"],
                    "source_clip_id": row["source_clip_id"],
                    "speaker_id": row["speaker_id"],
                    "sentence_id": row["sentence_id"],
                    "duration_sec": row.get("duration_sec", ""),
                    "onset_sec": row.get("onset_sec", ""),
                    "offset_sec": row.get("offset_sec", ""),
                    "split": "test",
                }
            )

    write_csv(output_root / "metadata.csv", metadata_rows, fieldnames=METADATA_COLUMNS)

    summary = {
        "name": "VocalGrad",
        "split": "test",
        "categories": CATEGORIES,
        "clips_per_category": category_counts,
        "total_clips": len(metadata_rows),
        "metadata_file": "metadata.csv",
        "audio_path_column": "file_name",
        "license": "CC BY 4.0",
        "source_dataset": "CSTR VCTK Corpus version 0.92",
        "source_dataset_doi": "https://doi.org/10.7488/ds/2645",
    }
    metadata_path = output_root / "metadata" / "summary.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(summary, indent=2) + "\n")
    write_readme(output_root)

    print(f"Wrote public VocalGrad dataset to {output_root}")
    print(f"Clips: {len(metadata_rows)}")
    print(f"Categories: {len(category_counts)}")


if __name__ == "__main__":
    main()

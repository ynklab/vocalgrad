from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ACTIVE_CATEGORY_SPECS = [
    ("volume", Path("scripts/generate_volume_benchmark.py"), Path("configs/volume_benchmark.json")),
    (
        "speaking_speed",
        Path("scripts/generate_speaking_speed_benchmark.py"),
        Path("configs/speaking_speed_benchmark.json"),
    ),
    (
        "voice_pitch",
        Path("scripts/generate_voice_pitch_benchmark.py"),
        Path("configs/voice_pitch_benchmark.json"),
    ),
    (
        "background_noise",
        Path("scripts/generate_background_noise_benchmark.py"),
        Path("configs/background_noise_benchmark.json"),
    ),
    (
        "audio_distortion",
        Path("scripts/generate_audio_distortion_benchmark.py"),
        Path("configs/audio_distortion_benchmark.json"),
    ),
    (
        "audio_roughness",
        Path("scripts/generate_audio_roughness_benchmark.py"),
        Path("configs/audio_roughness_benchmark.json"),
    ),
    (
        "voice_clarity",
        Path("scripts/generate_voice_clarity_benchmark.py"),
        Path("configs/voice_clarity_benchmark.json"),
    ),
    (
        "voice_vibration",
        Path("scripts/generate_voice_vibration_benchmark.py"),
        Path("configs/voice_vibration_benchmark.json"),
    ),
    ("echo", Path("scripts/generate_echo_benchmark.py"), Path("configs/echo_benchmark.json")),
]

SPLIT_SOURCE_CSV = {
    "test": Path("data/metadata/selection/test_source_clips_100.csv"),
    "train": Path("data/metadata/selection/train_source_clips_500.csv"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate active VocalGrad speech benchmarks.")
    parser.add_argument(
        "--split",
        choices=["test", "train", "all"],
        default="all",
        help="Speech split to generate.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/processed"),
        help="Root directory for generated benchmark outputs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of source clips to process per category.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def run_generator(
    *,
    script_path: Path,
    source_csv: Path,
    config_path: Path,
    output_root: Path,
    limit: int | None,
    overwrite: bool,
) -> None:
    command = [
        sys.executable,
        str(script_path),
        "--source-csv",
        str(source_csv),
        "--config",
        str(config_path),
        "--output-root",
        str(output_root),
    ]
    if limit is not None:
        command.extend(["--limit", str(limit)])
    if overwrite:
        command.append("--overwrite")

    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    splits = ["test", "train"] if args.split == "all" else [args.split]

    for split in splits:
        source_csv = SPLIT_SOURCE_CSV[split]
        for category, script_path, config_path in ACTIVE_CATEGORY_SPECS:
            output_root = args.processed_root / split / category
            print(f"[{split}] generating {category} -> {output_root}")
            run_generator(
                script_path=script_path,
                source_csv=source_csv,
                config_path=config_path,
                output_root=output_root,
                limit=args.limit,
                overwrite=args.overwrite,
            )


if __name__ == "__main__":
    main()

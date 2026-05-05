from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SOURCE_CSV = Path("data/selected/source_clips/ablation_beep/manifest.csv")
OUTPUT_ROOT = Path("data/processed/ablation_beep")

CATEGORY_SPECS = [
    ("volume", Path("scripts/generate_volume_benchmark.py"), Path("configs/volume_benchmark.json")),
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
        "voice_vibration",
        Path("scripts/generate_voice_vibration_benchmark.py"),
        Path("configs/voice_vibration_benchmark.json"),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the four VocalGrad ablation_beep benchmark categories.",
    )
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=SOURCE_CSV,
        help="Source clip CSV for the ablation study.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Root directory for ablation_beep processed outputs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of source clips to process.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite any existing outputs.",
    )
    return parser.parse_args()


def run_generator(
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


def write_readme(output_root: Path) -> None:
    readme_path = output_root / "README.md"
    readme_path.write_text(
        "# Ablation Beep Processed Data\n\n"
        "This directory contains benchmark datasets generated from the 10 synthetic "
        "`ablation_beep` source clips.\n\n"
        "## Layout\n\n"
        "- `<category>/audio/up/`\n"
        "- `<category>/audio/down/`\n"
        "- `<category>/metadata/*`\n\n"
        "## Notes\n\n"
        "- source clips: `data/selected/source_clips/ablation_beep/manifest.csv`\n"
        "- categories: 4 (`voice_pitch`, `volume`, `background_noise`, `voice_vibration`)\n"
        "- conditions per source clip: `3 tiers x 4 curves x 2 directions = 24`\n"
        "- expected outputs per category: `10 x 24 = 240`\n"
        "- expected total outputs across all categories: `960`\n",
    )


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_readme(args.output_root)

    for category, script_path, config_path in CATEGORY_SPECS:
        category_output_root = args.output_root / category
        print(f"[ablation_beep] generating {category} -> {category_output_root}")
        run_generator(
            script_path=script_path,
            source_csv=args.source_csv,
            config_path=config_path,
            output_root=category_output_root,
            limit=args.limit,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()

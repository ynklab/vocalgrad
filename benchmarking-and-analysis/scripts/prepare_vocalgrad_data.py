"""Arrange downloaded test audio or frozen VCTK sources for evaluation.

Uses only the Python standard library. Audio bytes are copied or symlinked,
never decoded or transformed. Existing destination files are not overwritten.
"""
from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

CATEGORIES = (
    "speaking_speed", "voice_pitch", "volume", "audio_distortion",
    "audio_roughness", "background_noise", "echo", "voice_clarity", "voice_vibration",
)


def component(value: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"Invalid path component: {value!r}")
    return value


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty metadata: {path}")
    return rows


def check_target(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Destination already exists; use a new output directory: {path}")


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(args: argparse.Namespace) -> None:
    output = args.output_root.resolve()
    transfers: list[tuple[Path, Path]] = []
    manifests: dict[Path, list[dict[str, str]]] = {}
    ids: set[tuple[str, str]] = set()
    destinations: set[Path] = set()
    if args.command == "hf-test":
        source_root = args.download_root.resolve()
        rows = read_rows(source_root / "metadata.csv")
        counts = Counter(row["category"] for row in rows)
        if counts != Counter({c: 2400 for c in CATEGORIES}):
            raise ValueError(f"Expected 2400 rows in each of nine categories; got {dict(counts)}")
        for row in rows:
            category = component(row["category"])
            clip_id = component(row["benchmark_clip_id"])
            direction = row["direction"]
            if direction not in {"up", "down"}:
                raise ValueError(f"Invalid direction: {direction}")
            if row.get("label") != {"up": "increase", "down": "decrease"}[direction]:
                raise ValueError(f"Label/direction mismatch: {clip_id}")
            if row.get("split", "test") != "test":
                raise ValueError(f"Unexpected split: {row.get('split')}")
            relative = Path(row["file_name"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Expected relative file_name: {relative}")
            source = source_root / relative
            target = output / category / "audio" / direction / f"{clip_id}.wav"
            key = (category, clip_id)
            if key in ids:
                raise ValueError(f"Duplicate clip ID: {key}")
            ids.add(key)
            manifests.setdefault(output / category / "metadata" / f"{category}_manifest.csv", []).append(dict(row))
            transfers.append((source, target))
    else:
        rows = read_rows(args.selection_csv)
        if len(rows) != 100:
            raise ValueError(f"Expected 100 frozen test sources; got {len(rows)}")
        source_root = args.vctk_root.resolve()
        for row in rows:
            clip_id = component(row["clip_id"])
            speaker = component(row["speaker_id"])
            item_id = component(row["test_item_id"])
            key = ("source", item_id)
            if key in ids:
                raise ValueError(f"Duplicate test item: {item_id}")
            ids.add(key)
            relative = Path("audio") / speaker / f"{clip_id}.wav"
            source = source_root / "wav48" / speaker / f"{clip_id}.wav"
            target = output / relative
            converted = dict(row)
            converted["original_audio_path"] = row["audio_path"]
            converted["selected_audio_path"] = relative.as_posix()
            manifests.setdefault(output / "manifest.csv", []).append(converted)
            transfers.append((source, target))
    # Preflight metadata, source existence and collisions before writing anything.
    # This does not decode audio or compute expensive content hashes.
    for source, target in transfers:
        if not source.is_file():
            raise FileNotFoundError(source)
        if target in destinations:
            raise ValueError(f"Duplicate destination: {target}")
        destinations.add(target)
        check_target(target)
    for path in manifests:
        check_target(path)
    for source, target in transfers:
        target.parent.mkdir(parents=True, exist_ok=True)
        if args.mode == "symlink":
            target.symlink_to(source.resolve())
        else:
            with source.open("rb") as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst)
    for path, records in manifests.items():
        write_rows(path, records)
    print(f"Prepared {len(transfers)} audio files and {len(manifests)} manifests at {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    hf = commands.add_parser("hf-test", help="Convert the complete public HF test release")
    hf.add_argument("--download-root", type=Path, required=True)
    sources = commands.add_parser("sources", help="Place the 100 unmodified VCTK source clips")
    sources.add_argument("--selection-csv", type=Path, required=True)
    sources.add_argument("--vctk-root", type=Path, required=True)
    for subparser in (hf, sources):
        subparser.add_argument("--output-root", type=Path, required=True)
        subparser.add_argument("--mode", choices=("copy", "symlink"), default="copy")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate VocalGrad volume-change benchmark clips.

For each source clip, this script applies all combinations of:
- tiers (e.g., high/mid/low)
- curves (linear / quad_first_flat / quad_last_flat / jump)
- directions (up/down)

Example with 100 source clips and 3x4x2 combinations => 2400 outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import wave
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=Path("data/metadata/selection/test_source_clips_100.csv"),
        help="CSV with base source clips.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/volume_benchmark.json"),
        help="JSON config for gain hyperparameters.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/test/volume"),
        help="Category output directory.",
    )
    parser.add_argument(
        "--onset-csv",
        type=Path,
        default=Path("data/metadata/analysis/onset_offset_min10_retuned/selected_clip_onset_times.csv"),
        help=(
            "Optional onset/offset CSV fallback. "
            "If source CSV already has onset_sec/offset_sec columns, they are used first."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing audio files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of base clips (for quick tuning runs).",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    required_top = {"step_duration_sec", "jump_at", "tiers", "curves"}
    missing = required_top - set(cfg)
    if missing:
        raise ValueError(f"Missing config keys: {sorted(missing)}")

    if not 0.0 < float(cfg["jump_at"]) < 1.0:
        raise ValueError("jump_at must be in (0, 1).")

    return cfg


def read_source_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in source CSV: {path}")
    return rows


def read_onset_bounds(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return {}

    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        source_item_id = (
            row.get("test_item_id", "").strip()
            or row.get("train_item_id", "").strip()
            or row.get("source_item_id", "").strip()
        )
        if not source_item_id:
            continue

        detected = str(row.get("detected", "")).strip().lower() == "true"
        onset = float(row["onset_sec"]) if row.get("onset_sec") else None
        offset = float(row["offset_sec"]) if row.get("offset_sec") else None
        out[source_item_id] = {
            "detected": detected,
            "onset_sec": onset,
            "offset_sec": offset,
        }

    return out


def parse_bool_text(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def onset_bounds_from_source_row(row: dict[str, str]) -> tuple[float, float] | None:
    onset_text = row.get("onset_sec", "").strip()
    offset_text = row.get("offset_sec", "").strip()
    if not onset_text or not offset_text:
        return None

    detected_text = row.get("onset_detected", "").strip()
    if detected_text and not parse_bool_text(detected_text):
        return None

    onset = float(onset_text)
    offset = float(offset_text)
    return onset, offset


def validate_onset_coverage(
    source_rows: list[dict[str, str]],
    onset_map: dict[str, dict[str, Any]],
) -> None:
    missing: list[str] = []
    invalid: list[str] = []
    for row in source_rows:
        if onset_bounds_from_source_row(row) is not None:
            continue

        source_item_id = (
            row.get("test_item_id", "").strip()
            or row.get("train_item_id", "").strip()
            or row.get("source_item_id", "").strip()
        )
        item = onset_map.get(source_item_id)
        if item is None:
            missing.append(source_item_id)
            continue
        if (not bool(item["detected"])) or item["onset_sec"] is None or item["offset_sec"] is None:
            invalid.append(source_item_id)

    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(
            f"Onset CSV missing {len(missing)} test_item_id entries from source CSV. "
            f"Examples: {preview}"
        )
    if invalid:
        preview = ", ".join(invalid[:10])
        raise ValueError(
            f"Onset CSV has {len(invalid)} entries without valid onset/offset. "
            f"Examples: {preview}"
        )


def progress_value(step_idx: int, n_steps: int, curve_mode: str, jump_at: float) -> float:
    if n_steps <= 1:
        t = 1.0
    else:
        t = step_idx / (n_steps - 1)

    if curve_mode == "linear":
        return t
    if curve_mode in {"quadratic_flat_first", "quad_first_flat"}:
        return t * t
    if curve_mode in {"quadratic_flat_last", "quad_last_flat"}:
        return 1.0 - (1.0 - t) * (1.0 - t)
    if curve_mode == "jump":
        return 0.0 if t < jump_at else 1.0
    raise ValueError(f"Unsupported curve mode: {curve_mode}")


def decode_pcm(frames: bytes, samp_width: int) -> np.ndarray:
    if samp_width == 1:
        return np.frombuffer(frames, dtype=np.uint8).astype(np.float64) - 128.0
    if samp_width == 2:
        return np.frombuffer(frames, dtype=np.int16).astype(np.float64)
    raise ValueError(f"Unsupported sample width: {samp_width}")


def encode_pcm(audio: np.ndarray, samp_width: int) -> bytes:
    if samp_width == 1:
        clipped = np.clip(audio + 128.0, 0.0, 255.0).astype(np.uint8)
        return clipped.tobytes()
    if samp_width == 2:
        clipped = np.clip(audio, -32768.0, 32767.0).astype(np.int16)
        return clipped.tobytes()
    raise ValueError(f"Unsupported sample width: {samp_width}")


def apply_gain_envelope_in_bounds(
    audio: np.ndarray,
    framerate: int,
    step_duration: float,
    delta_gain_db: float,
    direction: str,
    curve_mode: str,
    jump_at: float,
    onset_sec: float,
    offset_sec: float,
) -> tuple[np.ndarray, int, int, int]:
    samples_per_step = max(1, int(round(framerate * step_duration)))
    num_samples = audio.shape[0]
    onset_sample = int(round(onset_sec * framerate))
    offset_sample = int(round(offset_sec * framerate))
    onset_sample = max(0, min(num_samples, onset_sample))
    offset_sample = max(0, min(num_samples, offset_sample))
    if offset_sample <= onset_sample:
        onset_sample, offset_sample = 0, num_samples

    out = np.zeros_like(audio)

    if direction == "up":
        start_gain_db = -delta_gain_db
        end_gain_db = delta_gain_db
    elif direction == "down":
        start_gain_db = delta_gain_db
        end_gain_db = -delta_gain_db
    else:
        raise ValueError(f"Unsupported direction: {direction}")

    start_gain = 10.0 ** (start_gain_db / 20.0)
    end_gain = 10.0 ** (end_gain_db / 20.0)

    if onset_sample > 0:
        out[:onset_sample] = audio[:onset_sample] * start_gain
    if offset_sample < num_samples:
        out[offset_sample:] = audio[offset_sample:] * end_gain

    gain_window_samples = max(1, offset_sample - onset_sample)
    num_steps = max(1, math.ceil(gain_window_samples / samples_per_step))

    for step_idx in range(num_steps):
        start = onset_sample + (step_idx * samples_per_step)
        end = min(onset_sample + ((step_idx + 1) * samples_per_step), offset_sample)
        p = progress_value(step_idx, num_steps, curve_mode, jump_at)

        # Relative to original audio: interpolate from start_gain_db to end_gain_db.
        gain_db = start_gain_db + ((end_gain_db - start_gain_db) * p)
        gain_linear = 10.0 ** (gain_db / 20.0)
        out[start:end] = audio[start:end] * gain_linear

    return out, num_steps, onset_sample, offset_sample


def clipped_ratio(audio: np.ndarray, samp_width: int) -> float:
    if samp_width == 1:
        lo, hi = -128.0, 127.0
    elif samp_width == 2:
        lo, hi = -32768.0, 32767.0
    else:
        raise ValueError(f"Unsupported sample width: {samp_width}")

    clipped = (audio < lo) | (audio > hi)
    return float(np.mean(clipped))


def process_one(
    input_wav: Path,
    output_wav: Path,
    step_duration: float,
    delta_gain_db: float,
    direction: str,
    curve_mode: str,
    jump_at: float,
    onset_sec: float,
    offset_sec: float,
    overwrite: bool,
) -> dict[str, Any]:
    if output_wav.exists() and not overwrite:
        return {"status": "skipped_existing", "output": str(output_wav)}

    output_wav.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(input_wav), "rb") as wav_in:
        params = wav_in.getparams()
        n_channels, samp_width, framerate, n_frames, _, _ = params
        frames = wav_in.readframes(n_frames)

    decoded = decode_pcm(frames, samp_width)
    audio = decoded.reshape(-1, n_channels)

    processed, n_steps, onset_sample, offset_sample = apply_gain_envelope_in_bounds(
        audio=audio,
        framerate=framerate,
        step_duration=step_duration,
        delta_gain_db=delta_gain_db,
        direction=direction,
        curve_mode=curve_mode,
        jump_at=jump_at,
        onset_sec=onset_sec,
        offset_sec=offset_sec,
    )

    ratio = clipped_ratio(processed, samp_width)

    with wave.open(str(output_wav), "wb") as wav_out:
        wav_out.setparams(params)
        wav_out.writeframes(encode_pcm(processed.flatten(), samp_width))

    return {
        "status": "written",
        "output": str(output_wav),
        "duration_sec": n_frames / framerate,
        "num_steps": n_steps,
        "onset_sec_used": onset_sample / framerate,
        "offset_sec_used": offset_sample / framerate,
        "clipped_ratio": ratio,
    }


def write_processed_readme(processed_root: Path) -> None:
    readme_path = processed_root / "README.md"
    if readme_path.exists():
        return

    content = """# Processed Data

This directory contains generated benchmark artifacts and can be uploaded as-is to
cloud storage (for example Google Drive).

## Layout

- `volume/`: volume-change benchmark category
  - `audio/up/`: clips where loudness increases over time
  - `audio/down/`: clips where loudness decreases over time
  - `metadata/volume_manifest.csv`: per-clip metadata and source mapping
  - `metadata/summary.json`: aggregate statistics of the generation run
  - `metadata/generation_config.used.json`: exact config used for this run

## Notes

- Source clips are listed in `data/metadata/selection/test_source_clips_100.csv`.
- Each source clip is expanded by all `(tier x curve x direction)` combinations.
- Gain is applied relative to the original waveform, spanning `-delta_gain_db` to
  `+delta_gain_db` over time.
- Gain changes are applied only between detected onset and offset times. Before onset
  the clip is held at the initial gain, and after offset it is held at the final gain.
- Re-run `scripts/generate_volume_benchmark.py` after editing
  `configs/volume_benchmark.json` to tune perceptual strength.
"""
    readme_path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    source_rows = read_source_rows(args.source_csv)
    onset_map = read_onset_bounds(args.onset_csv)
    validate_onset_coverage(source_rows, onset_map)
    if args.limit is not None:
        source_rows = source_rows[: args.limit]

    curves = list(cfg["curves"].keys())
    tiers = list(cfg["tiers"].keys())
    directions = ["up", "down"]

    audio_root = args.output_root / "audio"
    meta_root = args.output_root / "metadata"
    meta_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []

    for idx, row in enumerate(source_rows):
        source_clip_id = row["clip_id"]
        source_item_id = (
            row.get("test_item_id", "").strip()
            or row.get("train_item_id", "").strip()
            or row.get("source_item_id", "").strip()
            or f"row{idx:04d}"
        )
        source_wav = Path(row["audio_path"])
        text_path = row.get("text_path", "")
        row_bounds = onset_bounds_from_source_row(row)
        if row_bounds is not None:
            onset_sec, offset_sec = row_bounds
        else:
            onset_info = onset_map[source_item_id]
            onset_sec = float(onset_info["onset_sec"])
            offset_sec = float(onset_info["offset_sec"])

        for tier in tiers:
            delta_gain_db = float(cfg["tiers"][tier]["delta_gain_db"])

            for curve in curves:
                curve_mode = str(cfg["curves"][curve]["mode"])

                for direction in directions:
                    out_name = (
                        f"{source_item_id}__{source_clip_id}__{curve}__"
                        f"{tier}__{direction}.wav"
                    )
                    out_wav = audio_root / direction / out_name

                    proc = process_one(
                        input_wav=source_wav,
                        output_wav=out_wav,
                        step_duration=float(cfg["step_duration_sec"]),
                        delta_gain_db=delta_gain_db,
                        direction=direction,
                        curve_mode=curve_mode,
                        jump_at=float(cfg["jump_at"]),
                        onset_sec=onset_sec,
                        offset_sec=offset_sec,
                        overwrite=args.overwrite,
                    )

                    manifest_rows.append(
                        {
                            "benchmark_clip_id": out_wav.stem,
                            "source_item_id": source_item_id,
                            "source_test_item_id": row.get("test_item_id", ""),
                            "source_train_item_id": row.get("train_item_id", ""),
                            "source_clip_id": source_clip_id,
                            "speaker_id": row.get("speaker_id", ""),
                            "sentence_id": row.get("sentence_id", ""),
                            "source_audio_path": str(source_wav),
                            "source_text_path": text_path,
                            "source_text": row.get("text", ""),
                            "output_audio_path": str(out_wav),
                            "category": "volume",
                            "direction": direction,
                            "curve": curve,
                            "curve_mode": curve_mode,
                            "tier": tier,
                            "step_duration_sec": cfg["step_duration_sec"],
                            "delta_gain_db": delta_gain_db,
                            "min_gain_db": -delta_gain_db,
                            "max_gain_db": delta_gain_db,
                            "jump_at": cfg["jump_at"],
                            "onset_sec": onset_sec,
                            "offset_sec": offset_sec,
                            "onset_sec_used": proc.get("onset_sec_used", ""),
                            "offset_sec_used": proc.get("offset_sec_used", ""),
                            "status": proc["status"],
                            "duration_sec": proc.get("duration_sec", ""),
                            "num_steps": proc.get("num_steps", ""),
                            "clipped_ratio": proc.get("clipped_ratio", ""),
                        }
                    )

    manifest_path = meta_root / "volume_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    expected = len(source_rows) * len(tiers) * len(curves) * len(directions)
    clipped_values = [float(r["clipped_ratio"]) for r in manifest_rows if r["clipped_ratio"] != ""]

    summary = {
        "num_source_rows": len(source_rows),
        "num_generated_rows": len(manifest_rows),
        "num_audio_files_expected": expected,
        "combination_count_per_source": len(tiers) * len(curves) * len(directions),
        "output_root": str(args.output_root),
        "manifest_csv": str(manifest_path),
        "config_path": str(args.config),
        "onset_csv": str(args.onset_csv),
        "counts_by_direction": {
            d: sum(1 for r in manifest_rows if r["direction"] == d) for d in directions
        },
        "counts_by_curve": {
            c: sum(1 for r in manifest_rows if r["curve"] == c) for c in curves
        },
        "counts_by_tier": {
            t: sum(1 for r in manifest_rows if r["tier"] == t) for t in tiers
        },
        "mean_clipped_ratio": float(np.mean(clipped_values)) if clipped_values else 0.0,
    }

    (meta_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (meta_root / "generation_config.used.json").write_text(
        json.dumps(cfg, indent=2),
        encoding="utf-8",
    )

    write_processed_readme(args.output_root.parents[1])

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

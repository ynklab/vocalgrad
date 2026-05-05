#!/usr/bin/env python3
"""Generate VocalGrad background-noise benchmark clips."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
        default=Path("configs/background_noise_benchmark.json"),
        help="JSON config for background-noise hyperparameters.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/test/background_noise"),
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

    required_top = {
        "step_duration_sec",
        "apply_region",
        "global_seed",
        "jump_at",
        "tiers",
        "curves",
    }
    missing = required_top - set(cfg)
    if missing:
        raise ValueError(f"Missing config keys: {sorted(missing)}")

    if cfg["apply_region"] != "speech_span":
        raise ValueError("Only apply_region='speech_span' is currently supported.")
    if not 0.0 < float(cfg["jump_at"]) < 1.0:
        raise ValueError("jump_at must be in (0, 1).")

    for tier_name, tier_cfg in cfg["tiers"].items():
        noise_ratio_max = float(tier_cfg["noise_ratio_max"])
        if noise_ratio_max <= 0.0:
            raise ValueError(f"tiers.{tier_name}.noise_ratio_max must be > 0.")

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

    return float(onset_text), float(offset_text)


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


def decode_pcm_normalized(path: Path) -> tuple[np.ndarray, int, int, int]:
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        samp_width = wf.getsampwidth()
        framerate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if samp_width == 1:
        audio = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif samp_width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        audio = audio / 32768.0
    else:
        raise ValueError(f"Unsupported sample width: {samp_width}")

    audio = audio.reshape(-1, n_channels)
    return audio, framerate, n_channels, samp_width


def encode_pcm_normalized(audio: np.ndarray, samp_width: int) -> bytes:
    if samp_width == 1:
        clipped = np.clip((audio * 128.0) + 128.0, 0.0, 255.0).astype(np.uint8)
        return clipped.tobytes()
    if samp_width == 2:
        clipped = np.clip(audio * 32768.0, -32768.0, 32767.0).astype(np.int16)
        return clipped.tobytes()
    raise ValueError(f"Unsupported sample width: {samp_width}")


def clipped_ratio(audio: np.ndarray) -> float:
    return float(np.mean((audio < -1.0) | (audio > 1.0)))


def deterministic_seed(global_seed: int, benchmark_clip_id: str) -> int:
    digest = hashlib.sha256(f"{global_seed}:{benchmark_clip_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def apply_noise_envelope_in_bounds(
    audio: np.ndarray,
    framerate: int,
    step_duration: float,
    noise_ratio_max: float,
    direction: str,
    curve_mode: str,
    jump_at: float,
    onset_sec: float,
    offset_sec: float,
    seed: int,
) -> tuple[np.ndarray, int, float, float]:
    num_samples = audio.shape[0]
    onset_sample = max(0, min(num_samples, int(round(onset_sec * framerate))))
    offset_sample = max(0, min(num_samples, int(round(offset_sec * framerate))))
    if offset_sample <= onset_sample:
        onset_sample, offset_sample = 0, num_samples

    out = audio.copy()
    speech = audio[onset_sample:offset_sample]
    if speech.size == 0:
        return out, 0, 0.0, 0.0

    samples_per_step = max(1, int(round(framerate * step_duration)))
    num_steps = max(1, math.ceil(len(speech) / samples_per_step))

    if direction == "up":
        start_ratio = 0.0
        end_ratio = noise_ratio_max
    elif direction == "down":
        start_ratio = noise_ratio_max
        end_ratio = 0.0
    else:
        raise ValueError(f"Unsupported direction: {direction}")

    reference_rms = np.sqrt(np.mean(speech * speech, axis=0))
    reference_rms = np.maximum(reference_rms, 1e-6).astype(np.float32)

    rng = np.random.default_rng(seed)
    raw_noise = rng.standard_normal(size=audio.shape).astype(np.float32)
    raw_noise_rms = np.sqrt(np.mean(raw_noise * raw_noise, axis=0))
    raw_noise_rms = np.maximum(raw_noise_rms, 1e-6).astype(np.float32)
    unit_noise = raw_noise / raw_noise_rms

    envelope = np.zeros(num_samples, dtype=np.float32)
    envelope[:onset_sample] = float(start_ratio)
    envelope[offset_sample:] = float(end_ratio)
    ratios: list[float] = []
    for step_idx in range(num_steps):
        start = step_idx * samples_per_step
        end = min((step_idx + 1) * samples_per_step, len(speech))
        p = progress_value(step_idx, num_steps, curve_mode, jump_at)
        ratio = start_ratio + ((end_ratio - start_ratio) * p)
        envelope[onset_sample + start : onset_sample + end] = float(ratio)
        ratios.append(float(ratio))

    scaled_noise = unit_noise * (reference_rms * envelope[:, None])
    out = audio + scaled_noise
    return out, num_steps, min(ratios), max(ratios)


def process_one(
    input_wav: Path,
    output_wav: Path,
    step_duration: float,
    noise_ratio_max: float,
    direction: str,
    curve_mode: str,
    jump_at: float,
    onset_sec: float,
    offset_sec: float,
    seed: int,
    overwrite: bool,
) -> dict[str, Any]:
    if output_wav.exists() and not overwrite:
        return {"status": "skipped_existing", "output": str(output_wav)}

    output_wav.parent.mkdir(parents=True, exist_ok=True)

    audio, framerate, n_channels, samp_width = decode_pcm_normalized(input_wav)
    processed, num_steps, min_ratio_used, max_ratio_used = apply_noise_envelope_in_bounds(
        audio=audio,
        framerate=framerate,
        step_duration=step_duration,
        noise_ratio_max=noise_ratio_max,
        direction=direction,
        curve_mode=curve_mode,
        jump_at=jump_at,
        onset_sec=onset_sec,
        offset_sec=offset_sec,
        seed=seed,
    )
    ratio = clipped_ratio(processed)

    with wave.open(str(output_wav), "wb") as wav_out:
        wav_out.setnchannels(n_channels)
        wav_out.setsampwidth(samp_width)
        wav_out.setframerate(framerate)
        wav_out.writeframes(encode_pcm_normalized(processed, samp_width))

    return {
        "status": "written",
        "output": str(output_wav),
        "duration_sec": processed.shape[0] / framerate,
        "num_steps": num_steps,
        "min_ratio_used": min_ratio_used,
        "max_ratio_used": max_ratio_used,
        "clipped_ratio": ratio,
    }


def write_processed_readme(processed_root: Path) -> None:
    readme_path = processed_root / "README.md"
    if readme_path.exists():
        return

    content = """# Processed Data

This directory contains generated benchmark artifacts and can be uploaded as-is to
cloud storage (for example Google Drive).
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
    global_seed = int(cfg["global_seed"])

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
            noise_ratio_max = float(cfg["tiers"][tier]["noise_ratio_max"])

            for curve in curves:
                curve_mode = str(cfg["curves"][curve]["mode"])

                for direction in directions:
                    out_name = (
                        f"{source_item_id}__{source_clip_id}__{curve}__"
                        f"{tier}__{direction}.wav"
                    )
                    out_wav = audio_root / direction / out_name
                    clip_seed = deterministic_seed(
                        global_seed=global_seed,
                        benchmark_clip_id=out_wav.stem,
                    )

                    proc = process_one(
                        input_wav=source_wav,
                        output_wav=out_wav,
                        step_duration=float(cfg["step_duration_sec"]),
                        noise_ratio_max=noise_ratio_max,
                        direction=direction,
                        curve_mode=curve_mode,
                        jump_at=float(cfg["jump_at"]),
                        onset_sec=onset_sec,
                        offset_sec=offset_sec,
                        seed=clip_seed,
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
                            "category": "background_noise",
                            "direction": direction,
                            "curve": curve,
                            "curve_mode": curve_mode,
                            "tier": tier,
                            "step_duration_sec": cfg["step_duration_sec"],
                            "apply_region": cfg["apply_region"],
                            "noise_ratio_max": noise_ratio_max,
                            "min_noise_ratio": 0.0,
                            "max_noise_ratio": noise_ratio_max,
                            "jump_at": cfg["jump_at"],
                            "onset_sec": onset_sec,
                            "offset_sec": offset_sec,
                            "seed": clip_seed,
                            "status": proc["status"],
                            "duration_sec": proc.get("duration_sec", ""),
                            "num_steps": proc.get("num_steps", ""),
                            "min_ratio_used": proc.get("min_ratio_used", ""),
                            "max_ratio_used": proc.get("max_ratio_used", ""),
                            "clipped_ratio": proc.get("clipped_ratio", ""),
                        }
                    )

    manifest_path = meta_root / "background_noise_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    expected = len(source_rows) * len(tiers) * len(curves) * len(directions)
    clipped_values = [
        float(row["clipped_ratio"]) for row in manifest_rows if row["clipped_ratio"] != ""
    ]

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
            direction: sum(1 for row in manifest_rows if row["direction"] == direction)
            for direction in directions
        },
        "counts_by_curve": {
            curve: sum(1 for row in manifest_rows if row["curve"] == curve)
            for curve in curves
        },
        "counts_by_tier": {
            tier: sum(1 for row in manifest_rows if row["tier"] == tier)
            for tier in tiers
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

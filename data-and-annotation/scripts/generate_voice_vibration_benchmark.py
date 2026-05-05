#!/usr/bin/env python3
"""Generate VocalGrad voice-vibration benchmark clips.

This category modulates the waveform envelope with a low-frequency oscillator
(LFO). Both the modulation depth and LFO rate change continuously over time,
yielding stronger or weaker perceived vibration while preserving a continuous
LFO phase.
"""

from __future__ import annotations

import argparse
import csv
import json
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
        default=Path("configs/voice_vibration_benchmark.json"),
        help="JSON config for voice-vibration hyperparameters.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/test/voice_vibration"),
        help="Category output directory.",
    )
    parser.add_argument(
        "--onset-csv",
        type=Path,
        default=Path(
            "data/metadata/analysis/onset_offset_min10_retuned/"
            "selected_clip_onset_times.csv"
        ),
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
        "jump_at",
        "lfo_hz_min",
        "lfo_hz_max",
        "lfo_phase_rad",
        "tiers",
        "curves",
        "modulation_profile",
    }
    missing = required_top - set(cfg)
    if missing:
        raise ValueError(f"Missing config keys: {sorted(missing)}")

    if not 0.0 < float(cfg["jump_at"]) < 1.0:
        raise ValueError("jump_at must be in (0, 1).")

    if float(cfg["lfo_hz_min"]) < 0.0:
        raise ValueError("lfo_hz_min must be >= 0.")
    if float(cfg["lfo_hz_max"]) <= float(cfg["lfo_hz_min"]):
        raise ValueError("lfo_hz_max must be > lfo_hz_min.")

    if str(cfg["modulation_profile"].get("mode", "")).strip() != "amplitude_lfo":
        raise ValueError("modulation_profile.mode must be 'amplitude_lfo'.")

    for tier_name, tier_cfg in cfg["tiers"].items():
        depth_max = float(tier_cfg["depth_max"])
        if not 0.0 < depth_max < 1.0:
            raise ValueError(f"tiers.{tier_name}.depth_max must be in (0, 1).")

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
        if (
            not bool(item["detected"])
            or item["onset_sec"] is None
            or item["offset_sec"] is None
        ):
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


def curve_progress(t: np.ndarray, curve_mode: str, jump_at: float) -> np.ndarray:
    if curve_mode == "linear":
        return t
    if curve_mode in {"quadratic_flat_first", "quad_first_flat"}:
        return t * t
    if curve_mode in {"quadratic_flat_last", "quad_last_flat"}:
        return 1.0 - (1.0 - t) * (1.0 - t)
    if curve_mode == "jump":
        return np.where(t < jump_at, 0.0, 1.0)
    raise ValueError(f"Unsupported curve mode: {curve_mode}")


def decode_pcm_normalized(path: Path) -> tuple[np.ndarray, int, int, int]:
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        samp_width = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        frames = wf.readframes(n_frames)

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


def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))


def maybe_match_rms(
    reference: np.ndarray,
    target: np.ndarray,
    preserve_rms: bool,
) -> np.ndarray:
    if not preserve_rms:
        return target

    ref_rms = rms(reference)
    target_rms = rms(target)
    if ref_rms <= 0.0 or target_rms <= 0.0:
        return target

    return target * (ref_rms / target_rms)


def build_continuous_envelope(
    num_samples: int,
    onset_sample: int,
    offset_sample: int,
    start_value: float,
    end_value: float,
    curve_mode: str,
    jump_at: float,
) -> np.ndarray:
    envelope = np.full(num_samples, start_value, dtype=np.float64)
    if num_samples == 0:
        return envelope

    if offset_sample <= onset_sample:
        envelope[:] = end_value
        return envelope

    speech_len = offset_sample - onset_sample
    if speech_len <= 1:
        envelope[onset_sample:offset_sample] = end_value
    else:
        t = np.linspace(0.0, 1.0, speech_len, endpoint=True, dtype=np.float64)
        progress = curve_progress(t, curve_mode=curve_mode, jump_at=jump_at)
        envelope[onset_sample:offset_sample] = start_value + (
            (end_value - start_value) * progress
        )

    envelope[offset_sample:] = end_value
    return envelope


def vibration_channel_in_bounds(
    channel: np.ndarray,
    framerate: int,
    depth_max: float,
    direction: str,
    curve_mode: str,
    jump_at: float,
    onset_sec: float,
    offset_sec: float,
    lfo_hz_min: float,
    lfo_hz_max: float,
    lfo_phase_rad: float,
    preserve_rms: bool,
) -> tuple[np.ndarray, int, float, float, float, float]:
    num_samples = len(channel)
    onset_sample = max(0, min(num_samples, int(round(onset_sec * framerate))))
    offset_sample = max(0, min(num_samples, int(round(offset_sec * framerate))))
    if offset_sample <= onset_sample:
        onset_sample, offset_sample = 0, num_samples

    if direction == "up":
        start_depth = 0.0
        end_depth = depth_max
        start_lfo_hz = lfo_hz_min
        end_lfo_hz = lfo_hz_max
    elif direction == "down":
        start_depth = depth_max
        end_depth = 0.0
        start_lfo_hz = lfo_hz_max
        end_lfo_hz = lfo_hz_min
    else:
        raise ValueError(f"Unsupported direction: {direction}")

    depth_env = build_continuous_envelope(
        num_samples=num_samples,
        onset_sample=onset_sample,
        offset_sample=offset_sample,
        start_value=start_depth,
        end_value=end_depth,
        curve_mode=curve_mode,
        jump_at=jump_at,
    )
    rate_env = build_continuous_envelope(
        num_samples=num_samples,
        onset_sample=onset_sample,
        offset_sample=offset_sample,
        start_value=start_lfo_hz,
        end_value=end_lfo_hz,
        curve_mode=curve_mode,
        jump_at=jump_at,
    )

    phase_step = (2.0 * np.pi * rate_env) / framerate
    phase = lfo_phase_rad + np.cumsum(
        np.concatenate([np.array([0.0], dtype=np.float64), phase_step[:-1]])
    )
    lfo = np.sin(phase)
    amplitude_env = 1.0 - (depth_env * (0.5 * (1.0 + lfo)))
    out = channel * amplitude_env.astype(np.float32)
    out = np.asarray(out, dtype=np.float32)
    out = maybe_match_rms(channel, out, preserve_rms)

    return (
        out,
        num_samples,
        float(np.min(depth_env)),
        float(np.max(depth_env)),
        float(np.min(rate_env)),
        float(np.max(rate_env)),
    )


def process_one(
    input_wav: Path,
    output_wav: Path,
    depth_max: float,
    direction: str,
    curve_mode: str,
    jump_at: float,
    onset_sec: float,
    offset_sec: float,
    lfo_hz_min: float,
    lfo_hz_max: float,
    lfo_phase_rad: float,
    preserve_rms: bool,
    overwrite: bool,
) -> dict[str, Any]:
    if output_wav.exists() and not overwrite:
        return {"status": "skipped_existing", "output": str(output_wav)}

    output_wav.parent.mkdir(parents=True, exist_ok=True)

    audio, framerate, n_channels, samp_width = decode_pcm_normalized(input_wav)

    processed_channels: list[np.ndarray] = []
    num_control_samples_list: list[int] = []
    min_depth_list: list[float] = []
    max_depth_list: list[float] = []
    min_lfo_hz_list: list[float] = []
    max_lfo_hz_list: list[float] = []

    for channel_idx in range(n_channels):
        (
            processed_channel,
            num_control_samples,
            min_depth_used,
            max_depth_used,
            min_lfo_hz_used,
            max_lfo_hz_used,
        ) = (
            vibration_channel_in_bounds(
                channel=audio[:, channel_idx],
                framerate=framerate,
                depth_max=depth_max,
                direction=direction,
                curve_mode=curve_mode,
                jump_at=jump_at,
                onset_sec=onset_sec,
                offset_sec=offset_sec,
                lfo_hz_min=lfo_hz_min,
                lfo_hz_max=lfo_hz_max,
                lfo_phase_rad=lfo_phase_rad,
                preserve_rms=preserve_rms,
            )
        )
        processed_channels.append(processed_channel)
        num_control_samples_list.append(num_control_samples)
        min_depth_list.append(min_depth_used)
        max_depth_list.append(max_depth_used)
        min_lfo_hz_list.append(min_lfo_hz_used)
        max_lfo_hz_list.append(max_lfo_hz_used)

    output_len = max(len(channel) for channel in processed_channels)
    output_audio = np.zeros((output_len, n_channels), dtype=np.float32)
    for channel_idx, channel in enumerate(processed_channels):
        output_audio[: len(channel), channel_idx] = channel

    ratio = clipped_ratio(output_audio)

    with wave.open(str(output_wav), "wb") as wav_out:
        wav_out.setnchannels(n_channels)
        wav_out.setsampwidth(samp_width)
        wav_out.setframerate(framerate)
        wav_out.writeframes(encode_pcm_normalized(output_audio, samp_width))

    return {
        "status": "written",
        "output": str(output_wav),
        "duration_sec": output_len / framerate,
        "num_control_samples": max(num_control_samples_list),
        "min_depth_used": min(min_depth_list),
        "max_depth_used": max(max_depth_list),
        "min_lfo_hz_used": min(min_lfo_hz_list),
        "max_lfo_hz_used": max(max_lfo_hz_list),
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
    lfo_hz_min = float(cfg["lfo_hz_min"])
    lfo_hz_max = float(cfg["lfo_hz_max"])
    lfo_phase_rad = float(cfg["lfo_phase_rad"])
    preserve_rms = bool(cfg["modulation_profile"].get("preserve_rms", True))

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
            depth_max = float(cfg["tiers"][tier]["depth_max"])

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
                        depth_max=depth_max,
                        direction=direction,
                        curve_mode=curve_mode,
                        jump_at=float(cfg["jump_at"]),
                        onset_sec=onset_sec,
                        offset_sec=offset_sec,
                        lfo_hz_min=lfo_hz_min,
                        lfo_hz_max=lfo_hz_max,
                        lfo_phase_rad=lfo_phase_rad,
                        preserve_rms=preserve_rms,
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
                            "category": "voice_vibration",
                            "direction": direction,
                            "curve": curve,
                            "curve_mode": curve_mode,
                            "tier": tier,
                            "lfo_hz_min": lfo_hz_min,
                            "lfo_hz_max": lfo_hz_max,
                            "lfo_phase_rad": lfo_phase_rad,
                            "depth_max": depth_max,
                            "modulation_profile_mode": cfg["modulation_profile"]["mode"],
                            "jump_at": cfg["jump_at"],
                            "onset_sec": onset_sec,
                            "offset_sec": offset_sec,
                            "status": proc["status"],
                            "duration_sec": proc.get("duration_sec", ""),
                            "num_control_samples": proc.get("num_control_samples", ""),
                            "min_depth_used": proc.get("min_depth_used", ""),
                            "max_depth_used": proc.get("max_depth_used", ""),
                            "min_lfo_hz_used": proc.get("min_lfo_hz_used", ""),
                            "max_lfo_hz_used": proc.get("max_lfo_hz_used", ""),
                            "clipped_ratio": proc.get("clipped_ratio", ""),
                        }
                    )

    manifest_path = meta_root / "voice_vibration_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    expected = len(source_rows) * len(tiers) * len(curves) * len(directions)
    clipped_values = [
        float(row["clipped_ratio"]) for row in manifest_rows if row["clipped_ratio"] != ""
    ]
    durations = [float(row["duration_sec"]) for row in manifest_rows if row["duration_sec"] != ""]

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
        "mean_output_duration_sec": float(np.mean(durations)) if durations else 0.0,
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

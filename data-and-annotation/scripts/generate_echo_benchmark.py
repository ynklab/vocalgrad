#!/usr/bin/env python3
"""Generate VocalGrad echo benchmark clips with a Schroeder reverb."""

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
        default=Path("configs/echo_benchmark.json"),
        help="JSON config for echo hyperparameters.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/test/echo"),
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

    required_top = {"jump_at", "tiers", "curves", "schroeder_reverb"}
    missing = required_top - set(cfg)
    if missing:
        raise ValueError(f"Missing config keys: {sorted(missing)}")

    if not 0.0 < float(cfg["jump_at"]) < 1.0:
        raise ValueError("jump_at must be in (0, 1).")

    for tier_name, tier_cfg in cfg["tiers"].items():
        room_scale_max = float(tier_cfg["room_scale_max"])
        wet_mix_max = float(tier_cfg["wet_mix_max"])
        if not 0.0 <= room_scale_max <= 1.0:
            raise ValueError(f"tiers.{tier_name}.room_scale_max must be in [0, 1].")
        if not 0.0 <= wet_mix_max <= 1.0:
            raise ValueError(f"tiers.{tier_name}.wet_mix_max must be in [0, 1].")

    reverb_cfg = cfg["schroeder_reverb"]
    for key in [
        "comb_delays_ms",
        "comb_feedback_base",
        "allpass_delays_ms",
        "allpass_feedback_base",
        "ir_duration_sec",
    ]:
        if key not in reverb_cfg:
            raise ValueError(f"schroeder_reverb missing key: {key}")

    if len(reverb_cfg["comb_delays_ms"]) != len(reverb_cfg["comb_feedback_base"]):
        raise ValueError("comb_delays_ms and comb_feedback_base must have the same length.")
    if len(reverb_cfg["allpass_delays_ms"]) != len(reverb_cfg["allpass_feedback_base"]):
        raise ValueError("allpass_delays_ms and allpass_feedback_base must have the same length.")

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


def ms_to_delay_samples(delay_ms: float, sr: int) -> int:
    return max(1, int(round((delay_ms / 1000.0) * sr)))


def schroeder_impulse_response(
    sr: int,
    room_scale: float,
    reverb_cfg: dict[str, Any],
) -> np.ndarray:
    ir_len = max(1, int(round(float(reverb_cfg["ir_duration_sec"]) * sr)))
    impulse = np.zeros(ir_len, dtype=np.float32)
    impulse[0] = 1.0

    comb_delays = [
        ms_to_delay_samples(delay_ms, sr) for delay_ms in reverb_cfg["comb_delays_ms"]
    ]
    comb_buffers = [np.zeros(delay, dtype=np.float32) for delay in comb_delays]
    comb_pos = [0 for _ in comb_delays]

    allpass_delays = [
        ms_to_delay_samples(delay_ms, sr) for delay_ms in reverb_cfg["allpass_delays_ms"]
    ]
    allpass_buffers = [np.zeros(delay, dtype=np.float32) for delay in allpass_delays]
    allpass_pos = [0 for _ in allpass_delays]

    output = np.zeros_like(impulse)
    for sample_idx, sample in enumerate(impulse):
        comb_sum = 0.0
        for idx, base_feedback in enumerate(reverb_cfg["comb_feedback_base"]):
            buf = comb_buffers[idx]
            pos = comb_pos[idx]
            feedback = float(base_feedback) * room_scale
            delayed = float(buf[pos])
            comb_out = float(sample) + (feedback * delayed)
            buf[pos] = comb_out
            comb_pos[idx] = (pos + 1) % len(buf)
            comb_sum += comb_out

        stage = comb_sum / len(reverb_cfg["comb_feedback_base"])
        for idx, base_feedback in enumerate(reverb_cfg["allpass_feedback_base"]):
            g = float(base_feedback) * room_scale
            if abs(g) < 1e-8:
                continue
            buf = allpass_buffers[idx]
            pos = allpass_pos[idx]
            delayed = float(buf[pos])
            out = (-g * stage) + delayed
            buf[pos] = stage + (g * out)
            allpass_pos[idx] = (pos + 1) % len(buf)
            stage = out

        output[sample_idx] = stage

    return output


def fft_convolve_same(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    full_len = len(x) + len(h) - 1
    fft_len = 1 << (full_len - 1).bit_length()
    x_fft = np.fft.rfft(x, n=fft_len)
    h_fft = np.fft.rfft(h, n=fft_len)
    out = np.fft.irfft(x_fft * h_fft, n=fft_len)[:full_len]
    return np.asarray(out[: len(x)], dtype=np.float32)


def build_wet_mix_envelope(
    num_samples: int,
    sr: int,
    wet_mix_max: float,
    direction: str,
    curve_mode: str,
    jump_at: float,
    onset_sec: float,
    offset_sec: float,
) -> np.ndarray:
    onset_sample = max(0, min(num_samples, int(round(onset_sec * sr))))
    offset_sample = max(0, min(num_samples, int(round(offset_sec * sr))))
    if offset_sample <= onset_sample:
        onset_sample, offset_sample = 0, num_samples

    if direction == "up":
        start_wet_mix = 0.0
        end_wet_mix = wet_mix_max
    elif direction == "down":
        start_wet_mix = wet_mix_max
        end_wet_mix = 0.0
    else:
        raise ValueError(f"Unsupported direction: {direction}")

    return build_continuous_envelope(
        num_samples=num_samples,
        onset_sample=onset_sample,
        offset_sample=offset_sample,
        start_value=start_wet_mix,
        end_value=end_wet_mix,
        curve_mode=curve_mode,
        jump_at=jump_at,
    )


def apply_reverb_channel(
    channel: np.ndarray,
    sr: int,
    room_scale_max: float,
    reverb_cfg: dict[str, Any],
    ir_cache: dict[tuple[int, float], np.ndarray],
) -> np.ndarray:
    cache_key = (sr, round(room_scale_max, 6))
    if cache_key not in ir_cache:
        ir_cache[cache_key] = schroeder_impulse_response(sr, room_scale_max, reverb_cfg)
    ir = ir_cache[cache_key]

    reverbed = fft_convolve_same(channel, ir)
    reverbed = maybe_match_rms(channel, reverbed, bool(reverb_cfg.get("preserve_rms", True)))
    return np.asarray(reverbed, dtype=np.float32)


def precompute_reverbed_variants(
    audio: np.ndarray,
    framerate: int,
    tiers_cfg: dict[str, dict[str, Any]],
    reverb_cfg: dict[str, Any],
    ir_cache: dict[tuple[int, float], np.ndarray],
) -> dict[str, np.ndarray]:
    reverbed_by_tier: dict[str, np.ndarray] = {}
    n_channels = audio.shape[1]

    for tier_name, tier_cfg in tiers_cfg.items():
        room_scale_max = float(tier_cfg["room_scale_max"])
        tier_channels: list[np.ndarray] = []
        for channel_idx in range(n_channels):
            tier_channels.append(
                apply_reverb_channel(
                    channel=audio[:, channel_idx],
                    sr=framerate,
                    room_scale_max=room_scale_max,
                    reverb_cfg=reverb_cfg,
                    ir_cache=ir_cache,
                )
            )
        reverbed_by_tier[tier_name] = np.stack(tier_channels, axis=1)

    return reverbed_by_tier


def process_one(
    audio: np.ndarray,
    reverbed_audio: np.ndarray,
    framerate: int,
    samp_width: int,
    output_wav: Path,
    room_scale_max: float,
    wet_mix_max: float,
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

    n_channels = audio.shape[1]
    wet_mix_env = build_wet_mix_envelope(
        num_samples=len(audio),
        sr=framerate,
        wet_mix_max=wet_mix_max,
        direction=direction,
        curve_mode=curve_mode,
        jump_at=jump_at,
        onset_sec=onset_sec,
        offset_sec=offset_sec,
    ).reshape(-1, 1)
    output_audio = ((1.0 - wet_mix_env) * audio) + (wet_mix_env * reverbed_audio)
    output_audio = np.asarray(output_audio, dtype=np.float32)

    ratio = clipped_ratio(output_audio)

    with wave.open(str(output_wav), "wb") as wav_out:
        wav_out.setnchannels(n_channels)
        wav_out.setsampwidth(samp_width)
        wav_out.setframerate(framerate)
        wav_out.writeframes(encode_pcm_normalized(output_audio, samp_width))

    return {
        "status": "written",
        "output": str(output_wav),
        "duration_sec": len(output_audio) / framerate,
        "min_room_scale_used": room_scale_max,
        "max_room_scale_used": room_scale_max,
        "min_wet_mix_used": float(np.min(wet_mix_env)),
        "max_wet_mix_used": float(np.max(wet_mix_env)),
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
    reverb_cfg = dict(cfg["schroeder_reverb"])
    ir_cache: dict[tuple[int, float], np.ndarray] = {}

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

        audio, framerate, _n_channels, samp_width = decode_pcm_normalized(source_wav)
        reverbed_by_tier = precompute_reverbed_variants(
            audio=audio,
            framerate=framerate,
            tiers_cfg=cfg["tiers"],
            reverb_cfg=reverb_cfg,
            ir_cache=ir_cache,
        )

        for tier in tiers:
            room_scale_max = float(cfg["tiers"][tier]["room_scale_max"])
            wet_mix_max = float(cfg["tiers"][tier]["wet_mix_max"])
            reverbed_audio = reverbed_by_tier[tier]

            for curve in curves:
                curve_mode = str(cfg["curves"][curve]["mode"])

                for direction in directions:
                    out_name = (
                        f"{source_item_id}__{source_clip_id}__{curve}__"
                        f"{tier}__{direction}.wav"
                    )
                    out_wav = audio_root / direction / out_name

                    proc = process_one(
                        audio=audio,
                        reverbed_audio=reverbed_audio,
                        framerate=framerate,
                        samp_width=samp_width,
                        output_wav=out_wav,
                        room_scale_max=room_scale_max,
                        wet_mix_max=wet_mix_max,
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
                            "category": "echo",
                            "direction": direction,
                            "curve": curve,
                            "curve_mode": curve_mode,
                            "tier": tier,
                            "room_scale_max": room_scale_max,
                            "wet_mix_max": wet_mix_max,
                            "comb_delays_ms": ",".join(
                                str(v) for v in reverb_cfg["comb_delays_ms"]
                            ),
                            "comb_feedback_base": ",".join(
                                str(v) for v in reverb_cfg["comb_feedback_base"]
                            ),
                            "allpass_delays_ms": ",".join(
                                str(v) for v in reverb_cfg["allpass_delays_ms"]
                            ),
                            "allpass_feedback_base": ",".join(
                                str(v) for v in reverb_cfg["allpass_feedback_base"]
                            ),
                            "ir_duration_sec": reverb_cfg["ir_duration_sec"],
                            "jump_at": cfg["jump_at"],
                            "onset_sec": onset_sec,
                            "offset_sec": offset_sec,
                            "status": proc["status"],
                            "duration_sec": proc.get("duration_sec", ""),
                            "min_room_scale_used": proc.get("min_room_scale_used", ""),
                            "max_room_scale_used": proc.get("max_room_scale_used", ""),
                            "min_wet_mix_used": proc.get("min_wet_mix_used", ""),
                            "max_wet_mix_used": proc.get("max_wet_mix_used", ""),
                            "clipped_ratio": proc.get("clipped_ratio", ""),
                        }
                    )

    manifest_path = meta_root / "echo_manifest.csv"
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

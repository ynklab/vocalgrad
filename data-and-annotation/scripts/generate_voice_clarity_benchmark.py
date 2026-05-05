#!/usr/bin/env python3
"""Generate VocalGrad voice-clarity benchmark clips.

This category applies an STFT-domain low-pass filter sweep to vary perceived
voice clarity over time. The speech span is divided into short chunks, each
chunk is processed with extra left/right context, and only the center chunk is
retained to reduce audible discontinuities.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import wave
from pathlib import Path
from typing import Any

import librosa
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
        default=Path("configs/voice_clarity_benchmark.json"),
        help="JSON config for voice-clarity hyperparameters.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/test/voice_clarity"),
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
        "step_duration_sec",
        "context_duration_sec",
        "crossfade_duration_sec",
        "jump_at",
        "tiers",
        "curves",
        "stft_kwargs",
        "filter_profile",
    }
    missing = required_top - set(cfg)
    if missing:
        raise ValueError(f"Missing config keys: {sorted(missing)}")

    if not 0.0 < float(cfg["jump_at"]) < 1.0:
        raise ValueError("jump_at must be in (0, 1).")

    if str(cfg["filter_profile"].get("mode", "")).strip() != "stft_lowpass":
        raise ValueError("filter_profile.mode must be 'stft_lowpass'.")

    transition_hz = float(cfg["filter_profile"].get("transition_hz", 0.0))
    if transition_hz < 0.0:
        raise ValueError("filter_profile.transition_hz must be >= 0.")

    for tier_name, tier_cfg in cfg["tiers"].items():
        cutoff_min_hz = float(tier_cfg["cutoff_min_hz"])
        if cutoff_min_hz <= 0.0:
            raise ValueError(f"tiers.{tier_name}.cutoff_min_hz must be > 0.")

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


def append_with_crossfade(
    acc: np.ndarray,
    segment: np.ndarray,
    crossfade_samples: int,
) -> np.ndarray:
    if acc.size == 0:
        return segment.copy()
    if segment.size == 0:
        return acc

    fade = min(crossfade_samples, len(acc), len(segment))
    if fade <= 0:
        return np.concatenate([acc, segment])

    fade_out = np.linspace(1.0, 0.0, fade, endpoint=True, dtype=np.float32)
    fade_in = 1.0 - fade_out
    overlap = (acc[-fade:] * fade_out) + (segment[:fade] * fade_in)
    return np.concatenate([acc[:-fade], overlap, segment[fade:]])


def stft_kwargs_for_chunk(chunk: np.ndarray, base_kwargs: dict[str, Any]) -> dict[str, Any]:
    kwargs = dict(base_kwargs)
    chunk_len = len(chunk)
    if chunk_len <= 0:
        return kwargs

    n_fft = int(kwargs.get("n_fft", 2048))
    if n_fft > chunk_len:
        kwargs["n_fft"] = max(32, chunk_len)

    win_length = int(kwargs.get("win_length", kwargs["n_fft"]))
    if win_length > int(kwargs["n_fft"]):
        win_length = int(kwargs["n_fft"])
    if win_length > chunk_len:
        win_length = max(32, min(chunk_len, int(kwargs["n_fft"])))
    kwargs["win_length"] = win_length

    hop_length = kwargs.get("hop_length")
    if hop_length is not None and int(hop_length) >= int(kwargs["win_length"]):
        kwargs["hop_length"] = max(1, int(kwargs["win_length"]) // 4)

    return kwargs


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


def lowpass_mask(
    freqs_hz: np.ndarray,
    cutoff_hz: float,
    transition_hz: float,
) -> np.ndarray:
    if transition_hz <= 0.0:
        return (freqs_hz <= cutoff_hz).astype(np.float32)

    lower = max(0.0, cutoff_hz - (transition_hz / 2.0))
    upper = cutoff_hz + (transition_hz / 2.0)

    mask = np.ones_like(freqs_hz, dtype=np.float32)
    mask[freqs_hz >= upper] = 0.0

    band = (freqs_hz > lower) & (freqs_hz < upper)
    if np.any(band):
        t = (freqs_hz[band] - lower) / (upper - lower)
        mask[band] = 0.5 * (1.0 + np.cos(np.pi * t))

    return mask


def lowpass_segment(
    segment: np.ndarray,
    framerate: int,
    cutoff_hz: float,
    stft_kwargs: dict[str, Any],
    transition_hz: float,
    preserve_rms: bool,
) -> np.ndarray:
    kwargs = stft_kwargs_for_chunk(segment, stft_kwargs)
    stft = librosa.stft(y=segment, **kwargs)
    freqs_hz = librosa.fft_frequencies(sr=framerate, n_fft=int(kwargs["n_fft"]))
    mask = lowpass_mask(freqs_hz, cutoff_hz=cutoff_hz, transition_hz=transition_hz)
    filtered_stft = stft * mask[:, None]
    filtered = librosa.istft(filtered_stft, length=len(segment), **kwargs)
    filtered = np.asarray(filtered, dtype=np.float32)
    return maybe_match_rms(segment, filtered, preserve_rms)


def clarity_channel_in_bounds(
    channel: np.ndarray,
    framerate: int,
    step_duration: float,
    context_duration_sec: float,
    cutoff_min_hz: float,
    direction: str,
    curve_mode: str,
    jump_at: float,
    onset_sec: float,
    offset_sec: float,
    crossfade_duration_sec: float,
    stft_kwargs: dict[str, Any],
    transition_hz: float,
    preserve_rms: bool,
) -> tuple[np.ndarray, int, float, float]:
    num_samples = len(channel)
    onset_sample = max(0, min(num_samples, int(round(onset_sec * framerate))))
    offset_sample = max(0, min(num_samples, int(round(offset_sec * framerate))))
    if offset_sample <= onset_sample:
        onset_sample, offset_sample = 0, num_samples

    pre = channel[:onset_sample]
    speech = channel[onset_sample:offset_sample]
    post = channel[offset_sample:]

    samples_per_step = max(1, int(round(framerate * step_duration)))
    context_samples = max(0, int(round(framerate * context_duration_sec)))
    crossfade_samples = max(0, int(round(framerate * crossfade_duration_sec)))
    cutoff_max_hz = framerate / 2.0

    if direction == "up":
        start_cutoff_hz = cutoff_min_hz
        end_cutoff_hz = cutoff_max_hz
    elif direction == "down":
        start_cutoff_hz = cutoff_max_hz
        end_cutoff_hz = cutoff_min_hz
    else:
        raise ValueError(f"Unsupported direction: {direction}")

    processed_pre = lowpass_segment(
        segment=pre,
        framerate=framerate,
        cutoff_hz=start_cutoff_hz,
        stft_kwargs=stft_kwargs,
        transition_hz=transition_hz,
        preserve_rms=preserve_rms,
    )
    processed_post = lowpass_segment(
        segment=post,
        framerate=framerate,
        cutoff_hz=end_cutoff_hz,
        stft_kwargs=stft_kwargs,
        transition_hz=transition_hz,
        preserve_rms=preserve_rms,
    )

    num_steps = max(1, math.ceil(len(speech) / samples_per_step))
    clarified_speech = np.array([], dtype=np.float32)
    applied_cutoffs: list[float] = [float(start_cutoff_hz)]

    for step_idx in range(num_steps):
        start = step_idx * samples_per_step
        end = min((step_idx + 1) * samples_per_step, len(speech))
        chunk = speech[start:end]
        if chunk.size == 0:
            continue

        context_start = max(0, start - context_samples)
        context_end = min(len(speech), end + context_samples)
        context_chunk = speech[context_start:context_end]
        trim_start = start - context_start
        trim_end = trim_start + (end - start)

        p = progress_value(step_idx, num_steps, curve_mode, jump_at)
        cutoff_hz = start_cutoff_hz + ((end_cutoff_hz - start_cutoff_hz) * p)
        filtered_context = lowpass_segment(
            segment=context_chunk,
            framerate=framerate,
            cutoff_hz=float(cutoff_hz),
            stft_kwargs=stft_kwargs,
            transition_hz=transition_hz,
            preserve_rms=preserve_rms,
        )
        applied_cutoffs.append(float(cutoff_hz))
        filtered_chunk = filtered_context[trim_start:trim_end]
        clarified_speech = append_with_crossfade(
            clarified_speech,
            filtered_chunk,
            crossfade_samples=crossfade_samples,
        )

    applied_cutoffs.append(float(end_cutoff_hz))

    out = np.concatenate([processed_pre, clarified_speech, processed_post])
    min_cutoff_used = min(applied_cutoffs) if applied_cutoffs else cutoff_max_hz
    max_cutoff_used = max(applied_cutoffs) if applied_cutoffs else cutoff_max_hz
    return out, num_steps, min_cutoff_used, max_cutoff_used


def process_one(
    input_wav: Path,
    output_wav: Path,
    step_duration: float,
    context_duration_sec: float,
    cutoff_min_hz: float,
    direction: str,
    curve_mode: str,
    jump_at: float,
    onset_sec: float,
    offset_sec: float,
    crossfade_duration_sec: float,
    stft_kwargs: dict[str, Any],
    transition_hz: float,
    preserve_rms: bool,
    overwrite: bool,
) -> dict[str, Any]:
    if output_wav.exists() and not overwrite:
        return {"status": "skipped_existing", "output": str(output_wav)}

    output_wav.parent.mkdir(parents=True, exist_ok=True)

    audio, framerate, n_channels, samp_width = decode_pcm_normalized(input_wav)

    processed_channels: list[np.ndarray] = []
    num_steps_list: list[int] = []
    min_cutoff_list: list[float] = []
    max_cutoff_list: list[float] = []

    for channel_idx in range(n_channels):
        processed_channel, num_steps, min_cutoff_used, max_cutoff_used = (
            clarity_channel_in_bounds(
                channel=audio[:, channel_idx],
                framerate=framerate,
                step_duration=step_duration,
                context_duration_sec=context_duration_sec,
                cutoff_min_hz=cutoff_min_hz,
                direction=direction,
                curve_mode=curve_mode,
                jump_at=jump_at,
                onset_sec=onset_sec,
                offset_sec=offset_sec,
                crossfade_duration_sec=crossfade_duration_sec,
                stft_kwargs=stft_kwargs,
                transition_hz=transition_hz,
                preserve_rms=preserve_rms,
            )
        )
        processed_channels.append(processed_channel)
        num_steps_list.append(num_steps)
        min_cutoff_list.append(min_cutoff_used)
        max_cutoff_list.append(max_cutoff_used)

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
        "num_steps": max(num_steps_list),
        "min_cutoff_used_hz": min(min_cutoff_list),
        "max_cutoff_used_hz": max(max_cutoff_list),
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
    stft_kwargs = dict(cfg["stft_kwargs"])
    transition_hz = float(cfg["filter_profile"].get("transition_hz", 0.0))
    preserve_rms = bool(cfg["filter_profile"].get("preserve_rms", True))

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

        with wave.open(str(source_wav), "rb") as wf:
            source_sr = wf.getframerate()

        for tier in tiers:
            cutoff_min_hz = float(cfg["tiers"][tier]["cutoff_min_hz"])

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
                        context_duration_sec=float(cfg["context_duration_sec"]),
                        cutoff_min_hz=cutoff_min_hz,
                        direction=direction,
                        curve_mode=curve_mode,
                        jump_at=float(cfg["jump_at"]),
                        onset_sec=onset_sec,
                        offset_sec=offset_sec,
                        crossfade_duration_sec=float(cfg["crossfade_duration_sec"]),
                        stft_kwargs=stft_kwargs,
                        transition_hz=transition_hz,
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
                            "category": "voice_clarity",
                            "direction": direction,
                            "curve": curve,
                            "curve_mode": curve_mode,
                            "tier": tier,
                            "step_duration_sec": cfg["step_duration_sec"],
                            "context_duration_sec": cfg["context_duration_sec"],
                            "crossfade_duration_sec": cfg["crossfade_duration_sec"],
                            "source_sample_rate": source_sr,
                            "cutoff_min_hz": cutoff_min_hz,
                            "cutoff_max_hz": source_sr / 2.0,
                            "transition_hz": transition_hz,
                            "filter_profile_mode": cfg["filter_profile"]["mode"],
                            "jump_at": cfg["jump_at"],
                            "onset_sec": onset_sec,
                            "offset_sec": offset_sec,
                            "status": proc["status"],
                            "duration_sec": proc.get("duration_sec", ""),
                            "num_steps": proc.get("num_steps", ""),
                            "min_cutoff_used_hz": proc.get("min_cutoff_used_hz", ""),
                            "max_cutoff_used_hz": proc.get("max_cutoff_used_hz", ""),
                            "clipped_ratio": proc.get("clipped_ratio", ""),
                        }
                    )

    manifest_path = meta_root / "voice_clarity_manifest.csv"
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

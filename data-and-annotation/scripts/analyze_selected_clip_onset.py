#!/usr/bin/env python3
"""Estimate first-speech onset times for selected source clips."""

from __future__ import annotations

import argparse
import csv
import json
import wave
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=Path("data/metadata/selection/test_source_clips_100.csv"),
        help="CSV containing selected source clips.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/metadata/analysis/onset_offset_min10_retuned"),
        help="Directory to write analysis artifacts.",
    )
    parser.add_argument(
        "--selection-output-csv",
        type=Path,
        default=None,
        help=(
            "Optional path to write source CSV with onset/offset columns merged. "
            "Default: overwrite --source-csv."
        ),
    )
    parser.add_argument(
        "--skip-selection-merge",
        action="store_true",
        help="If set, do not merge onset/offset columns into selection CSV.",
    )
    parser.add_argument(
        "--frame-ms",
        type=float,
        default=10.0,
        help="Frame length for RMS in milliseconds.",
    )
    parser.add_argument(
        "--hop-ms",
        type=float,
        default=10.0,
        help="Hop length for RMS in milliseconds.",
    )
    parser.add_argument(
        "--noise-ref-ms",
        type=float,
        default=200.0,
        help="Initial region length used to estimate noise floor.",
    )
    parser.add_argument(
        "--noise-multiplier",
        type=float,
        default=1.8,
        help="Threshold multiplier against noise floor RMS.",
    )
    parser.add_argument(
        "--peak-fraction",
        type=float,
        default=0.003,
        help="Threshold floor as a fraction of clip peak RMS.",
    )
    parser.add_argument(
        "--abs-threshold",
        type=float,
        default=5e-5,
        help="Absolute threshold floor on normalized waveform RMS.",
    )
    parser.add_argument(
        "--min-consecutive-frames",
        type=int,
        default=10,
        help="Minimum consecutive frames over threshold to declare onset.",
    )
    parser.add_argument(
        "--pre-emphasis-alpha",
        type=float,
        default=0.97,
        help="If > 0, apply pre-emphasis: y[t] = x[t] - alpha * x[t-1].",
    )
    return parser.parse_args()


def decode_normalized_audio(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        samp_width = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        frames = wf.readframes(n_frames)

    if samp_width == 1:
        x = np.frombuffer(frames, dtype=np.uint8).astype(np.float64)
        x = (x - 128.0) / 128.0
    elif samp_width == 2:
        x = np.frombuffer(frames, dtype=np.int16).astype(np.float64)
        x = x / 32768.0
    else:
        raise ValueError(f"Unsupported sample width: {samp_width} ({path})")

    if n_channels > 1:
        x = x.reshape(-1, n_channels).mean(axis=1)

    return x, framerate


def compute_rms_frames(x: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    if len(x) == 0:
        return np.array([], dtype=np.float64)

    if len(x) < frame_len:
        pad = np.zeros(frame_len - len(x), dtype=np.float64)
        x = np.concatenate([x, pad])

    rms_vals: list[float] = []
    for start in range(0, len(x) - frame_len + 1, hop_len):
        frame = x[start : start + frame_len]
        rms_vals.append(float(np.sqrt(np.mean(frame * frame))))

    return np.array(rms_vals, dtype=np.float64)


def apply_pre_emphasis(x: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0.0:
        return x
    y = x.copy()
    y[1:] = x[1:] - (alpha * x[:-1])
    return y


def find_active_segments(active: np.ndarray, min_consecutive_frames: int) -> list[tuple[int, int]]:
    """Return active frame segments as [start_idx, end_idx_exclusive)."""
    segments: list[tuple[int, int]] = []
    i = 0
    n = len(active)
    while i < n:
        if not bool(active[i]):
            i += 1
            continue

        j = i + 1
        while j < n and bool(active[j]):
            j += 1

        if (j - i) >= min_consecutive_frames:
            segments.append((i, j))

        i = j
    return segments


def detect_speech_bounds(
    x: np.ndarray,
    sr: int,
    frame_ms: float,
    hop_ms: float,
    noise_ref_ms: float,
    noise_multiplier: float,
    peak_fraction: float,
    abs_threshold: float,
    min_consecutive_frames: int,
) -> dict[str, Any]:
    frame_len = max(1, int(round(sr * frame_ms / 1000.0)))
    hop_len = max(1, int(round(sr * hop_ms / 1000.0)))
    rms = compute_rms_frames(x, frame_len, hop_len)

    if len(rms) == 0:
        return {
            "detected": False,
            "onset_sec": None,
            "offset_sec": None,
            "speech_duration_sec": None,
            "num_segments": 0,
            "threshold": None,
            "noise_floor": None,
            "peak_rms": None,
        }

    head_samples = max(1, int(round(sr * noise_ref_ms / 1000.0)))
    head_frame_limit = int(np.ceil(max(1, head_samples - frame_len + 1) / hop_len))
    head_frames = max(1, min(len(rms), head_frame_limit))

    noise_floor = float(np.median(rms[:head_frames]))
    peak_rms = float(np.max(rms))
    threshold = max(noise_floor * noise_multiplier, peak_rms * peak_fraction, abs_threshold)
    active = rms >= threshold
    segments = find_active_segments(active, min_consecutive_frames=min_consecutive_frames)
    if not segments:
        return {
            "detected": False,
            "onset_sec": None,
            "offset_sec": None,
            "speech_duration_sec": None,
            "num_segments": 0,
            "threshold": threshold,
            "noise_floor": noise_floor,
            "peak_rms": peak_rms,
        }

    onset_frame = segments[0][0]
    last_segment_end_exclusive = segments[-1][1]
    onset_sample = onset_frame * hop_len
    offset_sample = min(len(x), (last_segment_end_exclusive - 1) * hop_len + frame_len)
    return {
        "detected": True,
        "onset_sec": onset_sample / sr,
        "offset_sec": offset_sample / sr,
        "speech_duration_sec": max(0.0, (offset_sample - onset_sample) / sr),
        "num_segments": len(segments),
        "threshold": threshold,
        "noise_floor": noise_floor,
        "peak_rms": peak_rms,
    }


def detect_onset_sec(
    x: np.ndarray,
    sr: int,
    frame_ms: float,
    hop_ms: float,
    noise_ref_ms: float,
    noise_multiplier: float,
    peak_fraction: float,
    abs_threshold: float,
    min_consecutive_frames: int,
) -> dict[str, Any]:
    """Backward-compatible alias."""
    return detect_speech_bounds(
        x=x,
        sr=sr,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        noise_ref_ms=noise_ref_ms,
        noise_multiplier=noise_multiplier,
        peak_fraction=peak_fraction,
        abs_threshold=abs_threshold,
        min_consecutive_frames=min_consecutive_frames,
    )


def plot_hist(values: list[float], path: Path, title: str, xlabel: str) -> None:
    plt.figure(figsize=(9, 5))
    plt.hist(values, bins=25, color="#2E6F95", edgecolor="black", linewidth=0.4)
    plt.xlabel(xlabel)
    plt.ylabel("Number of clips")
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()


def merge_onset_into_selection_rows(
    selection_rows: list[dict[str, str]],
    onset_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    by_test_item_id: dict[str, dict[str, Any]] = {}
    by_clip_id: dict[str, dict[str, Any]] = {}
    for row in onset_rows:
        test_item_id = str(row.get("test_item_id", ""))
        clip_id = str(row.get("clip_id", ""))
        if test_item_id:
            by_test_item_id[test_item_id] = row
        if clip_id and clip_id not in by_clip_id:
            by_clip_id[clip_id] = row

    merged: list[dict[str, str]] = []
    for src in selection_rows:
        match = by_test_item_id.get(src.get("test_item_id", ""))
        if match is None:
            match = by_clip_id.get(src.get("clip_id", ""))

        row = dict(src)
        if match is None:
            row["onset_detected"] = "False"
            row["onset_sec"] = ""
            row["offset_sec"] = ""
            row["speech_duration_sec"] = ""
            row["onset_num_segments"] = ""
            row["onset_threshold"] = ""
            row["onset_noise_floor"] = ""
            row["onset_peak_rms"] = ""
        else:
            row["onset_detected"] = str(match["detected"])
            row["onset_sec"] = str(match["onset_sec"])
            row["offset_sec"] = str(match["offset_sec"])
            row["speech_duration_sec"] = str(match["speech_duration_sec"])
            row["onset_num_segments"] = str(match["num_segments"])
            row["onset_threshold"] = str(match["threshold"])
            row["onset_noise_floor"] = str(match["noise_floor"])
            row["onset_peak_rms"] = str(match["peak_rms"])
        merged.append(row)

    return merged


def write_selection_with_onset(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("No rows to write for merged selection CSV.")

    base_fields = list(rows[0].keys())
    ordered_fields = [
        field
        for field in base_fields
        if field
        not in {
            "onset_detected",
            "onset_sec",
            "offset_sec",
            "speech_duration_sec",
            "onset_num_segments",
            "onset_threshold",
            "onset_noise_floor",
            "onset_peak_rms",
        }
    ]
    ordered_fields.extend(
        [
            "onset_detected",
            "onset_sec",
            "offset_sec",
            "speech_duration_sec",
            "onset_num_segments",
            "onset_threshold",
            "onset_noise_floor",
            "onset_peak_rms",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.source_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    out_rows: list[dict[str, Any]] = []
    detected_onsets: list[float] = []
    detected_offsets: list[float] = []
    detected_speech_durations: list[float] = []

    for row in rows:
        wav_path = Path(row["audio_path"])
        x, sr = decode_normalized_audio(wav_path)
        x = apply_pre_emphasis(x, alpha=args.pre_emphasis_alpha)
        res = detect_speech_bounds(
            x=x,
            sr=sr,
            frame_ms=args.frame_ms,
            hop_ms=args.hop_ms,
            noise_ref_ms=args.noise_ref_ms,
            noise_multiplier=args.noise_multiplier,
            peak_fraction=args.peak_fraction,
            abs_threshold=args.abs_threshold,
            min_consecutive_frames=args.min_consecutive_frames,
        )

        onset = res["onset_sec"]
        offset = res["offset_sec"]
        speech_duration = res["speech_duration_sec"]
        if onset is not None:
            detected_onsets.append(float(onset))
        if offset is not None:
            detected_offsets.append(float(offset))
        if speech_duration is not None:
            detected_speech_durations.append(float(speech_duration))

        out_rows.append(
            {
                "test_item_id": row.get("test_item_id", ""),
                "clip_id": row.get("clip_id", ""),
                "speaker_id": row.get("speaker_id", ""),
                "sentence_id": row.get("sentence_id", ""),
                "audio_path": row.get("audio_path", ""),
                "detected": res["detected"],
                "onset_sec": "" if onset is None else f"{float(onset):.6f}",
                "offset_sec": "" if offset is None else f"{float(offset):.6f}",
                "speech_duration_sec": (
                    "" if speech_duration is None else f"{float(speech_duration):.6f}"
                ),
                "num_segments": res["num_segments"],
                "threshold": "" if res["threshold"] is None else f"{float(res['threshold']):.8f}",
                "noise_floor": (
                    "" if res["noise_floor"] is None else f"{float(res['noise_floor']):.8f}"
                ),
                "peak_rms": "" if res["peak_rms"] is None else f"{float(res['peak_rms']):.8f}",
            }
        )

    csv_path = args.output_dir / "selected_clip_onset_times.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    selection_output_csv: Path | None = None
    if not args.skip_selection_merge:
        selection_output_csv = args.selection_output_csv or args.source_csv
        merged_rows = merge_onset_into_selection_rows(
            selection_rows=rows,
            onset_rows=out_rows,
        )
        write_selection_with_onset(selection_output_csv, merged_rows)

    summary = {
        "source_csv": str(args.source_csv),
        "num_rows": len(out_rows),
        "num_detected": len(detected_onsets),
        "num_undetected": len(out_rows) - len(detected_onsets),
        "mean_onset_sec": float(np.mean(detected_onsets)) if detected_onsets else None,
        "median_onset_sec": float(np.median(detected_onsets)) if detected_onsets else None,
        "min_onset_sec": float(np.min(detected_onsets)) if detected_onsets else None,
        "max_onset_sec": float(np.max(detected_onsets)) if detected_onsets else None,
        "mean_offset_sec": float(np.mean(detected_offsets)) if detected_offsets else None,
        "median_offset_sec": float(np.median(detected_offsets)) if detected_offsets else None,
        "min_offset_sec": float(np.min(detected_offsets)) if detected_offsets else None,
        "max_offset_sec": float(np.max(detected_offsets)) if detected_offsets else None,
        "mean_speech_duration_sec": (
            float(np.mean(detected_speech_durations)) if detected_speech_durations else None
        ),
        "median_speech_duration_sec": (
            float(np.median(detected_speech_durations)) if detected_speech_durations else None
        ),
        "params": {
            "frame_ms": args.frame_ms,
            "hop_ms": args.hop_ms,
            "noise_ref_ms": args.noise_ref_ms,
            "noise_multiplier": args.noise_multiplier,
            "peak_fraction": args.peak_fraction,
            "abs_threshold": args.abs_threshold,
            "min_consecutive_frames": args.min_consecutive_frames,
            "pre_emphasis_alpha": args.pre_emphasis_alpha,
        },
        "output_csv": str(csv_path),
        "selection_output_csv": (
            str(selection_output_csv) if selection_output_csv is not None else None
        ),
        "output_hist_png": str(args.output_dir / "selected_clip_onset_histogram.png"),
        "output_offset_hist_png": str(args.output_dir / "selected_clip_offset_histogram.png"),
    }

    summary_path = args.output_dir / "selected_clip_onset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if detected_onsets:
        plot_hist(
            detected_onsets,
            args.output_dir / "selected_clip_onset_histogram.png",
            title="Selected Clip First-Speech Onset Distribution",
            xlabel="Onset time from clip start (sec)",
        )
    if detected_offsets:
        plot_hist(
            detected_offsets,
            args.output_dir / "selected_clip_offset_histogram.png",
            title="Selected Clip Speech-End (Offset) Distribution",
            xlabel="Offset time from clip start (sec)",
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot human and model VocalGrad accuracies with bootstrap CIs in a 3x3 grid."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np

default_mpl_config_dir = Path(os.environ.get("TMPDIR", "/tmp")) / f"plic-matplotlib-{os.getuid()}"
default_mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(default_mpl_config_dir))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_MODEL_ORDER = [
    "gemini-3-flash",
    "kimi-audio",
    "mimo-audio",
    "step-audio-2-mini",
    "audioflamingo3",
]

DEFAULT_CATEGORY_ORDER = [
    "speaking_speed",
    "voice_pitch",
    "volume",
    "audio_distortion",
    "audio_roughness",
    "background_noise",
    "echo",
    "voice_clarity",
    "voice_vibration",
]

CATEGORY_TITLES = {
    "speaking_speed": "Speaking Speed",
    "voice_pitch": "Voice Pitch",
    "volume": "Volume",
    "audio_distortion": "Audio Distortion",
    "audio_roughness": "Audio Roughness",
    "background_noise": "Background Noise",
    "echo": "Echo",
    "voice_clarity": "Voice Clarity",
    "voice_vibration": "Voice Vibration",
}

MODEL_DISPLAY_NAMES = {
    "human": "Human",
    "audioflamingo3": "AudioFlamingo3",
    "gemini-3-flash": "Gemini 3 Flash",
    "kimi-audio": "Kimi-Audio",
    "mimo-audio": "MiMo-Audio",
    "step-audio-2-mini": "Step-Audio-2 Mini",
}



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotation-root",
        type=Path,
        default=Path("outputs/annotation_data"),
    )
    parser.add_argument(
        "--model-raw-root",
        action="append",
        type=Path,
        default=[
            Path("outputs/raw/vocalgrad/default"),
            Path("outputs/raw/vocalgrad/audio-ref"),
        ],
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/analysis/vocalgrad/default/human_model_accuracy_ci_grid.pdf"),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("outputs/analysis/vocalgrad/default/human_model_accuracy_ci_grid.csv"),
    )
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc


def normalize_label(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return {
        "up": "increase",
        "increase": "increase",
        "increased": "increase",
        "down": "decrease",
        "decrease": "decrease",
        "decreased": "decrease",
    }.get(text, text)


from evaluation_common import parse_direction_label, prediction_for_row


def find_raw_file(raw_roots: list[Path], category: str, model_name: str) -> Path | None:
    for root in raw_roots:
        for suffix in (".jsonl", ".json"):
            path = root / category / f"{model_name}{suffix}"
            if path.exists():
                return path
    return None


def human_correctness(annotation_root: Path, category: str) -> np.ndarray:
    values: list[int] = []
    for path in sorted((annotation_root / category / "shared_annotation_50").glob("annotator_*.jsonl")):
        for row in read_jsonl(path):
            if "is_correct" in row:
                values.append(int(bool(row["is_correct"])))
                continue
            gold = normalize_label(row.get("ground_truth_label") or row.get("gold_label") or row.get("label"))
            pred = normalize_label(row.get("annotated_label") or row.get("prediction_label") or row.get("answer"))
            values.append(int(gold is not None and pred is not None and pred == gold))
    return np.asarray(values, dtype=np.int8)


def model_correctness(raw_roots: list[Path], category: str, model_name: str) -> np.ndarray:
    path = find_raw_file(raw_roots, category, model_name)
    if path is None:
        return np.asarray([], dtype=np.int8)
    values: list[int] = []
    for row in read_jsonl(path):
        pred = prediction_for_row(row)
        gold = normalize_label(row.get("gold_label"))
        values.append(int(pred is not None and gold is not None and pred == gold))
    return np.asarray(values, dtype=np.int8)


def bootstrap_ci(values: np.ndarray, n_bootstrap: int, rng: np.random.Generator) -> tuple[float, float, float]:
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(values.mean())
    samples = rng.binomial(values.size, point, size=n_bootstrap) / values.size
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return point, float(lo), float(hi)


def collect_stats(args: argparse.Namespace) -> list[dict[str, object]]:
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    series_order = ["human", *DEFAULT_MODEL_ORDER]
    for category in DEFAULT_CATEGORY_ORDER:
        for series in series_order:
            if series == "human":
                values = human_correctness(args.annotation_root, category)
            else:
                values = model_correctness(args.model_raw_root, category, series)
            point, lo, hi = bootstrap_ci(values, args.n_bootstrap, rng)
            rows.append(
                {
                    "category": category,
                    "series": series,
                    "display_name": MODEL_DISPLAY_NAMES[series],
                    "n": int(values.size),
                    "accuracy": point,
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["category", "series", "display_name", "n", "accuracy", "ci_low", "ci_high"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_grid(path: Path, rows: list[dict[str, object]], dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.titlesize": 18,
            "axes.labelsize": 18,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
        }
    )
    series_order = ["human", *DEFAULT_MODEL_ORDER]
    colors = {
        "human": "#8C8C8C",
        "gemini-3-flash": "#4C78A8",
        "kimi-audio": "#F58518",
        "mimo-audio": "#54A24B",
        "step-audio-2-mini": "#B279A2",
        "audioflamingo3": "#E45756",
    }
    markers = {
        "human": "o",
        "gemini-3-flash": "s",
        "kimi-audio": "^",
        "mimo-audio": "D",
        "step-audio-2-mini": "P",
        "audioflamingo3": "X",
    }
    by_category_series = {
        (str(row["category"]), str(row["series"])): row
        for row in rows
    }
    fig, axes = plt.subplots(3, 3, figsize=(18.0, 12.8), sharey=True)
    x = np.arange(len(series_order)) * 0.62
    bar_width = 0.42
    for ax, category in zip(axes.flat, DEFAULT_CATEGORY_ORDER):
        points = []
        lows = []
        highs = []
        for series in series_order:
            row = by_category_series[(category, series)]
            point = float(row["accuracy"])
            lo = float(row["ci_low"])
            hi = float(row["ci_high"])
            points.append(point)
            lows.append(point - lo)
            highs.append(hi - point)
        for i, series in enumerate(series_order):
            ax.bar(
                x[i],
                points[i] * 100,
                color=colors[series],
                width=bar_width,
                edgecolor="white",
                linewidth=0.5,
                zorder=2,
            )
            if lows[i] == 0 and highs[i] == 0:
                continue
            ax.errorbar(
                x[i],
                points[i] * 100,
                yerr=[[lows[i] * 100], [highs[i] * 100]],
                fmt="none",
                ecolor="#222222",
                elinewidth=1.8,
                capsize=4.5,
                capthick=1.8,
                zorder=3,
            )
        ax.axhline(50, color="#9a9a9a", linewidth=0.8, linestyle="--", zorder=0)
        ax.set_title(CATEGORY_TITLES[category])
        ax.set_ylim(35, 103)
        ax.set_xlim(x[0] - 0.35, x[-1] + 0.35)
        ax.set_xticks(x)
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Accuracy (%)")
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[s], edgecolor="none", label=MODEL_DISPLAY_NAMES[s])
        for s in series_order
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=(0, 0.11, 1, 1.0))
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = collect_stats(args)
    write_csv(args.out_csv, rows)
    plot_grid(args.out, rows, args.dpi)
    print(f"wrote CSV: {args.out_csv}")
    print(f"wrote plot: {args.out}")
    print(f"rows={len(rows)} n_bootstrap={args.n_bootstrap} seed={args.seed}")


if __name__ == "__main__":
    main()

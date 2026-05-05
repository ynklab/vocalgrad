#!/usr/bin/env python3
"""Plot tier-wise VocalGrad accuracies for human annotators and selected models."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from statistics import mean
from typing import Iterable

default_mpl_config_dir = Path(os.environ.get("TMPDIR", "/tmp")) / f"plic-matplotlib-{os.getuid()}"
default_mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(default_mpl_config_dir))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


TIER_ORDER = ["low", "mid", "high"]
DEFAULT_MODELS = ["kimi-audio", "mimo-audio"]
DEFAULT_EXCLUDED_CATEGORIES: list[str] = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotation-root",
        type=Path,
        default=Path("outputs/annotation_data"),
        help="Root containing per-category shared_annotation_50 JSONL files.",
    )
    parser.add_argument(
        "--model-analysis-root",
        type=Path,
        default=Path("outputs/analysis/vocalgrad/default"),
        help="Root containing per-category model summary JSON files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/analysis/vocalgrad/default/human_model_tier_accuracy_lines"),
        help="Directory for generated CSV and plots.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=DEFAULT_MODELS.copy(),
        help="Model stem to include. Can be repeated.",
    )
    parser.add_argument(
        "--exclude-category",
        action="append",
        default=DEFAULT_EXCLUDED_CATEGORIES.copy(),
        help="Category to exclude from plots. Can be repeated.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc


def row_is_correct(row: dict, path: Path) -> bool:
    if "is_correct" in row:
        return bool(row["is_correct"])
    gold = str(row.get("ground_truth_label", "")).strip().lower()
    pred = str(row.get("annotated_label", "")).strip().lower()
    if not gold or not pred:
        raise KeyError(f"Cannot infer correctness in {path}; expected is_correct or labels.")
    return gold == pred


def collect_human_tier_accuracy(annotation_root: Path) -> dict[str, dict[str, float]]:
    """Return category -> tier -> mean annotator accuracy."""
    summaries: dict[str, dict[str, float]] = {}
    for category_dir in sorted(path for path in annotation_root.iterdir() if path.is_dir()):
        annotator_paths = sorted((category_dir / "shared_annotation_50").glob("annotator_*.jsonl"))
        if not annotator_paths:
            continue

        tier_values: dict[str, list[float]] = {tier: [] for tier in TIER_ORDER}
        for path in annotator_paths:
            counts = {tier: {"correct": 0, "total": 0} for tier in TIER_ORDER}
            for row in read_jsonl(path):
                tier = str(row.get("difficulty", "")).strip().lower()
                if tier not in counts:
                    continue
                counts[tier]["total"] += 1
                counts[tier]["correct"] += int(row_is_correct(row, path))
            for tier, values in counts.items():
                if values["total"]:
                    tier_values[tier].append(values["correct"] / values["total"])

        summaries[category_dir.name] = {
            tier: mean(values) for tier, values in tier_values.items() if values
        }
    return summaries


def collect_model_tier_accuracy(
    model_analysis_root: Path, models: list[str]
) -> dict[str, dict[str, dict[str, float]]]:
    """Return category -> model -> tier -> accuracy."""
    summaries: dict[str, dict[str, dict[str, float]]] = {}
    for category_dir in sorted(path for path in model_analysis_root.iterdir() if path.is_dir()):
        category = category_dir.name
        for model in models:
            path = category_dir / f"{model}.json"
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            by_tier = payload.get("by_tier", {})
            tier_scores = {
                tier: float(by_tier[tier]["accuracy"])
                for tier in TIER_ORDER
                if tier in by_tier and by_tier[tier].get("accuracy") is not None
            }
            if tier_scores:
                summaries.setdefault(category, {})[model] = tier_scores
    return summaries


def ordered_categories(
    human: dict[str, dict[str, float]],
    models: dict[str, dict[str, dict[str, float]]],
    model_analysis_root: Path,
    excluded_categories: set[str],
) -> list[str]:
    comparison_csv = model_analysis_root / "accuracy_comparison.csv"
    if comparison_csv.exists():
        with comparison_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            ordered = [row["category"] for row in reader if row.get("category")]
    else:
        ordered = sorted(set(human) | set(models))
    extras = sorted((set(human) | set(models)) - set(ordered))
    return [category for category in [*ordered, *extras] if category not in excluded_categories]


def display_category(category: str) -> str:
    return category.replace("_", " ")


def display_series_name(name: str) -> str:
    names = {
        "human_avg": "Human Avg.",
        "kimi-audio": "Kimi-Audio",
        "mimo-audio": "MiMo-Audio",
    }
    return names.get(name, name.replace("-", " ").title())


def write_csv(
    path: Path,
    categories: list[str],
    human: dict[str, dict[str, float]],
    model_scores: dict[str, dict[str, dict[str, float]]],
    models: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["category", "tier", "human_avg", *models]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for category in categories:
            for tier in TIER_ORDER:
                row = {
                    "category": category,
                    "tier": tier,
                    "human_avg": human.get(category, {}).get(tier),
                }
                for model in models:
                    row[model] = model_scores.get(category, {}).get(model, {}).get(tier)
                writer.writerow(row)


def category_series(
    category: str,
    human: dict[str, dict[str, float]],
    model_scores: dict[str, dict[str, dict[str, float]]],
    models: list[str],
) -> dict[str, list[float | None]]:
    series = {
        "human_avg": [human.get(category, {}).get(tier) for tier in TIER_ORDER],
    }
    for model in models:
        series[model] = [model_scores.get(category, {}).get(model, {}).get(tier) for tier in TIER_ORDER]
    return series


def plot_category(
    out_path: Path,
    category: str,
    series: dict[str, list[float | None]],
    dpi: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    draw_lines(ax, series)
    ax.set_title(display_category(category))
    ax.set_xlabel("Tier")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.8, alpha=0.75)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def draw_lines(ax: plt.Axes, series: dict[str, list[float | None]], with_legend: bool = True) -> None:
    colors = {
        "human_avg": "#222222",
        "kimi-audio": "#D55E00",
        "mimo-audio": "#0072B2",
    }
    markers = {
        "human_avg": "o",
        "kimi-audio": "s",
        "mimo-audio": "^",
    }
    x = list(range(len(TIER_ORDER)))
    for name, values in series.items():
        ax.plot(
            x,
            values,
            marker=markers.get(name, "o"),
            linewidth=2.2,
            markersize=5,
            color=colors.get(name),
            label=display_series_name(name) if with_legend else None,
        )
    ax.set_xticks(x, [tier.title() for tier in TIER_ORDER])


def plot_overview(
    out_path: Path,
    categories: list[str],
    human: dict[str, dict[str, float]],
    model_scores: dict[str, dict[str, dict[str, float]]],
    models: list[str],
    dpi: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ncols = 3
    nrows = (len(categories) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.1 * nrows), sharey=True)
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for ax, category in zip(axes_list, categories):
        draw_lines(ax, category_series(category, human, model_scores, models), with_legend=False)
        ax.set_title(display_category(category), fontsize=11)
        ax.set_ylim(0.0, 1.02)
        ax.grid(axis="y", color="#d8d8d8", linewidth=0.8, alpha=0.75)
    for ax in axes_list[len(categories) :]:
        ax.axis("off")
    for ax in axes_list[::ncols]:
        ax.set_ylabel("Accuracy")
    handles, labels = axes_list[0].get_legend_handles_labels()
    if not handles:
        handles, labels = [], []
        for name in ["human_avg", *models]:
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    color={"human_avg": "#222222", "kimi-audio": "#D55E00", "mimo-audio": "#0072B2"}.get(name),
                    marker={"human_avg": "o", "kimi-audio": "s", "mimo-audio": "^"}.get(name, "o"),
                    linewidth=2.2,
                )
            )
            labels.append(display_series_name(name))
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False)
    fig.suptitle("Tier-wise VocalGrad Accuracy: Human Avg. vs Kimi-Audio vs MiMo-Audio", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    excluded_categories = set(args.exclude_category)
    human = collect_human_tier_accuracy(args.annotation_root)
    model_scores = collect_model_tier_accuracy(args.model_analysis_root, args.model)
    categories = ordered_categories(human, model_scores, args.model_analysis_root, excluded_categories)

    csv_path = args.out_dir / "human_model_tier_accuracy.csv"
    write_csv(csv_path, categories, human, model_scores, args.model)
    for category in categories:
        plot_category(
            args.out_dir / f"{category}.png",
            category,
            category_series(category, human, model_scores, args.model),
            args.dpi,
        )
    plot_overview(args.out_dir / "overview.png", categories, human, model_scores, args.model, args.dpi)

    print(f"wrote CSV: {csv_path}")
    print(f"wrote category plots: {args.out_dir}")
    print(f"categories={len(categories)} models={len(args.model)} excluded={sorted(excluded_categories)}")


if __name__ == "__main__":
    main()

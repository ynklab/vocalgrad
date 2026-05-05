#!/usr/bin/env python3
"""Plot mean linear-probing accuracy across selected representation stages."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from statistics import mean
from typing import Any

default_mpl_config_dir = Path(os.environ.get("TMPDIR", "/tmp")) / f"plic-matplotlib-{os.getuid()}"
default_mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(default_mpl_config_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_MODEL_ORDER = [
    "audioflamingo3",
    "kimi-audio",
    "mimo-audio",
    "step-audio-2-mini",
]
DEFAULT_STAGE_ORDER = ["mel", "pre_lm_audio", "lm_audio"]
DEFAULT_EXCLUDED_CATEGORIES: list[str] = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        default=Path("outputs/analysis/linear_probe/_comparisons/default/comparison.csv"),
        help="Linear-probe comparison CSV.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/analysis/linear_probe/_comparisons/default/stage_line_charts"),
        help="Output directory for the chart and summarized CSV.",
    )
    parser.add_argument("--metric", default="accuracy", choices=["accuracy", "balanced_accuracy"])
    parser.add_argument("--stage", action="append", default=DEFAULT_STAGE_ORDER.copy())
    parser.add_argument("--model", action="append", default=DEFAULT_MODEL_ORDER.copy())
    parser.add_argument(
        "--exclude-category",
        action="append",
        default=DEFAULT_EXCLUDED_CATEGORIES.copy(),
        help="Category to exclude before averaging. Can be repeated.",
    )
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--font-size", type=float, default=18.0)
    parser.add_argument("--font-weight", default="bold")
    parser.add_argument("--category-overview-ncols", type=int, default=3)
    parser.add_argument("--category-overview-width", type=float, default=16.0)
    parser.add_argument("--category-overview-row-height", type=float, default=1.9)
    parser.add_argument(
        "--category-overview-title",
        default="",
        help="Figure-level title for the category overview. Use empty string to disable.",
    )
    parser.add_argument(
        "--category-overview-legend-location",
        choices=["top", "bottom"],
        default="bottom",
    )
    return parser.parse_args()


def display_model_name(model: str) -> str:
    names = {
        "audioflamingo3": "AudioFlamingo3",
        "kimi-audio": "Kimi-Audio",
        "mimo-audio": "MiMo-Audio",
        "step-audio-2-mini": "Step-Audio-2 Mini",
    }
    return names.get(model, model.replace("-", " ").title())


def display_stage_name(stage: str) -> str:
    names = {
        "mel": "mel",
        "pre_lm_audio": "pre_lm_audio",
        "lm_audio": "lm_audio",
    }
    return names.get(stage, stage)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def summarize(
    rows: list[dict[str, str]],
    metric: str,
    models: list[str],
    stages: list[str],
    excluded_categories: set[str],
) -> list[dict[str, object]]:
    values: dict[tuple[str, str], list[float]] = {
        (model, stage): [] for model in models for stage in stages
    }
    for row in rows:
        model = row.get("model_stem")
        stage = row.get("stage")
        category = row.get("category")
        if model not in models or stage not in stages or category in excluded_categories:
            continue
        value = row.get(metric)
        if value in (None, ""):
            continue
        values[(model, stage)].append(float(value))

    summary_rows = []
    for model in models:
        for stage in stages:
            stage_values = values[(model, stage)]
            summary_rows.append(
                {
                    "model_stem": model,
                    "stage": stage,
                    f"mean_{metric}": mean(stage_values) if stage_values else None,
                    "n_categories": len(stage_values),
                }
            )
    return summary_rows


def category_order(rows: list[dict[str, str]], excluded_categories: set[str]) -> list[str]:
    seen = []
    for row in rows:
        category = row.get("category")
        if not category or category in excluded_categories or category in seen:
            continue
        seen.append(category)
    return seen


def summarize_by_category(
    rows: list[dict[str, str]],
    metric: str,
    models: list[str],
    stages: list[str],
    categories: list[str],
) -> list[dict[str, object]]:
    values: dict[tuple[str, str, str], float] = {}
    for row in rows:
        category = row.get("category")
        model = row.get("model_stem")
        stage = row.get("stage")
        if category not in categories or model not in models or stage not in stages:
            continue
        value = row.get(metric)
        if value in (None, ""):
            continue
        values[(category, model, stage)] = float(value)

    summary_rows = []
    for category in categories:
        for model in models:
            for stage in stages:
                summary_rows.append(
                    {
                        "category": category,
                        "model_stem": model,
                        "stage": stage,
                        metric: values.get((category, model, stage)),
                    }
                )
    return summary_rows


def write_summary_csv(path: Path, rows: list[dict[str, object]], metric: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model_stem", "stage", f"mean_{metric}", "n_categories"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_category_csv(path: Path, rows: list[dict[str, object]], metric: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["category", "model_stem", "stage", metric]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_style() -> tuple[dict[str, str], dict[str, str]]:
    colors = {
        "audioflamingo3": "#7A5195",
        "kimi-audio": "#EF5675",
        "mimo-audio": "#FFA600",
        "step-audio-2-mini": "#003F5C",
    }
    markers = {
        "audioflamingo3": "o",
        "kimi-audio": "s",
        "mimo-audio": "^",
        "step-audio-2-mini": "D",
    }
    return colors, markers


def draw_model_stage_lines(
    ax: plt.Axes,
    by_model_stage: dict[tuple[str, str], Any],
    metric: str,
    models: list[str],
    stages: list[str],
    font_size: float,
    font_weight: str,
    stage_tick_labels: list[str] | None = None,
    legend: bool = True,
) -> None:
    colors, markers = plot_style()
    x = list(range(len(stages)))
    for model in models:
        y = [by_model_stage.get((model, stage)) for stage in stages]
        ax.plot(
            x,
            y,
            label=display_model_name(model) if legend else None,
            color=colors.get(model),
            marker=markers.get(model, "o"),
            linewidth=2.4,
            markersize=6,
        )
    tick_labels = stage_tick_labels or [display_stage_name(stage) for stage in stages]
    ax.set_xticks(x, tick_labels, fontsize=font_size)
    for label in ax.get_xticklabels():
        label.set_fontweight(font_weight)
    ax.tick_params(axis="y", labelsize=font_size)
    for label in ax.get_yticklabels():
        label.set_fontweight(font_weight)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=font_size, fontweight=font_weight)
    ax.set_ylim(0.55, 1.02)
    ax.grid(axis="y", color="#d7d7d7", linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_lines(
    out_png: Path,
    out_svg: Path,
    summary_rows: list[dict[str, object]],
    metric: str,
    models: list[str],
    stages: list[str],
    excluded_categories: set[str],
    dpi: int,
    font_size: float,
    font_weight: str,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    by_model_stage = {
        (str(row["model_stem"]), str(row["stage"])): row[f"mean_{metric}"]
        for row in summary_rows
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    draw_model_stage_lines(
        ax,
        by_model_stage,
        f"Mean {metric}",
        models,
        stages,
        font_size=font_size,
        font_weight=font_weight,
    )
    ax.set_ylabel(f"Mean {metric.replace('_', ' ')}", fontsize=font_size, fontweight=font_weight)
    ax.set_ylim(0.70, 1.00)
    legend_obj = ax.legend(frameon=False, loc="lower left", ncol=2, fontsize=font_size)
    for text in legend_obj.get_texts():
        text.set_fontweight(font_weight)
    title = "Average Linear-Probing Accuracy by Representation Stage"
    if excluded_categories:
        title += f" (excluding {', '.join(sorted(excluded_categories))})"
    ax.set_title(title, fontsize=font_size, fontweight=font_weight)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    fig.savefig(out_svg)
    plt.close(fig)


def display_category(category: str) -> str:
    return category.replace("_", " ")


def category_values(
    category_rows: list[dict[str, object]],
    category: str,
    metric: str,
) -> dict[tuple[str, str], object]:
    return {
        (str(row["model_stem"]), str(row["stage"])): row[metric]
        for row in category_rows
        if row["category"] == category
    }


def plot_category_lines(
    out_dir: Path,
    category_rows: list[dict[str, object]],
    metric: str,
    models: list[str],
    stages: list[str],
    categories: list[str],
    dpi: int,
    font_size: float,
    font_weight: str,
) -> None:
    png_dir = out_dir / "by_category_png"
    svg_dir = out_dir / "by_category_svg"
    png_dir.mkdir(parents=True, exist_ok=True)
    svg_dir.mkdir(parents=True, exist_ok=True)
    for category in categories:
        values = category_values(category_rows, category, metric)
        fig, ax = plt.subplots(figsize=(6.2, 3.9))
        draw_model_stage_lines(
            ax,
            values,
            metric,
            models,
            stages,
            font_size=font_size,
            font_weight=font_weight,
        )
        ax.set_title(display_category(category), fontsize=font_size, fontweight=font_weight)
        legend_obj = ax.legend(frameon=False, loc="lower left", ncol=2, fontsize=font_size)
        for text in legend_obj.get_texts():
            text.set_fontweight(font_weight)
        fig.tight_layout()
        fig.savefig(png_dir / f"{category}.png", dpi=dpi)
        fig.savefig(svg_dir / f"{category}.svg")
        plt.close(fig)


def plot_category_overview(
    out_png: Path,
    out_svg: Path,
    category_rows: list[dict[str, object]],
    metric: str,
    models: list[str],
    stages: list[str],
    categories: list[str],
    dpi: int,
    font_size: float,
    font_weight: str,
    ncols: int,
    width: float,
    row_height: float,
    title: str,
    legend_location: str,
    out_pdf: Path | None = None,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    nrows = (len(categories) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, row_height * nrows), sharey=True)
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for plot_idx, (ax, category) in enumerate(zip(axes_list, categories)):
        row_idx = plot_idx // ncols
        draw_model_stage_lines(
            ax,
            category_values(category_rows, category, metric),
            metric,
            models,
            stages,
            font_size=font_size,
            font_weight=font_weight,
            stage_tick_labels=["Mel", "Pre-LLM", "Late-LLM"],
            legend=False,
        )
        ax.set_title(display_category(category), fontsize=font_size, fontweight=font_weight)
        ax.set_ylabel("")
        if row_idx < nrows - 1:
            ax.tick_params(axis="x", labelbottom=False)
    for ax in axes_list[len(categories) :]:
        ax.axis("off")
    for ax in axes_list[::ncols]:
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=font_size, fontweight=font_weight)

    colors, markers = plot_style()
    handles = [
        plt.Line2D(
            [0],
            [0],
            color=colors.get(model),
            marker=markers.get(model, "o"),
            linewidth=2.4,
            markersize=6,
        )
        for model in models
    ]
    labels = [display_model_name(model) for model in models]
    if legend_location == "top":
        legend_obj = fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=font_size)
        legend_rect = [0, 0, 1, 0.90]
        title_y = 0.99
    else:
        legend_obj = fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=font_size)
        legend_rect = [0, 0.07, 1, 1]
        title_y = 0.99
    for text in legend_obj.get_texts():
        text.set_fontweight(font_weight)
    if title:
        fig.suptitle(title, y=title_y, fontsize=font_size, fontweight=font_weight)
    fig.tight_layout(rect=legend_rect, h_pad=0.35, w_pad=0.6)
    fig.savefig(out_png, dpi=dpi)
    fig.savefig(out_svg)
    if out_pdf is not None:
        fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plt.rcParams.update(
        {
            "font.size": args.font_size,
            "font.weight": args.font_weight,
            "axes.titlesize": args.font_size,
            "axes.titleweight": args.font_weight,
            "axes.labelsize": args.font_size,
            "axes.labelweight": args.font_weight,
            "legend.fontsize": args.font_size,
            "xtick.labelsize": args.font_size,
            "ytick.labelsize": args.font_size,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    excluded_categories = set(args.exclude_category)
    rows = read_rows(args.comparison_csv)
    categories = category_order(rows, excluded_categories)
    summary_rows = summarize(rows, args.metric, args.model, args.stage, excluded_categories)
    category_rows = summarize_by_category(rows, args.metric, args.model, args.stage, categories)
    summary_csv = args.out_dir / f"mean_{args.metric}_by_audio_stage.csv"
    category_csv = args.out_dir / f"category_{args.metric}_by_audio_stage.csv"
    out_png = args.out_dir / f"mean_{args.metric}_by_audio_stage.png"
    out_svg = args.out_dir / f"mean_{args.metric}_by_audio_stage.svg"
    category_overview_png = args.out_dir / f"category_{args.metric}_by_audio_stage_overview.png"
    category_overview_svg = args.out_dir / f"category_{args.metric}_by_audio_stage_overview.svg"
    category_overview_pdf = args.out_dir / f"category_{args.metric}_by_audio_stage_overview.pdf"
    write_summary_csv(summary_csv, summary_rows, args.metric)
    write_category_csv(category_csv, category_rows, args.metric)
    plot_lines(
        out_png=out_png,
        out_svg=out_svg,
        summary_rows=summary_rows,
        metric=args.metric,
        models=args.model,
        stages=args.stage,
        excluded_categories=excluded_categories,
        dpi=args.dpi,
        font_size=args.font_size,
        font_weight=args.font_weight,
    )
    plot_category_lines(
        out_dir=args.out_dir,
        category_rows=category_rows,
        metric=args.metric,
        models=args.model,
        stages=args.stage,
        categories=categories,
        dpi=args.dpi,
        font_size=args.font_size,
        font_weight=args.font_weight,
    )
    plot_category_overview(
        out_png=category_overview_png,
        out_svg=category_overview_svg,
        category_rows=category_rows,
        metric=args.metric,
        models=args.model,
        stages=args.stage,
        categories=categories,
        dpi=args.dpi,
        font_size=args.font_size,
        font_weight=args.font_weight,
        ncols=args.category_overview_ncols,
        width=args.category_overview_width,
        row_height=args.category_overview_row_height,
        title=args.category_overview_title,
        legend_location=args.category_overview_legend_location,
        out_pdf=category_overview_pdf,
    )
    print(f"wrote CSV: {summary_csv}")
    print(f"wrote category CSV: {category_csv}")
    print(f"wrote PNG: {out_png}")
    print(f"wrote SVG: {out_svg}")
    print(f"wrote category overview PNG: {category_overview_png}")
    print(f"wrote category overview SVG: {category_overview_svg}")
    print(f"wrote category overview PDF: {category_overview_pdf}")
    print(
        f"models={len(args.model)} categories={len(categories)} "
        f"stages={args.stage} excluded={sorted(excluded_categories)}"
    )


if __name__ == "__main__":
    main()

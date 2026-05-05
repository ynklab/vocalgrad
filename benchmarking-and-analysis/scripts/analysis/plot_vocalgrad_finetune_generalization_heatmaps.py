from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from plic.kimi_finetune import DEFAULT_KIMI_FINETUNE_ANALYSIS_ROOT, resolve_categories


NEUTRAL_ACCURACY = 0.5
DEFAULT_EXCLUDED_CATEGORIES: list[str] = []
DISPLAY_CATEGORY_NAMES = {
    "speaking_speed": "speed",
    "voice_pitch": "pitch",
    "volume": "volume",
    "audio_distortion": "distortion",
    "audio_roughness": "roughness",
    "background_noise": "noise",
    "echo": "echo",
    "voice_clarity": "clarity",
    "voice_vibration": "vibration",
}

DISPLAY_SCOPE_NAMES = {
    "all_linear": "All-Linear LoRA Fine-Tuned Model",
    "lm_head_only": "LM-Head-Only LoRA Fine-Tuned Model",
}
ACCURACY_CMAP = LinearSegmentedColormap.from_list(
    "accuracy_neutral",
    [
        (0.0, "#2166ac"),
        (0.5, "#bdbdbd"),
        (1.0, "#b2182b"),
    ],
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot heatmaps for Kimi fine-tuning generalization results.")
    parser.add_argument("--root", type=Path, default=DEFAULT_KIMI_FINETUNE_ANALYSIS_ROOT)
    parser.add_argument("--run-name", default="default")
    parser.add_argument("--comparison-dir", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/vocalgrad/test"))
    parser.add_argument("--font-size", type=float, default=20.0)
    parser.add_argument("--font-weight", default="bold")
    parser.add_argument("--exclude-category", action="append", default=DEFAULT_EXCLUDED_CATEGORIES.copy())
    return parser


def _display_category_name(category: str) -> str:
    return DISPLAY_CATEGORY_NAMES.get(category, category)


def _display_scope_name(scope: str) -> str:
    return DISPLAY_SCOPE_NAMES.get(scope, scope.replace("_", " ").title())


def _plot_heatmap(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    out_prefix: Path,
    categories: list[str],
    font_size: float,
    font_weight: str,
) -> None:
    if df.empty:
        return
    pivot = df.pivot(index="train_category", columns="test_category", values=value_col)
    missing_mask = pivot.reindex(index=categories, columns=categories).isna()
    pivot = pivot.reindex(index=categories, columns=categories, fill_value=NEUTRAL_ACCURACY).fillna(NEUTRAL_ACCURACY)
    annotations = pivot.copy().astype(object)
    for row_label in annotations.index:
        for col_label in annotations.columns:
            if bool(missing_mask.loc[row_label, col_label]):
                annotations.loc[row_label, col_label] = "NA"
            else:
                annotations.loc[row_label, col_label] = f"{float(pivot.loc[row_label, col_label]):.2f}"
    plt.figure(figsize=(10.5, 8.0))
    ax = sns.heatmap(
        pivot,
        annot=annotations,
        fmt="",
        cmap=ACCURACY_CMAP,
        vmin=0.0,
        vmax=1.0,
        annot_kws={"size": font_size * 0.8, "weight": font_weight},
        cbar_kws={"fraction": 0.035, "pad": 0.02, "aspect": 28, "shrink": 0.82},
        square=True,
    )
    ax.set_title(title, fontweight=font_weight)
    ax.set_xlabel("Test Category", fontweight=font_weight)
    ax.set_ylabel("Train Category", fontweight=font_weight)
    ax.set_xticklabels([_display_category_name(c) for c in pivot.columns], rotation=45, ha="right")
    ax.set_yticklabels([_display_category_name(c) for c in pivot.index], rotation=0)
    ax.tick_params(axis="both", labelsize=font_size)
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontweight(font_weight)
    ax.xaxis.label.set_size(font_size)
    ax.yaxis.label.set_size(font_size)
    ax.title.set_size(font_size)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=font_size)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight(font_weight)
    plt.tight_layout()
    plt.savefig(out_prefix.with_suffix(".png"), dpi=220)
    plt.savefig(out_prefix.with_suffix(".svg"))
    plt.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close()


def _plot_delta_heatmap(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    out_prefix: Path,
    categories: list[str],
    font_size: float,
    font_weight: str,
) -> None:
    subset = df.dropna(subset=[value_col]).copy()
    if subset.empty:
        return
    pivot = subset.pivot(index="train_category", columns="test_category", values=value_col)
    pivot = pivot.reindex(index=categories, columns=categories, fill_value=0.0).fillna(0.0)
    vmax = float(pivot.abs().max().max()) if not pivot.empty else 0.0
    vmax = max(vmax, 1e-6)
    plt.figure(figsize=(10.5, 8.0))
    ax = sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0.0,
        vmin=-vmax,
        vmax=vmax,
        annot_kws={"size": font_size * 0.8, "weight": font_weight},
        cbar_kws={"fraction": 0.035, "pad": 0.02, "aspect": 28, "shrink": 0.82},
        square=True,
    )
    ax.set_title(title, fontweight=font_weight)
    ax.set_xlabel("Test Category", fontweight=font_weight)
    ax.set_ylabel("Train Category", fontweight=font_weight)
    ax.set_xticklabels([_display_category_name(c) for c in pivot.columns], rotation=45, ha="right")
    ax.set_yticklabels([_display_category_name(c) for c in pivot.index], rotation=0)
    ax.tick_params(axis="both", labelsize=font_size)
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontweight(font_weight)
    ax.xaxis.label.set_size(font_size)
    ax.yaxis.label.set_size(font_size)
    ax.title.set_size(font_size)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=font_size)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight(font_weight)
    plt.tight_layout()
    plt.savefig(out_prefix.with_suffix(".png"), dpi=220)
    plt.savefig(out_prefix.with_suffix(".svg"))
    plt.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close()


def main() -> None:
    args = build_parser().parse_args()
    comparison_dir = args.comparison_dir or (args.root / "_comparisons" / args.run_name)
    records_path = comparison_dir / "generalization_records.csv"
    if not records_path.exists():
        raise SystemExit(f"comparison records not found: {records_path}")

    out_dir = comparison_dir / "heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.size": args.font_size,
            "font.weight": args.font_weight,
            "axes.titlesize": args.font_size,
            "axes.titleweight": args.font_weight,
            "axes.labelsize": args.font_size,
            "axes.labelweight": args.font_weight,
            "xtick.labelsize": args.font_size,
            "ytick.labelsize": args.font_size,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    df = pd.read_csv(records_path)
    categories = resolve_categories(args.dataset_root, None)
    excluded_categories = set(args.exclude_category)
    if excluded_categories:
        df = df[
            ~df["train_category"].isin(excluded_categories) & ~df["test_category"].isin(excluded_categories)
        ].copy()
        categories = [category for category in categories if category not in excluded_categories]
    for scope in sorted(df["scope"].unique()):
        scope_df = df[df["scope"] == scope].copy()
        scope_name = _display_scope_name(scope)
        _plot_heatmap(
            scope_df,
            "balanced_accuracy",
            f"{scope_name} Balanced Accuracy",
            out_dir / f"balanced_accuracy__{scope}",
            categories,
            args.font_size,
            args.font_weight,
        )
        _plot_heatmap(
            scope_df,
            "accuracy",
            f"{scope_name} Accuracy",
            out_dir / f"accuracy__{scope}",
            categories,
            args.font_size,
            args.font_weight,
        )
        _plot_delta_heatmap(
            scope_df,
            "delta_vs_base_accuracy",
            f"{scope_name} Accuracy Delta vs. Base Kimi-Audio",
            out_dir / f"delta_vs_base_accuracy__{scope}",
            categories,
            args.font_size,
            args.font_weight,
        )
    print(f"wrote Kimi fine-tuning heatmaps to {out_dir}")


if __name__ == "__main__":
    main()

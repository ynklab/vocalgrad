from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd
import seaborn as sns

DEFAULT_ROOT = Path("outputs/analysis/linear_probe_lm_text_layers")
DEFAULT_MODEL_ORDER = [
    "kimi-audio",
    "mimo-audio",
    "step-audio-2-mini",
    "audioflamingo3",
]
DEFAULT_EXCLUDED_CATEGORIES: list[str] = []


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot line charts across hidden-state indices for lm_text linear-probe results."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--run-name", default="default")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--model", action="append", default=DEFAULT_MODEL_ORDER.copy())
    parser.add_argument(
        "--exclude-category",
        action="append",
        default=DEFAULT_EXCLUDED_CATEGORIES.copy(),
    )
    parser.add_argument("--font-size", type=float, default=18.0)
    parser.add_argument("--font-weight", default="bold")
    parser.add_argument("--overview-width", type=float, default=16.0)
    parser.add_argument("--overview-height", type=float, default=3.8)
    parser.add_argument("--overview-x-tick-step", type=int, default=10)
    return parser


def display_model_name(model_name: str) -> str:
    names = {
        "audioflamingo3": "AudioFlamingo3",
        "kimi-audio": "Kimi-Audio",
        "mimo-audio": "MiMo-Audio",
        "step-audio-2-mini": "Step-Audio-2 Mini",
    }
    return names.get(model_name, model_name.replace("-", " ").title())


def display_category_name(category_name: str) -> str:
    return category_name.replace("_", " ")


def resolve_model_max_layer(df: pd.DataFrame, model_name: str) -> int | None:
    subset = df[df["model"] == model_name]
    if subset.empty:
        return None
    return int(subset["layer"].max())


def _append_layer_records(
    records: list[dict[str, object]],
    model_name: str,
    payload: dict[str, object],
    *,
    category: str | None,
) -> None:
    layers = payload.get("layers", {})
    if not isinstance(layers, dict):
        return
    for layer_key, section in layers.items():
        if not str(layer_key).isdigit() or not isinstance(section, dict):
            continue
        layer_idx = int(layer_key)
        for metric_name in ("accuracy", "balanced_accuracy"):
            value = section.get(metric_name)
            if isinstance(value, (int, float)):
                records.append(
                    {
                        "model": model_name,
                        "category": category,
                        "layer": layer_idx,
                        "metric": metric_name,
                        "value": float(value),
                    }
                )


def load_mean_records(root: Path, run_name: str) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("_"):
            continue
        run_dir = model_dir / run_name
        summary_path = run_dir / "results.json"
        if not summary_path.exists():
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        _append_layer_records(records, model_dir.name, payload, category=None)
    if not records:
        raise SystemExit(f"no layer-wise lm_text probe results found under {root} for run-name={run_name}")
    return pd.DataFrame.from_records(records)


def load_category_records(root: Path, run_name: str) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("_"):
            continue
        run_dir = model_dir / run_name
        if not run_dir.exists():
            continue
        for category_path in sorted(run_dir.glob("*.json")):
            if category_path.name == "results.json":
                continue
            payload = json.loads(category_path.read_text(encoding="utf-8"))
            category_name = payload.get("category")
            if not isinstance(category_name, str) or not category_name:
                category_name = category_path.stem
            _append_layer_records(records, model_dir.name, payload, category=category_name)
    if not records:
        raise SystemExit(f"no per-category lm_text probe results found under {root} for run-name={run_name}")
    return pd.DataFrame.from_records(records)


def draw_model_mean_line_chart(df: pd.DataFrame, metric_name: str, out_prefix: Path) -> None:
    subset = df[df["metric"] == metric_name].copy()
    subset = subset.sort_values(["model", "layer"]).reset_index(drop=True)

    plt.figure(figsize=(10.2, 5.2))
    ax = sns.lineplot(
        data=subset,
        x="layer",
        y="value",
        hue="model",
        style="model",
        markers=True,
        dashes=False,
    )
    ax.set_title(f"LM-Text Probe {metric_name.replace('_', ' ').title()} By Hidden-State Index")
    ax.set_xlabel("Hidden-State Index")
    ax.set_ylabel(metric_name.replace("_", " ").title())
    ax.set_ylim(0.0, 1.0)
    ax.legend(title="Model", loc="best")
    plt.tight_layout()
    plt.savefig(out_prefix.with_suffix(".png"), dpi=220)
    plt.savefig(out_prefix.with_suffix(".svg"))
    plt.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close()


def draw_category_line_chart(df: pd.DataFrame, metric_name: str, model_name: str, out_prefix: Path) -> None:
    subset = df[(df["metric"] == metric_name) & (df["model"] == model_name)].copy()
    if subset.empty:
        return
    max_layer = resolve_model_max_layer(subset, model_name)
    subset = subset.sort_values(["category", "layer"]).reset_index(drop=True)

    plt.figure(figsize=(11.5, 6.4))
    ax = sns.lineplot(
        data=subset,
        x="layer",
        y="value",
        hue="category",
        style="category",
        markers=False,
        dashes=False,
        linewidth=2.0,
    )
    ax.set_title(f"{display_model_name(model_name)} LM-Text Probe {metric_name.replace('_', ' ').title()} By Category")
    ax.set_xlabel("Hidden-State Index")
    ax.set_ylabel(metric_name.replace("_", " ").title())
    ax.set_ylim(0.5, 1.0)
    if max_layer is not None:
        ax.set_xlim(0.0, float(max_layer + 1))
    ax.margins(x=0.0)
    ax.legend(title="Category", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    plt.tight_layout()
    plt.savefig(out_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.savefig(out_prefix.with_suffix(".svg"), bbox_inches="tight")
    plt.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close()


def draw_category_overview(
    df: pd.DataFrame,
    metric_name: str,
    models: list[str],
    out_prefix: Path,
    font_size: float,
    font_weight: str,
    width: float,
    height: float,
    x_tick_step: int,
) -> None:
    subset = df[df["metric"] == metric_name].copy()
    subset = subset[subset["model"].isin(models)].copy()
    if subset.empty:
        return
    subset = subset.sort_values(["model", "category", "layer"]).reset_index(drop=True)

    palette = sns.color_palette("tab10", n_colors=max(1, subset["category"].nunique()))
    category_order = sorted(subset["category"].unique().tolist())
    color_map = {category: palette[idx] for idx, category in enumerate(category_order)}

    fig, axes = plt.subplots(1, len(models), figsize=(width, height), sharex=False, sharey=True)
    axes_list = list(axes.flat)
    legend_handles = legend_labels = None

    for idx, (ax, model_name) in enumerate(zip(axes_list, models)):
        model_df = subset[subset["model"] == model_name].copy()
        max_layer = resolve_model_max_layer(model_df, model_name)
        if model_df.empty:
            ax.axis("off")
            continue
        sns.lineplot(
            data=model_df,
            x="layer",
            y="value",
            hue="category",
            hue_order=category_order,
            palette=color_map,
            linewidth=2.0,
            legend=False,
            ax=ax,
        )
        ax.set_title(display_model_name(model_name), fontsize=font_size, fontweight=font_weight)
        ax.set_xlabel("Hidden-State Index", fontsize=font_size, fontweight=font_weight)
        ax.set_ylabel(metric_name.replace("_", " ").title() if idx == 0 else "", fontsize=font_size, fontweight=font_weight)
        ax.set_ylim(0.5, 1.0)
        if max_layer is not None:
            ax.set_xlim(0.0, float(max_layer + 1))
        ax.xaxis.set_major_locator(MultipleLocator(x_tick_step))
        ax.tick_params(axis="both", labelsize=font_size)
        ax.tick_params(axis="y", labelleft=(idx == 0))
        for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
            label.set_fontweight(font_weight)
        ax.margins(x=0.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if legend_handles is None:
            legend_handles = [
                plt.Line2D([0], [0], color=color_map[category], linewidth=2.0)
                for category in category_order
            ]
            legend_labels = [display_category_name(category) for category in category_order]

    for ax in axes_list[len(models) :]:
        ax.axis("off")

    if legend_handles and legend_labels:
        legend_obj = fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=min(5, len(legend_labels)),
            frameon=False,
            bbox_to_anchor=(0.5, -0.01),
            fontsize=font_size,
        )
        for text in legend_obj.get_texts():
            text.set_fontweight(font_weight)
    fig.tight_layout(rect=[0, 0.22, 1, 1], w_pad=0.55)
    fig.savefig(out_prefix.with_suffix(".png"), dpi=220)
    fig.savefig(out_prefix.with_suffix(".svg"))
    fig.savefig(out_prefix.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    out_dir = args.out_dir or (args.root / "_comparisons" / args.run_name / "layer_lines")
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
            "legend.fontsize": args.font_size,
            "xtick.labelsize": args.font_size,
            "ytick.labelsize": args.font_size,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    mean_df = load_mean_records(args.root, args.run_name)
    category_df = load_category_records(args.root, args.run_name)
    excluded_categories = set(args.exclude_category)
    if excluded_categories:
        category_df = category_df[~category_df["category"].isin(excluded_categories)].copy()
    draw_model_mean_line_chart(mean_df, "balanced_accuracy", out_dir / "balanced_accuracy_by_layer")
    draw_model_mean_line_chart(mean_df, "accuracy", out_dir / "accuracy_by_layer")
    requested_models = [model for model in args.model if model in set(category_df["model"].unique())]
    for model_name in requested_models:
        safe_model_name = str(model_name).replace("/", "_")
        draw_category_line_chart(
            category_df,
            "balanced_accuracy",
            str(model_name),
            out_dir / f"balanced_accuracy_by_layer__{safe_model_name}__by_category",
        )
        draw_category_line_chart(
            category_df,
            "accuracy",
            str(model_name),
            out_dir / f"accuracy_by_layer__{safe_model_name}__by_category",
        )
    draw_category_overview(
        category_df,
        "balanced_accuracy",
        requested_models,
        out_dir / "balanced_accuracy_by_layer__all_models__by_category_overview",
        font_size=args.font_size,
        font_weight=args.font_weight,
        width=args.overview_width,
        height=args.overview_height,
        x_tick_step=args.overview_x_tick_step,
    )
    draw_category_overview(
        category_df,
        "accuracy",
        requested_models,
        out_dir / "accuracy_by_layer__all_models__by_category_overview",
        font_size=args.font_size,
        font_weight=args.font_weight,
        width=args.overview_width,
        height=args.overview_height,
        x_tick_step=args.overview_x_tick_step,
    )
    print(f"wrote layer-wise lm_text line charts to {out_dir}")


if __name__ == "__main__":
    main()

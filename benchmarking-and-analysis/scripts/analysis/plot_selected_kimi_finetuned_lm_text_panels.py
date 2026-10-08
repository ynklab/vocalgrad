#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd
import seaborn as sns

DEFAULT_ROOT = Path("outputs/analysis/linear_probe_lm_text_layers_finetuned_kimi")
DEFAULT_OUT_DIR = Path("outputs/analysis/kimi_finetuned_probe_completed/epoch1/lm_text_layer_lines")
DEFAULT_MODELS = ["kimi-audio-ft-volume", "kimi-audio-ft-voice_pitch"]
DEFAULT_EXCLUDED_CATEGORIES = ["voice_brightness"]

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
DISPLAY_PANEL_NAMES = {
    "kimi-audio-ft-volume": "Volume",
    "kimi-audio-ft-voice_pitch": "Voice Pitch",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot selected finetuned-Kimi lm_text category panels.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--run-name", default="epoch1")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model", action="append", default=DEFAULT_MODELS.copy())
    parser.add_argument("--exclude-category", action="append", default=DEFAULT_EXCLUDED_CATEGORIES.copy())
    parser.add_argument("--font-size", type=float, default=18.0)
    parser.add_argument("--font-weight", default="bold")
    parser.add_argument("--width", type=float, default=8.6)
    parser.add_argument("--height", type=float, default=3.8)
    parser.add_argument("--x-tick-step", type=int, default=10)
    return parser


def display_category_name(category_name: str) -> str:
    return DISPLAY_CATEGORY_NAMES.get(category_name, category_name.replace("_", " "))


def display_panel_name(model_name: str) -> str:
    return DISPLAY_PANEL_NAMES.get(model_name, model_name.replace("kimi-audio-ft-", "").replace("_", " ").title())


def resolve_model_max_layer(df: pd.DataFrame, model_name: str) -> int | None:
    subset = df[df["model"] == model_name]
    if subset.empty:
        return None
    return int(subset["layer"].max())


def _append_layer_records(records: list[dict[str, object]], model_name: str, payload: dict[str, object], *, category: str) -> None:
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


def draw_selected_panels(
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
    subset = df[(df["metric"] == metric_name) & (df["model"].isin(models))].copy()
    if subset.empty:
        return
    subset = subset.sort_values(["model", "category", "layer"]).reset_index(drop=True)

    category_order = sorted(subset["category"].unique().tolist())
    palette = sns.color_palette("tab10", n_colors=max(1, len(category_order)))
    color_map = {category: palette[idx] for idx, category in enumerate(category_order)}

    fig, axes = plt.subplots(1, len(models), figsize=(width, height), sharey=True)
    if len(models) == 1:
        axes = [axes]

    legend_handles = [
        plt.Line2D([0], [0], color=color_map[category], linewidth=2.0)
        for category in category_order
    ]
    legend_labels = [display_category_name(category) for category in category_order]

    for idx, (ax, model_name) in enumerate(zip(axes, models)):
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
        ax.set_title(display_panel_name(model_name), fontsize=font_size, fontweight=font_weight)
        ax.set_xlabel("Layer Index", fontsize=font_size, fontweight=font_weight)
        ax.set_ylabel(
            metric_name.replace("_", " ").title() if idx == 0 else "",
            fontsize=font_size,
            fontweight=font_weight,
        )
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

    legend_obj = fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.16),
        fontsize=font_size,
    )
    for text in legend_obj.get_texts():
        text.set_fontweight(font_weight)
    fig.tight_layout(rect=[0, 0.12, 1, 1], w_pad=0.55)
    fig.savefig(out_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(out_prefix.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.15)
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
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
    category_df = load_category_records(args.root, args.run_name)
    excluded = set(args.exclude_category)
    if excluded:
        category_df = category_df[~category_df["category"].isin(excluded)].copy()
    requested_models = [model for model in args.model if model in set(category_df["model"].unique())]
    if not requested_models:
        raise SystemExit("no requested models found in lm_text records")
    model_tag = "__".join(model.replace("kimi-audio-ft-", "") for model in requested_models)
    draw_selected_panels(
        category_df,
        "balanced_accuracy",
        requested_models,
        args.out_dir / f"balanced_accuracy_by_layer__{model_tag}__by_category_panels",
        args.font_size,
        args.font_weight,
        args.width,
        args.height,
        args.x_tick_step,
    )
    draw_selected_panels(
        category_df,
        "accuracy",
        requested_models,
        args.out_dir / f"accuracy_by_layer__{model_tag}__by_category_panels",
        args.font_size,
        args.font_weight,
        args.width,
        args.height,
        args.x_tick_step,
    )
    print(f"wrote selected finetuned-Kimi lm_text panels to {args.out_dir}")


if __name__ == "__main__":
    main()

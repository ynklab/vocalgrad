from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from plic.kimi_finetune import resolve_categories


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
DISPLAY_MODEL_NAMES = {
    "audioflamingo3": "AudioFlamingo3",
    "kimi-audio": "Kimi-Audio",
    "mimo-audio": "MiMo-Audio",
    "step-audio-2-mini": "Step-Audio-2 Mini",
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
    parser = argparse.ArgumentParser(
        description="Plot VocalGrad cross-attribute accuracy heatmaps."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/analysis/vocalgrad_cross_attribute/default"),
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/vocalgrad/test"))
    parser.add_argument("--font-size", type=float, default=20.0)
    parser.add_argument("--font-weight", default="bold")
    parser.add_argument("--exclude-category", action="append", default=DEFAULT_EXCLUDED_CATEGORIES.copy())
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="Restrict output to one or more model names. Defaults to all models plus overall mean.",
    )
    return parser


def _display_category_name(category: str) -> str:
    return DISPLAY_CATEGORY_NAMES.get(category, category)


def _read_records(root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(root.glob("*/*/*.json")):
        if "_runs" in path.parts:
            continue
        data = json.loads(path.read_text())
        try:
            rel = path.relative_to(root)
        except ValueError:
            rel = path
        actual_attribute = str(data.get("actual_attribute") or rel.parts[0])
        prompt_attribute = str(data.get("prompt_attribute") or rel.parts[1])
        overall = data.get("overall", {})
        if "accuracy" not in overall:
            continue
        records.append(
            {
                "audio_category": actual_attribute,
                "query_word_category": prompt_attribute,
                "model": path.stem,
                "accuracy": float(overall["accuracy"]),
                "path": str(path),
            }
        )
    return records


def _mean_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    by_pair: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in records:
        key = (str(record["audio_category"]), str(record["query_word_category"]))
        by_pair[key].append(float(record["accuracy"]))
    return [
        {
            "audio_category": audio_category,
            "query_word_category": query_word_category,
            "model": "overall_mean",
            "accuracy": mean(values),
        }
        for (audio_category, query_word_category), values in sorted(by_pair.items())
        if values
    ]


def _plot_heatmap(
    df: pd.DataFrame,
    title: str,
    out_prefix: Path,
    categories: list[str],
    font_size: float,
    font_weight: str,
) -> None:
    if df.empty:
        return
    pivot = df.pivot(index="query_word_category", columns="audio_category", values="accuracy")
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
    ax.set_xlabel("Audio Category", fontweight=font_weight)
    ax.set_ylabel("Query Attribute Word", fontweight=font_weight)
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


def _title_for_model(model: str) -> str:
    if model == "overall_mean":
        return "Mean"
    display_name = DISPLAY_MODEL_NAMES.get(model, model.replace("-", " ").title())
    return display_name


def main() -> None:
    args = build_parser().parse_args()
    records = _read_records(args.root)
    if not records:
        raise SystemExit(f"no cross-attribute analysis files found under {args.root}")

    out_dir = args.out_dir or (args.root / "heatmaps")
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

    categories = resolve_categories(args.dataset_root, None)
    excluded_categories = set(args.exclude_category)
    if excluded_categories:
        records = [
            record
            for record in records
            if str(record["audio_category"]) not in excluded_categories
            and str(record["query_word_category"]) not in excluded_categories
        ]
        categories = [category for category in categories if category not in excluded_categories]

    df = pd.DataFrame(records)
    models = sorted(args.model or df["model"].unique())
    for model in models:
        model_df = df[df["model"] == model].copy()
        _plot_heatmap(
            model_df,
            _title_for_model(model),
            out_dir / f"accuracy__{model}",
            categories,
            args.font_size,
            args.font_weight,
        )

    if args.model is None:
        mean_df = pd.DataFrame(_mean_records(records))
        _plot_heatmap(
            mean_df,
            _title_for_model("overall_mean"),
            out_dir / "accuracy__overall_mean",
            categories,
            args.font_size,
            args.font_weight,
        )

    print(f"wrote VocalGrad cross-attribute heatmaps to {out_dir}")


if __name__ == "__main__":
    main()

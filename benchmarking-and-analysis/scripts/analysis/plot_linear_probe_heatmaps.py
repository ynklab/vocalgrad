from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

LINEAR_PROBE_STAGES = ("mel", "pre_lm_audio", "lm_audio", "lm_text")
DEFAULT_ROOT = Path("outputs/analysis/linear_probe")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot model x stage heatmaps for category-specific linear-probe results."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--run-name", default="default")
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser


def make_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "blue_gray_red",
        [
            (0.0, "#2b6cb0"),
            (0.5, "#b8b8b8"),
            (1.0, "#c53030"),
        ],
    )


def load_records(root: Path, run_name: str) -> pd.DataFrame:
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
            category = payload.get("category")
            stages = payload.get("stages", {})
            if not isinstance(category, str):
                continue
            for stage in LINEAR_PROBE_STAGES:
                section = stages.get(stage)
                if not isinstance(section, dict):
                    continue
                for metric_name in ("accuracy", "balanced_accuracy"):
                    value = section.get(metric_name)
                    if isinstance(value, (int, float)):
                        records.append(
                            {
                                "model": model_dir.name,
                                "category": category,
                                "stage": stage,
                                "metric": metric_name,
                                "value": float(value),
                            }
                        )
    if not records:
        raise SystemExit(f"no linear probe results found under {root} for run-name={run_name}")
    return pd.DataFrame.from_records(records)


def draw_heatmap(df: pd.DataFrame, metric_name: str, out_prefix: Path) -> None:
    subset = df[df["metric"] == metric_name].copy()
    grouped = subset.groupby(["stage", "model"], as_index=False)["value"].mean()
    pivot = grouped.pivot(index="stage", columns="model", values="value")
    pivot = pivot.reindex(LINEAR_PROBE_STAGES)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    max_delta = float((grouped["value"] - 0.5).abs().max())
    max_delta = max(max_delta, 0.01)
    norm = TwoSlopeNorm(vmin=0.5 - max_delta, vcenter=0.5, vmax=0.5 + max_delta)

    plt.figure(figsize=(9.5, 4.8))
    ax = sns.heatmap(
        pivot,
        cmap=make_cmap(),
        norm=norm,
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        linecolor="#f2f2f2",
        cbar_kws={"label": metric_name.replace("_", " ").title()},
        mask=pivot.isna(),
    )
    ax.set_title(f"Linear Probe Mean {metric_name.replace('_', ' ').title()} (center=0.5)")
    ax.set_xlabel("Model")
    ax.set_ylabel("Stage")
    plt.xticks(rotation=20, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_prefix.with_suffix(".png"), dpi=220)
    plt.savefig(out_prefix.with_suffix(".svg"))
    plt.close()


def main() -> None:
    args = build_parser().parse_args()
    out_dir = args.out_dir or (args.root / "_comparisons" / args.run_name / "heatmaps")
    out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="white")
    df = load_records(args.root, args.run_name)
    draw_heatmap(df, "balanced_accuracy", out_dir / "balanced_accuracy")
    draw_heatmap(df, "accuracy", out_dir / "accuracy")
    print(f"wrote linear probe heatmaps to {out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create a compact two-panel tier-accuracy figure for selected categories."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

default_mpl_config_dir = Path(os.environ.get("TMPDIR", "/tmp")) / f"plic-matplotlib-{os.getuid()}"
default_mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(default_mpl_config_dir))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


TIER_ORDER = ["low", "mid", "high"]
SERIES_ORDER = ["human_avg", "kimi-audio", "mimo-audio"]
COLORS = {
    "human_avg": "#222222",
    "kimi-audio": "#D55E00",
    "mimo-audio": "#0072B2",
}
MARKERS = {
    "human_avg": "o",
    "kimi-audio": "s",
    "mimo-audio": "^",
}
DISPLAY_NAMES = {
    "human_avg": "Human Avg.",
    "kimi-audio": "Kimi-Audio",
    "mimo-audio": "MiMo-Audio",
}
DISPLAY_CATEGORY = {
    "background_noise": "Background Noise",
    "voice_pitch": "Voice Pitch",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("outputs/analysis/vocalgrad/default/human_model_tier_accuracy_lines/human_model_tier_accuracy.csv"),
    )
    parser.add_argument(
        "--categories",
        nargs=2,
        default=["background_noise", "voice_pitch"],
    )
    parser.add_argument(
        "--out-pdf",
        type=Path,
        default=Path(
            "outputs/analysis/vocalgrad/default/human_model_tier_accuracy_lines/"
            "background_noise_voice_pitch_pair.pdf"
        ),
    )
    parser.add_argument(
        "--out-tex",
        type=Path,
        default=Path(
            "outputs/analysis/vocalgrad/default/human_model_tier_accuracy_lines/"
            "background_noise_voice_pitch_pair.tex"
        ),
    )
    parser.add_argument("--font-size", type=float, default=18.0)
    parser.add_argument("--width", type=float, default=10.4)
    parser.add_argument("--height", type=float, default=3.0)
    parser.add_argument("--y-min", type=float, default=0.4)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_csv(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    data: dict[str, dict[str, dict[str, float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = str(row["category"])
            tier = str(row["tier"])
            tier_map = data.setdefault(category, {})
            series_map = tier_map.setdefault(tier, {})
            for series_name in SERIES_ORDER:
                value = row.get(series_name)
                if value in {None, "", "None"}:
                    continue
                series_map[series_name] = float(value)
    return data


def category_series(
    data: dict[str, dict[str, dict[str, float]]],
    category: str,
) -> dict[str, list[float | None]]:
    return {
        series_name: [data.get(category, {}).get(tier, {}).get(series_name) for tier in TIER_ORDER]
        for series_name in SERIES_ORDER
    }


def draw_lines(
    ax: plt.Axes,
    series: dict[str, list[float | None]],
    font_size: float,
    show_legend: bool,
    y_min: float,
) -> None:
    x = list(range(len(TIER_ORDER)))
    for name in SERIES_ORDER:
        ax.plot(
            x,
            series[name],
            marker=MARKERS[name],
            linewidth=2.8,
            markersize=8,
            color=COLORS[name],
            label=DISPLAY_NAMES[name] if show_legend else None,
        )
    ax.set_xticks(x, [tier.title() for tier in TIER_ORDER], fontsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)
    ax.set_yticks([0.5, 1.0])
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.9, alpha=0.8)
    ax.set_ylim(y_min, 1.02)


def write_latex(path: Path, pdf_path: Path) -> None:
    rel_pdf = os.path.relpath(pdf_path, start=path.parent)
    lines = [
        r"\begin{figure}[t]",
        r"\centering",
        rf"\includegraphics[width=\linewidth]{{{rel_pdf}}}",
        r"\caption{Tier-wise accuracy for Background Noise and Voice Pitch.}",
        r"\label{fig:bgnoise-pitch-tier-accuracy}",
        r"\end{figure}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    plt.rcParams.update(
        {
            "font.size": args.font_size,
            "axes.titlesize": args.font_size,
            "axes.labelsize": args.font_size,
            "legend.fontsize": args.font_size - 1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    data = load_csv(args.input_csv)
    categories = args.categories

    fig, axes = plt.subplots(1, 2, figsize=(args.width, args.height), sharey=True)
    for idx, (ax, category) in enumerate(zip(axes, categories)):
        draw_lines(
            ax,
            category_series(data, category),
            font_size=args.font_size,
            show_legend=(idx == 0),
            y_min=args.y_min,
        )
        ax.set_title(DISPLAY_CATEGORY.get(category, category.replace("_", " ").title()), pad=8)
        if idx == 0:
            ax.set_ylabel("Accuracy")
        else:
            ax.set_ylabel("")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.06),
        handlelength=2.0,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.21, top=0.72, wspace=0.14)

    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_pdf, dpi=args.dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    write_latex(args.out_tex, args.out_pdf)
    print(f"wrote PDF: {args.out_pdf}")
    print(f"wrote LaTeX: {args.out_tex}")


if __name__ == "__main__":
    main()

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np

CATEGORY_ORDER = [
    "volume",
    "speaking_speed",
    "voice_pitch",
    "background_noise",
    "audio_distortion",
    "audio_roughness",
    "voice_clarity",
    "voice_vibration",
    "echo",
]
TIER_ORDER = ["low", "mid", "high"]
TRAJECTORY_ORDER = ["linear", "quad_first_flat", "quad_last_flat", "jump"]
LABEL_ORDER = ["increase", "decrease"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize manual annotation results by category, tier, and trajectory.",
    )
    parser.add_argument(
        "--annotations-root",
        type=Path,
        default=Path("annotation_tool/data/results"),
        help="Root directory containing annotation JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("annotation_tool/data/analysis/annotation_results_summary"),
        help="Directory for summary tables and plots.",
    )
    return parser.parse_args()


def load_rows(annotations_root: Path) -> list[dict]:
    rows: list[dict] = []
    for jsonl_path in sorted(annotations_root.glob("*/*/*.jsonl")):
        with jsonl_path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                row["is_correct"] = bool(row["is_correct"])
                row["response_time_ms"] = float(row["response_time_ms"])
                row["replay_count"] = float(row["replay_count"])
                rows.append(row)
    if not rows:
        raise SystemExit(f"No annotation rows found under {annotations_root}")
    return rows


def group_rows(rows: list[dict], *keys: str) -> dict[tuple, list[dict]]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    return grouped


def category_summary(rows: list[dict]) -> list[dict]:
    grouped = group_rows(rows, "category")
    summaries: list[dict] = []
    for category in CATEGORY_ORDER:
        group = grouped.get((category,), [])
        if not group:
            continue
        label_counts = defaultdict(int)
        for row in group:
            label_counts[row["annotated_label"]] += 1
        total = len(group)
        summaries.append(
            {
                "category": category,
                "total_annotations": total,
                "unique_annotators": len({row["annotator_id"] for row in group}),
                "accuracy": mean(1.0 if row["is_correct"] else 0.0 for row in group),
                "avg_response_time_ms": mean(row["response_time_ms"] for row in group),
                "avg_replay_count": mean(row["replay_count"] for row in group),
                "increase_rate": label_counts["increase"] / total,
                "decrease_rate": label_counts["decrease"] / total,
            }
        )
    return summaries


def grouped_accuracy(
    rows: list[dict],
    group_key: str,
    group_values: list[str],
) -> dict[str, dict[str, float]]:
    grouped = group_rows(rows, "category", group_key)
    table: dict[str, dict[str, float]] = {}
    for category in CATEGORY_ORDER:
        table[category] = {}
        for value in group_values:
            group = grouped.get((category, value), [])
            if group:
                table[category][value] = mean(1.0 if row["is_correct"] else 0.0 for row in group)
            else:
                table[category][value] = float("nan")
    return table


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_matrix_csv(path: Path, columns: list[str], matrix: dict[str, dict[str, float]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["category", *columns])
        for category in CATEGORY_ORDER:
            if category not in matrix:
                continue
            row = [category]
            for column in columns:
                value = matrix[category][column]
                row.append("" if np.isnan(value) else f"{value * 100.0:.2f}")
            writer.writerow(row)


def to_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    separator = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def plot_category_accuracy(summary_rows: list[dict], output_path: Path) -> None:
    ordered = sorted(summary_rows, key=lambda row: row["accuracy"], reverse=True)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(
        [row["category"] for row in ordered],
        [row["accuracy"] * 100.0 for row in ordered],
        color="#35618f",
    )
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Mean Accuracy by Category")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    plt.xticks(rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_heatmap(
    matrix: dict[str, dict[str, float]],
    column_order: list[str],
    title: str,
    output_path: Path,
) -> None:
    row_labels = [category for category in CATEGORY_ORDER if category in matrix]
    values = np.array(
        [[matrix[category][column] for column in column_order] for category in row_labels],
    )
    values_pct = values * 100.0

    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    image = ax.imshow(values_pct, aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)
    ax.set_title(title)
    ax.set_xticks(range(len(column_order)))
    ax.set_xticklabels(column_order)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    for row_idx in range(values_pct.shape[0]):
        for col_idx in range(values_pct.shape[1]):
            value = values_pct[row_idx, col_idx]
            text = "-" if np.isnan(value) else f"{value:.1f}"
            text_color = "black" if np.isnan(value) or value < 55 else "white"
            ax.text(col_idx, row_idx, text, ha="center", va="center", color=text_color, fontsize=9)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Accuracy (%)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_label_proportions(summary_rows: list[dict], output_path: Path) -> None:
    ordered = [row for row in summary_rows if row["category"] in CATEGORY_ORDER]
    categories = [row["category"] for row in ordered]
    increase = [row["increase_rate"] * 100.0 for row in ordered]
    decrease = [row["decrease_rate"] * 100.0 for row in ordered]

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.barh(categories, increase, color="#4c78a8", label="Answered increase")
    ax.barh(categories, decrease, left=increase, color="#f58518", label="Answered decrease")
    ax.set_xlabel("Share of answers (%)")
    ax.set_xlim(0, 100)
    ax.set_title("Annotated Label Proportions by Category")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_report(
    output_path: Path,
    summary_rows: list[dict],
    tier_matrix: dict[str, dict[str, float]],
    trajectory_matrix: dict[str, dict[str, float]],
) -> None:
    summary_table = to_markdown_table(
        [
            "Category",
            "Annotations",
            "Annotators",
            "Accuracy",
            "Avg response time (ms)",
            "Avg replay count",
            "Answered increase",
            "Answered decrease",
        ],
        [
            [
                row["category"],
                str(row["total_annotations"]),
                str(row["unique_annotators"]),
                f"{row['accuracy'] * 100.0:.1f}%",
                f"{row['avg_response_time_ms']:.1f}",
                f"{row['avg_replay_count']:.2f}",
                f"{row['increase_rate'] * 100.0:.1f}%",
                f"{row['decrease_rate'] * 100.0:.1f}%",
            ]
            for row in summary_rows
        ],
    )

    tier_table = to_markdown_table(
        ["Category", *TIER_ORDER],
        [
            [
                category,
                *[
                    "-"
                    if np.isnan(tier_matrix[category][tier])
                    else f"{tier_matrix[category][tier] * 100.0:.1f}%"
                    for tier in TIER_ORDER
                ],
            ]
            for category in CATEGORY_ORDER
        ],
    )

    trajectory_table = to_markdown_table(
        ["Category", *TRAJECTORY_ORDER],
        [
            [
                category,
                *[
                    "-"
                    if np.isnan(trajectory_matrix[category][trajectory])
                    else f"{trajectory_matrix[category][trajectory] * 100.0:.1f}%"
                    for trajectory in TRAJECTORY_ORDER
                ],
            ]
            for category in CATEGORY_ORDER
        ],
    )

    output_path.write_text(
        "# Annotation Results Summary\n\n"
        "## Mean Accuracy by Category\n\n"
        f"{summary_table}\n\n"
        "![Mean accuracy by category](./category_accuracy_comparison.png)\n\n"
        "## Accuracy by Tier\n\n"
        f"{tier_table}\n\n"
        "![Accuracy by tier](./tier_accuracy_heatmap.png)\n\n"
        "## Accuracy by Trajectory\n\n"
        f"{trajectory_table}\n\n"
        "![Accuracy by trajectory](./trajectory_accuracy_heatmap.png)\n\n"
        "## Annotated Label Proportions\n\n"
        "The `increase` / `decrease` answer ratios are included in the category table above.\n\n"
        "![Annotated label proportions](./annotated_label_proportions.png)\n",
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.annotations_root)
    summary_rows = category_summary(rows)
    tier_matrix = grouped_accuracy(rows, "difficulty", TIER_ORDER)
    trajectory_matrix = grouped_accuracy(rows, "trajectory", TRAJECTORY_ORDER)

    write_csv(
        args.output_dir / "category_accuracy_comparison.csv",
        [
            "category",
            "total_annotations",
            "unique_annotators",
            "accuracy",
            "avg_response_time_ms",
            "avg_replay_count",
            "increase_rate",
            "decrease_rate",
        ],
        summary_rows,
    )
    write_matrix_csv(args.output_dir / "tier_accuracy_by_category.csv", TIER_ORDER, tier_matrix)
    write_matrix_csv(
        args.output_dir / "trajectory_accuracy_by_category.csv",
        TRAJECTORY_ORDER,
        trajectory_matrix,
    )
    write_csv(
        args.output_dir / "annotated_label_proportions_by_category.csv",
        ["category", "increase_rate", "decrease_rate"],
        [
            {
                "category": row["category"],
                "increase_rate": row["increase_rate"],
                "decrease_rate": row["decrease_rate"],
            }
            for row in summary_rows
        ],
    )

    plot_category_accuracy(summary_rows, args.output_dir / "category_accuracy_comparison.png")
    plot_heatmap(
        tier_matrix,
        TIER_ORDER,
        "Accuracy by Category and Tier",
        args.output_dir / "tier_accuracy_heatmap.png",
    )
    plot_heatmap(
        trajectory_matrix,
        TRAJECTORY_ORDER,
        "Accuracy by Category and Trajectory",
        args.output_dir / "trajectory_accuracy_heatmap.png",
    )
    plot_label_proportions(summary_rows, args.output_dir / "annotated_label_proportions.png")

    write_report(
        args.output_dir / "REPORT.md",
        summary_rows,
        tier_matrix,
        trajectory_matrix,
    )

    print(f"Wrote annotation summary to {args.output_dir}")


if __name__ == "__main__":
    main()

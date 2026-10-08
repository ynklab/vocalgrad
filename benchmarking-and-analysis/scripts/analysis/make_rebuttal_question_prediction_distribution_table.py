"""Create all-augmentation-average prediction distributions for question formats."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FORMATS = {
    "Binary variation (yes/no)": {
        "raw_root": REPO_ROOT
        / "outputs/raw/rebuttal/binary_variation_question/default",
        "predictions": ("yes", "no", "unparsed"),
        "display_predictions": ("yes", "no"),
        "out_name": "binary_prediction_distribution_by_clip_type.md",
    },
    "Ternary variation (increase/decrease/constant)": {
        "raw_root": REPO_ROOT / "outputs/raw/rebuttal/ternary_question/default",
        "predictions": ("increase", "decrease", "constant", "unparsed"),
        "display_predictions": ("increase", "decrease", "constant", "unparsed"),
        "out_name": "ternary_prediction_distribution_by_clip_type.md",
    },
}
GOLD_GROUPS = ("Increasing", "Decreasing", "Source")


def _gold_group(row: dict[str, object]) -> str:
    if row.get("sample_origin") == "source":
        return "Source"
    direction = str((row.get("metadata") or {}).get("direction", "")).lower()
    if direction == "up":
        return "Increasing"
    if direction == "down":
        return "Decreasing"
    raise ValueError(f"cannot determine clip type for {row.get('benchmark_clip_id')}")


def _read_category_rows(raw_root: Path) -> dict[str, list[dict[str, object]]]:
    categories: dict[str, list[dict[str, object]]] = {}
    for category_dir in sorted(
        path
        for path in raw_root.iterdir()
        if path.is_dir() and not path.name.startswith("_")
    ):
        paths = sorted(category_dir.glob("*.jsonl"))
        if len(paths) != 1:
            raise ValueError(
                f"expected one model JSONL in {category_dir}, found {len(paths)}"
            )
        with paths[0].open(encoding="utf-8") as handle:
            categories[category_dir.name] = [
                json.loads(line) for line in handle if line.strip()
            ]
    if not categories:
        raise FileNotFoundError(f"no category JSONL files found under {raw_root}")
    return categories


def _macro_prediction_rates(
    raw_root: Path, predictions: tuple[str, ...]
) -> dict[str, dict[str, float]]:
    """Average each conditional prediction rate equally over augmentations."""
    per_augmentation: list[dict[str, dict[str, float]]] = []
    for rows in _read_category_rows(raw_root).values():
        counts: dict[str, dict[str, int]] = {
            group: defaultdict(int) for group in GOLD_GROUPS
        }
        totals: defaultdict[str, int] = defaultdict(int)
        for row in rows:
            group = _gold_group(row)
            prediction = str(row.get("prediction_label") or "unparsed").lower()
            if prediction not in predictions:
                prediction = "unparsed"
            counts[group][prediction] += 1
            totals[group] += 1
        if missing := [group for group in GOLD_GROUPS if not totals[group]]:
            raise ValueError(f"augmentation is missing clip groups: {missing}")
        per_augmentation.append(
            {
                prediction: {
                    group: counts[group][prediction] / totals[group]
                    for group in GOLD_GROUPS
                }
                for prediction in predictions
            }
        )

    return {
        prediction: {
            group: sum(values[prediction][group] for values in per_augmentation)
            / len(per_augmentation)
            for group in GOLD_GROUPS
        }
        for prediction in predictions
    }


def _write_markdown(
    path: Path,
    rates: dict[str, dict[str, float]],
    predictions: tuple[str, ...],
) -> None:
    lines = [
        "| Prediction | Increasing | Decreasing | Source |",
        "|---|---:|---:|---:|",
    ]
    for prediction in predictions:
        group_rates = rates[prediction]
        lines.append(
            f"| {prediction} | "
            + " | ".join(f"{group_rates[group] * 100:.2f}%" for group in GOLD_GROUPS)
            + " |"
        )
    lines.append("")
    lines.append(
        "Each cell is the conditional prediction rate within that clip group, averaged equally over augmentations."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a question-format prediction-distribution Markdown table."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT
        / "outputs/analysis/rebuttal/question_format_comparison/default",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, spec in FORMATS.items():
        out_path = args.out_dir / spec["out_name"]
        rates = _macro_prediction_rates(spec["raw_root"], spec["predictions"])
        _write_markdown(out_path, rates, spec["display_predictions"])
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

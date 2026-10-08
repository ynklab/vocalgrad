"""Create a Markdown table of augmentation-wise macro-F1 by format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUTS = {
    "Binary variation (yes/no)": REPO_ROOT
    / "outputs/analysis/rebuttal/binary_variation_question/default",
    "Ternary variation (increase/decrease/constant)": REPO_ROOT
    / "outputs/analysis/rebuttal/ternary_question/default",
}


def _load_results(root: Path) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for path in sorted(root.glob("*/*.json")):
        with path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        results[str(result["category"])] = result
    if not results:
        raise FileNotFoundError(
            f"no per-augmentation result JSON files found under {root}"
        )
    return results


def _macro_f1_by_augmentation(root: Path) -> dict[str, float]:
    """Compute equal-class-weight macro-F1 per augmentation.

    Unparsed answers have no confusion-matrix row, but remain false negatives
    through the complete gold-label count used for each class recall.
    """
    values: dict[str, float] = {}
    for augmentation, result in _load_results(root).items():
        gold_counts = result["gold_label_counts"]
        confusion = result["confusion_matrix"]["table"]
        class_f1: list[float] = []
        for label, gold_count in gold_counts.items():
            # The outer confusion-matrix key is the predicted label and the
            # inner key is the gold label, so this is its diagonal entry.
            true_positive = int(confusion.get(label, {}).get(label, 0))
            predicted_positive = sum(
                int(gold_count) for gold_count in confusion.get(label, {}).values()
            )
            precision = (
                true_positive / predicted_positive if predicted_positive else 0.0
            )
            recall = true_positive / int(gold_count) if gold_count else 0.0
            class_f1.append(
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
        values[augmentation] = sum(class_f1) / len(class_f1) if class_f1 else 0.0
    return values


def _write_markdown(path: Path, rows: list[tuple[str, dict[str, float]]]) -> None:
    augmentations = sorted(
        {augmentation for _, values in rows for augmentation in values}
    )
    lines = [
        "| Problem format | " + " | ".join(augmentations) + " |",
        "|---|" + "|".join("---:" for _ in augmentations) + "|",
    ]
    for problem_format, values_by_augmentation in rows:
        values = [
            "--"
            if augmentation not in values_by_augmentation
            else f"{values_by_augmentation[augmentation] * 100:.2f}%"
            for augmentation in augmentations
        ]
        lines.append("| " + problem_format + " | " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create one augmentation-wise macro-F1 Markdown table."
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=REPO_ROOT
        / "outputs/analysis/rebuttal/question_format_comparison/default/macro_f1_by_augmentation.md",
    )
    args = parser.parse_args()
    rows = [
        (name, _macro_f1_by_augmentation(root)) for name, root in DEFAULT_INPUTS.items()
    ]
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown(args.out_path, rows)
    print(f"wrote {args.out_path}")


if __name__ == "__main__":
    main()

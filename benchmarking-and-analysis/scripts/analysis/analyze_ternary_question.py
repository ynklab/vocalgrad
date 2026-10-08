#!/usr/bin/env python3
"""Summarize ternary VocalGrad question results by class, direction, and tier."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


LABELS = ("increase", "decrease", "constant")
DEFAULT_RAW_ROOT = Path("outputs/raw/rebuttal/ternary_question/default")
DEFAULT_OUT_ROOT = Path("outputs/analysis/rebuttal/ternary_question/default")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report ternary class/macro accuracy and augmented direction/tier accuracy. Unparsed outputs count incorrect."
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    return parser.parse_args()


def read_rows(root: Path) -> Iterable[tuple[str, str, dict[str, Any]]]:
    if not root.exists():
        raise FileNotFoundError(f"raw root not found: {root}")
    for category_dir in sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith("_")
    ):
        for path in sorted(category_dir.glob("*.jsonl")):
            with path.open("r", encoding="utf-8") as handle:
                for number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        yield path.stem, category_dir.name, json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"invalid JSONL at {path}:{number}") from exc


def accuracy(rows: list[dict[str, Any]]) -> float | None:
    return (
        sum(row.get("is_correct") is True for row in rows) / len(rows) if rows else None
    )


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_latex(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Ternary question accuracy by model and VocalGrad augmentation class. Increase and decrease are augmented clips; constant is the unmodified source clip. Unparsed outputs count as incorrect.}",
        r"\label{tab:ternary-question-macro-accuracy}",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Model & Class & Inc. & Dec. & Const. & Macro & $n_{inc}$ & $n_{dec}$ & $n_{const}$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['model']} & {row['augmentation_class']} & {float(row['accuracy_increase']) * 100:.1f} & "
            f"{float(row['accuracy_decrease']) * 100:.1f} & {float(row['accuracy_constant']) * 100:.1f} & "
            f"{float(row['macro_accuracy_three_class']) * 100:.1f} & {row['n_increase']} & {row['n_decrease']} & {row['n_constant']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    by_class: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_direction: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    by_tier: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for model, category, row in read_rows(args.raw_root):
        by_class[(model, category)].append(row)
        metadata = row.get("metadata") or {}
        if row.get("sample_origin") == "augmented":
            direction = {"up": "increase", "down": "decrease"}.get(
                str(metadata.get("direction", "")).lower(), "unknown"
            )
            tier = str(metadata.get("tier", "unknown"))
        elif row.get("sample_origin") == "source":
            direction = "constant"
            tier = "source"
        else:
            continue
        by_direction[(model, category, direction)].append(row)
        by_tier[(model, category, tier, direction)].append(row)

    class_rows: list[dict[str, object]] = []
    for (model, category), rows in sorted(by_class.items()):
        subsets = {
            label: [row for row in rows if row.get("gold_label") == label]
            for label in LABELS
        }
        metrics = {label: accuracy(subsets[label]) for label in LABELS}
        macro = (
            sum(metrics.values()) / len(LABELS)
            if all(value is not None for value in metrics.values())
            else None
        )
        class_rows.append(
            {
                "model": model,
                "augmentation_class": category,
                **{f"n_{label}": len(subsets[label]) for label in LABELS},
                **{f"accuracy_{label}": fmt(metrics[label]) for label in LABELS},
                "macro_accuracy_three_class": fmt(macro),
                "n_total": len(rows),
                "strict_accuracy_all": fmt(accuracy(rows)),
            }
        )

    direction_rows = [
        {
            "model": model,
            "augmentation_class": category,
            "direction": direction,
            "n": len(rows),
            "accuracy": fmt(accuracy(rows)),
        }
        for (model, category, direction), rows in sorted(by_direction.items())
    ]
    lookup = {
        (row["model"], row["augmentation_class"], row["direction"]): row
        for row in direction_rows
    }
    difference_rows: list[dict[str, object]] = []
    for model, category in sorted(by_class):
        increase = lookup.get((model, category, "increase"))
        decrease = lookup.get((model, category, "decrease"))
        constant = lookup.get((model, category, "constant"))
        inc_acc = (
            float(increase["accuracy"]) if increase and increase["accuracy"] else None
        )
        dec_acc = (
            float(decrease["accuracy"]) if decrease and decrease["accuracy"] else None
        )
        constant_acc = (
            float(constant["accuracy"]) if constant and constant["accuracy"] else None
        )
        difference_rows.append(
            {
                "model": model,
                "augmentation_class": category,
                "n_increase": increase["n"] if increase else 0,
                "accuracy_increase": fmt(inc_acc),
                "n_decrease": decrease["n"] if decrease else 0,
                "accuracy_decrease": fmt(dec_acc),
                "n_constant": constant["n"] if constant else 0,
                "accuracy_constant": fmt(constant_acc),
                "increase_minus_decrease": fmt(
                    None if inc_acc is None or dec_acc is None else inc_acc - dec_acc
                ),
            }
        )
    tier_rows = [
        {
            "model": model,
            "augmentation_class": category,
            "tier": tier,
            "direction": direction,
            "n": len(rows),
            "accuracy": fmt(accuracy(rows)),
        }
        for (model, category, tier, direction), rows in sorted(by_tier.items())
    ]

    class_fields = [
        "model",
        "augmentation_class",
        "n_increase",
        "accuracy_increase",
        "n_decrease",
        "accuracy_decrease",
        "n_constant",
        "accuracy_constant",
        "macro_accuracy_three_class",
        "n_total",
        "strict_accuracy_all",
    ]
    write_csv(
        args.out_root / "ternary_class_macro_accuracy.csv", class_fields, class_rows
    )
    write_csv(
        args.out_root / "ternary_augmented_direction_accuracy.csv",
        ["model", "augmentation_class", "direction", "n", "accuracy"],
        direction_rows,
    )
    write_csv(
        args.out_root / "ternary_augmented_direction_difference.csv",
        list(difference_rows[0])
        if difference_rows
        else ["model", "augmentation_class"],
        difference_rows,
    )
    write_csv(
        args.out_root / "ternary_augmented_direction_tier_accuracy.csv",
        ["model", "augmentation_class", "tier", "direction", "n", "accuracy"],
        tier_rows,
    )
    if class_rows:
        write_latex(args.out_root / "ternary_class_macro_accuracy.tex", class_rows)
    for name in (
        "ternary_class_macro_accuracy.csv",
        "ternary_augmented_direction_accuracy.csv",
        "ternary_augmented_direction_difference.csv",
        "ternary_augmented_direction_tier_accuracy.csv",
    ):
        print(f"wrote {args.out_root / name}")


if __name__ == "__main__":
    main()

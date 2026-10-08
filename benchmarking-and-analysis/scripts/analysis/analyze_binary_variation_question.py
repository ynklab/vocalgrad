#!/usr/bin/env python3
"""Summarize the binary VocalGrad variation-question rebuttal experiment."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_RAW_ROOT = Path("outputs/raw/rebuttal/binary_variation_question/default")
DEFAULT_OUT_ROOT = Path("outputs/analysis/rebuttal/binary_variation_question/default")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report Yes/No macro accuracy and direction/tier accuracy for the "
            "binary VocalGrad variation question. Unparsed responses count incorrect."
        )
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    return parser.parse_args()


def read_rows(raw_root: Path) -> Iterable[tuple[str, str, dict[str, Any]]]:
    if not raw_root.exists():
        raise FileNotFoundError(f"raw root not found: {raw_root}")
    for category_dir in sorted(
        path
        for path in raw_root.iterdir()
        if path.is_dir() and not path.name.startswith("_")
    ):
        for path in sorted(category_dir.glob("*.jsonl")):
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        yield path.stem, category_dir.name, json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"invalid JSONL at {path}:{line_number}"
                        ) from exc


def accuracy(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return sum(row.get("is_correct") is True for row in rows) / len(rows)


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
        r"\caption{Binary variation-question accuracy by model and VocalGrad augmentation class. Yes denotes augmented clips and No denotes source clips; unparsed outputs count as incorrect.}",
        r"\label{tab:binary-variation-macro-accuracy}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Model & Class & Yes acc. & No acc. & Macro acc. & $n_{yes}$ & $n_{no}$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['model']} & {row['augmentation_class']} & "
            f"{float(row['accuracy_yes']) * 100:.1f} & {float(row['accuracy_no']) * 100:.1f} & "
            f"{float(row['macro_accuracy_yes_no']) * 100:.1f} & {row['n_yes']} & {row['n_no']} \\\\"
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
        if row.get("sample_origin") != "augmented":
            continue
        metadata = row.get("metadata") or {}
        direction = {"up": "increase", "down": "decrease"}.get(
            str(metadata.get("direction", "")).lower(), "unknown"
        )
        tier = str(metadata.get("tier", "unknown"))
        by_direction[(model, category, direction)].append(row)
        by_tier[(model, category, tier, direction)].append(row)

    macro_rows: list[dict[str, object]] = []
    for (model, category), rows in sorted(by_class.items()):
        yes_rows = [row for row in rows if row.get("gold_label") == "yes"]
        no_rows = [row for row in rows if row.get("gold_label") == "no"]
        yes_accuracy, no_accuracy = accuracy(yes_rows), accuracy(no_rows)
        if yes_accuracy is None or no_accuracy is None:
            macro_accuracy = None
        else:
            macro_accuracy = (yes_accuracy + no_accuracy) / 2
        macro_rows.append(
            {
                "model": model,
                "augmentation_class": category,
                "n_yes": len(yes_rows),
                "accuracy_yes": fmt(yes_accuracy),
                "n_no": len(no_rows),
                "accuracy_no": fmt(no_accuracy),
                "macro_accuracy_yes_no": fmt(macro_accuracy),
                "n_total": len(rows),
                "strict_accuracy_all": fmt(accuracy(rows)),
            }
        )

    direction_rows: list[dict[str, object]] = []
    for (model, category, direction), rows in sorted(by_direction.items()):
        direction_rows.append(
            {
                "model": model,
                "augmentation_class": category,
                "direction": direction,
                "n": len(rows),
                "accuracy": fmt(accuracy(rows)),
            }
        )

    direction_lookup = {
        (row["model"], row["augmentation_class"], row["direction"]): row
        for row in direction_rows
    }
    difference_rows: list[dict[str, object]] = []
    for model, category in sorted(by_class):
        increase = direction_lookup.get((model, category, "increase"))
        decrease = direction_lookup.get((model, category, "decrease"))
        increase_accuracy = (
            float(increase["accuracy"]) if increase and increase["accuracy"] else None
        )
        decrease_accuracy = (
            float(decrease["accuracy"]) if decrease and decrease["accuracy"] else None
        )
        difference_rows.append(
            {
                "model": model,
                "augmentation_class": category,
                "n_increase": increase["n"] if increase else 0,
                "accuracy_increase": fmt(increase_accuracy),
                "n_decrease": decrease["n"] if decrease else 0,
                "accuracy_decrease": fmt(decrease_accuracy),
                "increase_minus_decrease": fmt(
                    None
                    if increase_accuracy is None or decrease_accuracy is None
                    else increase_accuracy - decrease_accuracy
                ),
            }
        )

    tier_rows: list[dict[str, object]] = []
    for (model, category, tier, direction), rows in sorted(by_tier.items()):
        tier_rows.append(
            {
                "model": model,
                "augmentation_class": category,
                "tier": tier,
                "direction": direction,
                "n": len(rows),
                "accuracy": fmt(accuracy(rows)),
            }
        )

    write_csv(
        args.out_root / "yes_no_macro_accuracy.csv",
        list(macro_rows[0]) if macro_rows else ["model", "augmentation_class"],
        macro_rows,
    )
    write_csv(
        args.out_root / "augmented_direction_accuracy.csv",
        ["model", "augmentation_class", "direction", "n", "accuracy"],
        direction_rows,
    )
    write_csv(
        args.out_root / "augmented_direction_difference.csv",
        list(difference_rows[0])
        if difference_rows
        else ["model", "augmentation_class"],
        difference_rows,
    )
    write_csv(
        args.out_root / "augmented_direction_tier_accuracy.csv",
        ["model", "augmentation_class", "tier", "direction", "n", "accuracy"],
        tier_rows,
    )
    if macro_rows:
        write_latex(args.out_root / "yes_no_macro_accuracy.tex", macro_rows)

    print(f"wrote {args.out_root / 'yes_no_macro_accuracy.csv'}")
    print(f"wrote {args.out_root / 'augmented_direction_accuracy.csv'}")
    print(f"wrote {args.out_root / 'augmented_direction_difference.csv'}")
    print(f"wrote {args.out_root / 'augmented_direction_tier_accuracy.csv'}")


if __name__ == "__main__":
    main()

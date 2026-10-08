from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


from .direction_evaluation import (
    SCHEMA_VERSION, binary_metrics as _binary_metrics, read_rows, validate_rows,
)


def _group_metrics(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        md = row.get("metadata") or {}
        grouped[str(md.get(key, "unknown"))].append(row)
    return {group: _binary_metrics(group_rows) for group, group_rows in grouped.items()}


def summarize_rows(rows: list[dict[str, Any]], input_path: str) -> dict[str, Any]:
    validate_rows(rows)
    raw_response_counter: Counter[str] = Counter()
    category_counter: Counter[str] = Counter()

    for row in rows:
        raw_response_counter[str(row.get("raw_response", ""))] += 1
        category_counter[str(row.get("category", "unknown"))] += 1

    overall = _binary_metrics(rows)
    by_category = defaultdict(list)
    for row in rows:
        by_category[str(row.get("category", "unknown"))].append(row)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "parser_version": "synonym-unique-direction-v1" if any(r.get("answer_words") for r in rows) else "main-unique-direction-v1",
        "input": input_path,
        "total_rows": len(rows),
        "by_category_count": dict(category_counter),
        "overall": overall,
        "by_category": {
            category: _binary_metrics(category_rows)
            for category, category_rows in by_category.items()
        },
        "by_tier": _group_metrics(rows, "tier"),
        "by_curve": _group_metrics(rows, "curve"),
        "by_direction": _group_metrics(rows, "direction"),
        "raw_response_top20": dict(raw_response_counter.most_common(20)),
    }

    for key in ("actual_attribute", "prompt_attribute", "model_id", "backend", "prompt_variant", "answer_words"):
        values = [r.get(key) for r in rows]
        if values and values[0] is not None and all(value == values[0] for value in values):
            summary[key] = values[0]
    if "actual_attribute" in summary and "prompt_attribute" in summary:
        summary["evaluation_type"] = "vocalgrad_cross_attribute"
    return summary


def summarize_vocalgrad_file(input_path: Path, out_path: Path) -> dict[str, Any]:
    rows = read_rows(input_path)

    summary = summarize_rows(rows, str(input_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize VocalGrad single-audio results. "
            "Positive class is `increase`."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/analysis/vocalgrad_summary.json"),
    )
    args = parser.parse_args()

    summary = summarize_vocalgrad_file(args.input, args.out)

    print(f"wrote summary to {args.out}")
    overall = summary["overall"]
    print(
        "overall: "
        f"acc={overall['accuracy']:.4f} "
        f"prec_increase={overall['precision_increase']:.4f} "
        f"recall_increase={overall['recall_increase']:.4f} "
        f"f1_increase={overall['f1_increase']:.4f}"
    )
    cm = overall["confusion_matrix"]["counts"]
    print(
        "confusion_matrix(increase positive): "
        f"TP={cm['tp_increase_increase']} FP={cm['fp_increase_decrease']} "
        f"FN={cm['fn_decrease_increase']} TN={cm['tn_decrease_decrease']}"
    )


if __name__ == "__main__":
    main()

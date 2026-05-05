from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _normalize_label(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    lowered = value.strip().lower()
    if lowered in {"increase", "decrease"}:
        return lowered
    if lowered in {"up", "upward"}:
        return "increase"
    if lowered in {"down", "downward"}:
        return "decrease"
    return None


def _binary_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    total = 0

    for row in rows:
        pred = _normalize_label(row.get("prediction_label"))
        gold = _normalize_label(row.get("gold_label"))
        if pred is None or gold is None:
            continue

        total += 1
        pred_pos = pred == "increase"
        gold_pos = gold == "increase"

        if pred_pos and gold_pos:
            tp += 1
        elif pred_pos and not gold_pos:
            fp += 1
        elif (not pred_pos) and gold_pos:
            fn += 1
        else:
            tn += 1

    accuracy = _safe_div(tp + tn, total)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    return {
        "total_evaluable": total,
        "accuracy": accuracy,
        "precision_increase": precision,
        "recall_increase": recall,
        "f1_increase": f1,
        "confusion_matrix": {
            "labels": ["increase", "decrease"],
            "counts": {
                "tp_increase_increase": tp,
                "fp_increase_decrease": fp,
                "fn_decrease_increase": fn,
                "tn_decrease_decrease": tn,
            },
            "table": {
                "pred_increase": {"gold_increase": tp, "gold_decrease": fp},
                "pred_decrease": {"gold_increase": fn, "gold_decrease": tn},
            },
        },
    }


def _group_metrics(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        md = row.get("metadata") or {}
        grouped[str(md.get(key, "unknown"))].append(row)
    return {group: _binary_metrics(group_rows) for group, group_rows in grouped.items()}


def summarize_rows(rows: list[dict[str, Any]], input_path: str) -> dict[str, Any]:
    raw_response_counter: Counter[str] = Counter()
    category_counter: Counter[str] = Counter()

    for row in rows:
        raw_response_counter[str(row.get("raw_response", ""))] += 1
        category_counter[str(row.get("category", "unknown"))] += 1

    overall = _binary_metrics(rows)
    by_category = defaultdict(list)
    for row in rows:
        by_category[str(row.get("category", "unknown"))].append(row)

    return {
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


def summarize_vocalgrad_file(input_path: Path, out_path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))

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

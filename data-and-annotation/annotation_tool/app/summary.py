from __future__ import annotations

from collections import defaultdict
from typing import Any


def _group_accuracy(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)

    out: dict[str, dict[str, float | int]] = {}
    for key, group_rows in sorted(grouped.items()):
        total = len(group_rows)
        correct = sum(1 for row in group_rows if bool(row["is_correct"]))
        out[key] = {
            "total": total,
            "correct": correct,
            "accuracy": (correct / total) if total else 0.0,
        }
    return out


def compute_summary(rows: list[dict[str, Any]], annotator_id: str | None = None) -> dict[str, Any]:
    total = len(rows)
    correct = sum(1 for row in rows if bool(row["is_correct"]))

    avg_replay = (
        sum(int(row["replay_count"]) for row in rows) / total
        if total
        else 0.0
    )
    avg_response_time = (
        sum(int(row["response_time_ms"]) for row in rows) / total
        if total
        else 0.0
    )

    return {
        "annotator_id": annotator_id,
        "total_completed_items": total,
        "overall_accuracy": (correct / total) if total else 0.0,
        "accuracy_by_category": _group_accuracy(rows, "category"),
        "accuracy_by_difficulty": _group_accuracy(rows, "difficulty"),
        "accuracy_by_trajectory": _group_accuracy(rows, "trajectory"),
        "average_replay_count": avg_replay,
        "average_response_time_ms": avg_response_time,
    }

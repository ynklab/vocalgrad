from __future__ import annotations

import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from .config import SUPPORTED_CATEGORIES
from .results_store import manifest_results_dir

VALID_LABELS = ("increase", "decrease")


def _load_all_results(category: str) -> dict[str, list[dict[str, Any]]]:
    results_dir = manifest_results_dir(category)
    out: dict[str, list[dict[str, Any]]] = {}
    if not results_dir.exists():
        return out

    for path in sorted(results_dir.glob("*.jsonl")):
        annotator_id = path.stem
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    rows.append(json.loads(stripped))
        out[annotator_id] = rows
    return out


def _pairwise_percent_agreement(
    labels_a: dict[str, str],
    labels_b: dict[str, str],
) -> tuple[float | None, int]:
    overlap = sorted(set(labels_a) & set(labels_b))
    if not overlap:
        return None, 0
    matches = sum(1 for item_id in overlap if labels_a[item_id] == labels_b[item_id])
    return matches / len(overlap), len(overlap)


def _cohen_kappa(labels_a: dict[str, str], labels_b: dict[str, str]) -> float | None:
    overlap = sorted(set(labels_a) & set(labels_b))
    if not overlap:
        return None

    observed = sum(
        1 for item_id in overlap if labels_a[item_id] == labels_b[item_id]
    ) / len(overlap)
    counts_a = Counter(labels_a[item_id] for item_id in overlap)
    counts_b = Counter(labels_b[item_id] for item_id in overlap)
    expected = sum(
        (counts_a[label] / len(overlap)) * (counts_b[label] / len(overlap))
        for label in VALID_LABELS
    )
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def compute_agreement_summary(category: str) -> dict[str, Any]:
    rows_by_annotator = _load_all_results(category)
    annotators = sorted(rows_by_annotator)
    item_to_labels: dict[str, dict[str, str]] = defaultdict(dict)

    for annotator_id, rows in rows_by_annotator.items():
        for row in rows:
            item_to_labels[str(row["item_id"])][annotator_id] = str(row["annotated_label"])

    item_level: dict[str, dict[str, Any]] = {}
    item_agreements: list[float] = []
    for item_id, labels_by_annotator in sorted(item_to_labels.items()):
        labels = list(labels_by_annotator.values())
        counts = Counter(labels)
        n = len(labels)
        majority_label = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
        if n <= 1:
            agreement = 1.0
        else:
            agreeing_pairs = sum(1 for a, b in combinations(labels, 2) if a == b)
            total_pairs = n * (n - 1) / 2
            agreement = agreeing_pairs / total_pairs
        item_agreements.append(agreement)
        item_level[item_id] = {
            "num_annotators": n,
            "majority_vote_label": majority_label,
            "agreement_rate": agreement,
        }

    pairwise_percent_agreement: dict[str, float] = {}
    pairwise_cohen_kappa: dict[str, float] = {}
    for ann_a, ann_b in combinations(annotators, 2):
        labels_a = {
            str(row["item_id"]): str(row["annotated_label"]) for row in rows_by_annotator[ann_a]
        }
        labels_b = {
            str(row["item_id"]): str(row["annotated_label"]) for row in rows_by_annotator[ann_b]
        }
        agreement, _ = _pairwise_percent_agreement(labels_a, labels_b)
        if agreement is not None:
            pairwise_percent_agreement[f"{ann_a}__{ann_b}"] = agreement
            kappa = _cohen_kappa(labels_a, labels_b)
            if kappa is not None:
                pairwise_cohen_kappa[f"{ann_a}__{ann_b}"] = kappa

    full_overlap_items = 0
    if annotators:
        full_overlap_items = sum(
            1
            for labels_by_annotator in item_to_labels.values()
            if len(labels_by_annotator) == len(annotators)
        )

    return {
        "category": category,
        "results_dir": str(manifest_results_dir(category)),
        "annotators": annotators,
        "num_items_with_any_annotations": len(item_to_labels),
        "num_items_with_full_overlap": full_overlap_items,
        "pairwise_percent_agreement": pairwise_percent_agreement,
        "pairwise_cohen_kappa": pairwise_cohen_kappa,
        "item_level_mean_agreement": (
            sum(item_agreements) / len(item_agreements) if item_agreements else None
        ),
        "item_level": item_level,
    }


def write_agreement_summary_file(category: str) -> Path:
    summary = compute_agreement_summary(category)
    out_path = manifest_results_dir(category) / "agreement_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_path


def write_all_agreement_summaries() -> list[Path]:
    out: list[Path] = []
    for category in SUPPORTED_CATEGORIES:
        out.append(write_agreement_summary_file(category))
    return out


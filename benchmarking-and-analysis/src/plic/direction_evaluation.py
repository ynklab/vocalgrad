"""CPU-only direction parsing and all-clip metrics for the published benchmark.

Metrics are computed from raw_response and gold_label. total_evaluable is
the parsed response count, not the accuracy denominator.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

SCHEMA_VERSION = "vocalgrad-all-clips-v1"
INCREASE_WORDS = set("increase increases increasing rise rises rising upward upwards faster accelerate accelerates accelerating".split())
DECREASE_WORDS = set("decrease decreases decreasing fall falls falling downward downwards slower decline declines declining decelerate decelerates decelerating".split())
MAIN_RE = re.compile(r"\b(increase|decrease|increases|decreases)\b", re.I)
ANSWER_RE = re.compile(r"\b(" + "|".join(sorted(INCREASE_WORDS | DECREASE_WORDS, key=lambda x: (-len(x), x))) + r")\b", re.I)


def parse_labels(raw, paraphrase=False):
    words = (ANSWER_RE if paraphrase else MAIN_RE).findall(str(raw or "").lower())
    return {"increase" if w in INCREASE_WORDS else "decrease" for w in words}


def parse_direction_label(raw):
    labels = parse_labels(raw)
    return next(iter(labels)) if len(labels) == 1 else None


def parse_paraphrase_label(raw):
    labels = parse_labels(raw, True)
    return next(iter(labels)) if len(labels) == 1 else None


def normalize_label(value):
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    return {"up": "increase", "upward": "increase", "increased": "increase",
            "down": "decrease", "downward": "decrease", "decreased": "decrease"}.get(value, value) if value in {"increase", "decrease", "up", "upward", "increased", "down", "downward", "decreased"} else None


def row_paraphrase(row):
    return bool(row.get("answer_words"))


def prediction_for_row(row, paraphrase=None):
    if row.get("error"):
        return None
    return (parse_paraphrase_label if (row_paraphrase(row) if paraphrase is None else paraphrase) else parse_direction_label)(row.get("raw_response"))


def read_rows(path):
    text = Path(path).read_text(encoding="utf-8")
    rows = json.loads(text) if text.lstrip().startswith("[") else [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise ValueError(f"Expected prediction rows: {path}")
    return rows


def validate_rows(rows, expected_count=None, require_ids=False):
    if not rows:
        raise ValueError("No prediction rows")
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} clips, found {len(rows)}")
    ids = []
    for i, row in enumerate(rows):
        if normalize_label(row.get("gold_label")) is None:
            raise ValueError(f"Invalid gold_label at row {i}: {row.get('gold_label')!r}")
        clip = row.get("benchmark_clip_id", row.get("sample_id"))
        if require_ids and clip is None:
            raise ValueError(f"Missing clip ID at row {i}")
        if clip is not None:
            ids.append(clip)
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate clip IDs")


def binary_metrics(rows, paraphrase=None):
    n = len(rows)
    counts = {key: 0 for key in ("tp_increase_increase", "fp_increase_decrease", "fn_decrease_increase", "tn_decrease_decrease")}
    totals = dict(increase=0, decrease=0)
    correct = dict(increase=0, decrease=0)
    unparsed = dict(increase=0, decrease=0)
    parsed = ambiguous = errors = 0
    for row in rows:
        gold = normalize_label(row.get("gold_label"))
        if gold is None:
            raise ValueError(f"Invalid gold_label: {row.get('gold_label')!r}")
        para = row_paraphrase(row) if paraphrase is None else paraphrase
        ambiguous += len(parse_labels(row.get("raw_response"), para)) > 1
        errors += bool(row.get("error"))
        pred = prediction_for_row(row, para)
        totals[gold] += 1
        if pred is None:
            unparsed[gold] += 1
            continue
        parsed += 1
        correct[gold] += pred == gold
        key = {(True, True): "tp_increase_increase", (True, False): "fp_increase_decrease", (False, True): "fn_decrease_increase", (False, False): "tn_decrease_decrease"}[(pred == "increase", gold == "increase")]
        counts[key] += 1
    tp, fp, fn, tn = (counts[k] for k in counts)
    div = lambda a, b: a / b if b else 0.0
    recalls = {k: correct[k] / v if v else None for k, v in totals.items()}
    available = [v for v in recalls.values() if v is not None]
    precision = div(tp, tp + fp)
    recall = div(tp, totals["increase"])
    accuracy = div(sum(correct.values()), n)
    return {
        "schema_version": SCHEMA_VERSION, "metric": "accuracy_all_clips",
        "n_rows": n, "total_rows": n, "n_parsed": parsed, "parsed_rows": parsed,
        "total_evaluable": parsed, "n_unparsed": n - parsed,
        "n_correct": sum(correct.values()), "correct_rows": sum(correct.values()),
        "n_ambiguous": ambiguous, "n_error": errors,
        "accuracy": accuracy, "accuracy_all": accuracy,
        "accuracy_parsed": div(sum(correct.values()), parsed),
        "balanced_accuracy": sum(available) / len(available) if available else 0.0,
        "accuracy_up": recalls["increase"], "accuracy_down": recalls["decrease"],
        "precision_increase": precision, "recall_increase": recall,
        "f1_increase": div(2 * precision * recall, precision + recall),
        "increase_ratio_parsed": div(tp + fp, parsed),
        "increase_ratio_all": div(tp + fp, n),
        "unparsed_by_gold": unparsed,
        "confusion_matrix": {"labels": ["increase", "decrease"], "counts": counts,
            "table": {"pred_increase": {"gold_increase": tp, "gold_decrease": fp},
                      "pred_decrease": {"gold_increase": fn, "gold_decrease": tn}}},
    }

#!/usr/bin/env python3
"""Create a LaTeX table comparing human and model VocalGrad accuracies."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Iterable


DEFAULT_MODEL_ORDER = [
    "gemini-3-flash",
    "kimi-audio",
    "mimo-audio",
    "step-audio-2-mini",
    "audioflamingo3",
]
DEFAULT_MODEL_RAW_ROOTS = [
    Path("outputs/raw/vocalgrad/default"),
    Path("outputs/raw/vocalgrad/audio-ref"),
]

DEFAULT_EXCLUDED_CATEGORIES = ["voice_brightness"]
DEFAULT_CATEGORY_ORDER = [
    "speaking_speed",
    "voice_pitch",
    "volume",
    "audio_distortion",
    "audio_roughness",
    "background_noise",
    "echo",
    "voice_clarity",
    "voice_vibration",
]

CATEGORY_ABBREVIATIONS = {
    "speaking_speed": "speed",
    "voice_pitch": "pitch",
    "volume": "volume",
    "audio_distortion": "distortion",
    "audio_roughness": "roughness",
    "background_noise": "noise",
    "echo": "echo",
    "voice_clarity": "clarity",
    "voice_vibration": "vibration",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotation-root",
        type=Path,
        default=Path("outputs/annotation_data"),
        help="Root containing per-category shared_annotation_50 JSONL files.",
    )
    parser.add_argument(
        "--model-analysis-root",
        type=Path,
        default=Path("outputs/analysis/vocalgrad/default"),
        help="Root containing per-category model summary JSON files. Used only for category fallback.",
    )
    parser.add_argument(
        "--model-raw-root",
        action="append",
        type=Path,
        default=None,
        help=(
            "Root containing per-category raw model JSONL files. Can be repeated. "
            "When supplied, replaces the default paper-result roots."
        ),
    )
    parser.add_argument(
        "--out-tex",
        type=Path,
        default=Path("outputs/analysis/vocalgrad/default/human_model_accuracy_comparison.tex"),
        help="Output LaTeX table path.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("outputs/analysis/vocalgrad/default/human_model_accuracy_comparison.csv"),
        help="Output CSV table path with raw 0-1 accuracies.",
    )
    parser.add_argument(
        "--caption",
        default=(
            "Human annotator average and model accuracies on VocalGrad by category. "
            "Human values are means over annotator-level accuracies on shared 50-item subsets. "
            "Model values use a regex-based direction parser; responses containing both increase- "
            "and decrease-family direction words are treated as parsing failures and counted "
            "as incorrect."
        ),
    )
    parser.add_argument("--label", default="tab:vocalgrad-human-model-accuracy")
    parser.add_argument(
        "--exclude-category",
        action="append",
        default=DEFAULT_EXCLUDED_CATEGORIES.copy(),
        help="Category to exclude from the table. Can be repeated.",
    )
    args = parser.parse_args()
    args.model_raw_root = args.model_raw_root or DEFAULT_MODEL_RAW_ROOTS
    return args


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc


def normalize_label(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    aliases = {
        "up": "increase",
        "increase": "increase",
        "increased": "increase",
        "down": "decrease",
        "decrease": "decrease",
        "decreased": "decrease",
    }
    return aliases.get(text, text)


def row_is_correct(row: dict, path: Path) -> bool:
    if "is_correct" in row:
        return bool(row["is_correct"])

    gold = normalize_label(row.get("ground_truth_label") or row.get("gold_label") or row.get("label"))
    pred = normalize_label(row.get("annotated_label") or row.get("prediction_label") or row.get("answer"))
    if gold is None or pred is None:
        raise KeyError(
            f"Cannot infer correctness in {path}; expected is_correct or gold/pred labels."
        )
    return gold == pred


def collect_human_accuracies(annotation_root: Path) -> dict[str, dict]:
    summaries: dict[str, dict] = {}
    for category_dir in sorted(p for p in annotation_root.iterdir() if p.is_dir()):
        shared_dir = category_dir / "shared_annotation_50"
        annotator_paths = sorted(shared_dir.glob("annotator_*.jsonl"))
        if not annotator_paths:
            continue

        annotator_rows = []
        annotator_accuracies = []
        for path in annotator_paths:
            total = 0
            correct = 0
            for row in read_jsonl(path):
                total += 1
                correct += int(row_is_correct(row, path))
            if total == 0:
                continue
            annotator_rows.append(
                {
                    "annotator": path.stem,
                    "total": total,
                    "correct": correct,
                    "accuracy": correct / total,
                }
            )
            annotator_accuracies.append(correct / total)

        if annotator_accuracies:
            summaries[category_dir.name] = {
                "human_avg": mean(annotator_accuracies),
                "n_annotators": len(annotator_accuracies),
                "annotators": annotator_rows,
            }
    return summaries


LABEL_RE = re.compile(r"\b(increase|decrease|increases|decreases)\b")


def parse_direction_label(raw_text: object) -> str | None:
    tokens = LABEL_RE.findall(str(raw_text or "").strip().lower())
    labels = {
        "increase" if token in {"increase", "increases"} else "decrease"
        for token in tokens
    }
    if len(labels) != 1:
        return None
    return next(iter(labels))


def find_raw_prediction_file(raw_roots: list[Path], category: str, model_name: str) -> Path | None:
    for root in raw_roots:
        for suffix in (".jsonl", ".json"):
            path = root / category / f"{model_name}{suffix}"
            if path.exists():
                return path
    return None


def accuracy_from_raw_file(path: Path) -> float | None:
    correct = 0
    total = 0
    for row in read_jsonl(path):
        total += 1
        pred = parse_direction_label(row.get("raw_response"))
        gold = normalize_label(row.get("gold_label"))
        if pred is None or gold is None:
            continue
        correct += int(pred == gold)
    if total == 0:
        return None
    return correct / total


def collect_model_accuracies(raw_roots: list[Path]) -> tuple[dict[str, dict[str, float]], list[str]]:
    model_accuracies: dict[str, dict[str, float]] = {}
    model_names = set()

    for raw_root in raw_roots:
        if not raw_root.exists():
            continue
        for category_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
            category = category_dir.name
            for path in sorted([*category_dir.glob("*.jsonl"), *category_dir.glob("*.json")]):
                model_name = path.stem
                if model_name not in DEFAULT_MODEL_ORDER:
                    continue
                if model_name in model_accuracies.get(category, {}):
                    continue
                accuracy = accuracy_from_raw_file(path)
                if accuracy is None:
                    continue
                model_accuracies.setdefault(category, {})[model_name] = accuracy
                model_names.add(model_name)

    for category in DEFAULT_CATEGORY_ORDER:
        for model_name in DEFAULT_MODEL_ORDER:
            if model_name in model_accuracies.get(category, {}):
                continue
            path = find_raw_prediction_file(raw_roots, category, model_name)
            if path is None:
                continue
            accuracy = accuracy_from_raw_file(path)
            if accuracy is None:
                continue
            model_accuracies.setdefault(category, {})[model_name] = accuracy
            model_names.add(model_name)

    ordered_models = [name for name in DEFAULT_MODEL_ORDER if name in model_names]
    return model_accuracies, ordered_models


def ordered_categories(
    human: dict[str, dict], models: dict[str, dict[str, float]], model_analysis_root: Path
) -> list[str]:
    known_categories = set(human) | set(models)
    ordered = [category for category in DEFAULT_CATEGORY_ORDER if category in known_categories]
    extras = sorted(known_categories - set(ordered))
    if ordered or extras:
        return ordered + extras

    comparison_csv = model_analysis_root / "accuracy_comparison.csv"
    if comparison_csv.exists():
        with comparison_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
        ordered = [row["category"] for row in reader if row.get("category")]
        extras = sorted((set(human) | set(models)) - set(ordered))
        return ordered + extras
    return sorted(set(human) | set(models))


def build_category_rows(
    human: dict[str, dict],
    models: dict[str, dict[str, float]],
    model_names: list[str],
    categories: list[str],
) -> list[dict[str, object]]:
    rows = []
    for category in categories:
        row: dict[str, object] = {
            "category": category,
            "human_avg": human.get(category, {}).get("human_avg"),
            "n_annotators": human.get(category, {}).get("n_annotators", 0),
        }
        model_values = []
        for model_name in model_names:
            value = models.get(category, {}).get(model_name)
            row[model_name] = value
            if value is not None:
                model_values.append(value)
        rows.append(row)
    return rows


def build_transposed_rows(
    category_rows: list[dict[str, object]],
    model_names: list[str],
    categories: list[str],
) -> list[dict[str, object]]:
    category_lookup = {str(row["category"]): row for row in category_rows}

    rows: list[dict[str, object]] = []
    human_row: dict[str, object] = {"row_name": "Human Avg."}
    for category in categories:
        human_row[category] = category_lookup.get(category, {}).get("human_avg")
    human_values = [human_row.get(category) for category in categories]
    human_clean = [float(value) for value in human_values if value is not None]
    human_row["avg"] = mean(human_clean) if human_clean else None
    rows.append(human_row)

    for model_name in model_names:
        row: dict[str, object] = {"row_name": display_model_name(model_name)}
        for category in categories:
            row[category] = category_lookup.get(category, {}).get(model_name)
        model_values = [row.get(category) for category in categories]
        model_clean = [float(value) for value in model_values if value is not None]
        row["avg"] = mean(model_clean) if model_clean else None
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], categories: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["row_name", "avg", *categories]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def latex_escape(text: object) -> str:
    value = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in value)


def display_model_name(model_name: str) -> str:
    names = {
        "audioflamingo3": "AudioFlamingo3",
        "gemini-3-flash": "Gemini 3 Flash",
        "kimi-audio": "Kimi-Audio",
        "mimo-audio": "MiMo-Audio",
        "step-audio-2-mini": "Step-Audio-2 Mini",
    }
    return names.get(model_name, model_name.replace("-", " ").title())


def fmt_accuracy(value: object) -> str:
    if value is None:
        return r"\textemdash{}"
    if isinstance(value, float) and math.isnan(value):
        return r"\textemdash{}"
    return f"{float(value) * 100:.1f}"


def column_maxima(rows: list[dict[str, object]], categories: list[str]) -> dict[str, float]:
    maxima: dict[str, float] = {}
    model_rows = [row for row in rows if row.get("row_name") != "Human Avg."]
    for category in ["avg", *categories]:
        values = [
            float(row[category])
            for row in model_rows
            if row.get(category) is not None and not (isinstance(row.get(category), float) and math.isnan(row[category]))
        ]
        if values:
            maxima[category] = max(values)
    return maxima


def write_latex(
    path: Path,
    rows: list[dict[str, object]],
    categories: list[str],
    caption: str,
    label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        ("row_name", "Model"),
        ("avg", "Avg."),
        *[(category, CATEGORY_ABBREVIATIONS.get(category, category)) for category in categories],
    ]
    alignment = "l" + "r" * (len(columns) - 1)
    maxima = column_maxima(rows, categories)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{latex_escape(label)}}}",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(latex_escape(header) for _, header in columns) + r" \\",
        r"\midrule",
    ]
    for row_index, row in enumerate(rows):
        cells = []
        for key, _ in columns:
            if key == "row_name":
                cells.append(latex_escape(row[key]))
            else:
                formatted = fmt_accuracy(row.get(key))
                value = row.get(key)
                if (
                    key in maxima
                    and value is not None
                    and not (isinstance(value, float) and math.isnan(value))
                    and abs(float(value) - maxima[key]) < 1e-12
                ):
                    formatted = rf"\textbf{{{formatted}}}"
                cells.append(formatted)
        lines.append(" & ".join(cells) + r" \\")
        if row_index == 0:
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    human = collect_human_accuracies(args.annotation_root)
    models, model_names = collect_model_accuracies(args.model_raw_root)
    categories = ordered_categories(human, models, args.model_analysis_root)
    excluded_categories = set(args.exclude_category)
    categories = [category for category in categories if category not in excluded_categories]
    category_rows = build_category_rows(human, models, model_names, categories)
    rows = build_transposed_rows(category_rows, model_names, categories)
    write_csv(args.out_csv, rows, categories)
    write_latex(args.out_tex, rows, categories, args.caption, args.label)

    print(f"wrote CSV: {args.out_csv}")
    print(f"wrote LaTeX: {args.out_tex}")
    print(f"rows={len(rows)} categories={len(categories)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create a LaTeX table for VocalGrad ablation-beep accuracies."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


DEFAULT_MODEL_ORDER = [
    "gemini-3-flash",
    "kimi-audio",
    "mimo-audio",
    "step-audio-2-mini",
    "audioflamingo3",
]

DEFAULT_CATEGORY_ORDER = [
    "voice_pitch",
    "volume",
    "background_noise",
    "voice_vibration",
]

CATEGORY_ABBREVIATIONS = {
    "voice_pitch": "pitch",
    "volume": "volume",
    "background_noise": "noise",
    "voice_vibration": "vibration",
}



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=Path("outputs/analysis/vocalgrad_ablation_beep/audio-ref"),
        help="Root containing per-category ablation-beep model summary JSON files. Used only as a fallback.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("outputs/raw/vocalgrad_ablation_beep/audio-ref"),
        help="Root containing per-category ablation-beep raw model JSONL files.",
    )
    parser.add_argument(
        "--out-tex",
        type=Path,
        default=Path("outputs/analysis/vocalgrad_ablation_beep/audio-ref/ablation_beep_accuracy_comparison.tex"),
        help="Output LaTeX table path.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("outputs/analysis/vocalgrad_ablation_beep/audio-ref/ablation_beep_accuracy_comparison.csv"),
        help="Output CSV table path with raw 0-1 accuracies.",
    )
    parser.add_argument(
        "--caption",
        default=(
            "Model accuracies on VocalGrad ablation-beep subsets by category. "
            "Model values use a regex-based direction parser; responses containing both "
            "increase- and decrease-family direction words, or neither family, are treated "
            "as parsing failures and counted as incorrect."
        ),
    )
    parser.add_argument("--label", default="tab:vocalgrad-ablation-beep-accuracy")
    return parser.parse_args()


def collect_model_accuracies(analysis_root: Path) -> tuple[dict[str, dict[str, float]], list[str]]:
    model_accuracies: dict[str, dict[str, float]] = {}
    model_names = set()

    for category_dir in sorted(p for p in analysis_root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        category = category_dir.name
        for path in sorted(category_dir.glob("*.json")):
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            accuracy = payload.get("overall", {}).get("accuracy")
            if accuracy is None:
                continue
            model_name = path.stem
            model_accuracies.setdefault(category, {})[model_name] = float(accuracy)
            model_names.add(model_name)

    ordered_models = [name for name in DEFAULT_MODEL_ORDER if name in model_names]
    ordered_models.extend(sorted(model_names - set(ordered_models)))
    return model_accuracies, ordered_models


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
    return rows


def normalize_label(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return {
        "up": "increase",
        "increase": "increase",
        "increased": "increase",
        "down": "decrease",
        "decrease": "decrease",
        "decreased": "decrease",
    }.get(text, text)


from evaluation_common import parse_direction_label, prediction_for_row


def accuracy_from_raw_file(path: Path) -> float | None:
    correct = 0
    total = 0
    for row in read_jsonl(path):
        total += 1
        pred = prediction_for_row(row)
        gold = normalize_label(row.get("gold_label"))
        if pred is None or gold is None:
            continue
        correct += int(pred == gold)
    if total == 0:
        return None
    return correct / total


def collect_model_accuracies_from_raw(raw_root: Path) -> tuple[dict[str, dict[str, float]], list[str]]:
    model_accuracies: dict[str, dict[str, float]] = {}
    model_names = set()
    if not raw_root.exists():
        return model_accuracies, []
    for category_dir in sorted(p for p in raw_root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        category = category_dir.name
        for model_name in DEFAULT_MODEL_ORDER:
            path = category_dir / f"{model_name}.jsonl"
            if not path.exists():
                path = category_dir / f"{model_name}.json"
            if not path.exists():
                continue
            accuracy = accuracy_from_raw_file(path)
            if accuracy is None:
                continue
            model_accuracies.setdefault(category, {})[model_name] = accuracy
            model_names.add(model_name)
    ordered_models = [name for name in DEFAULT_MODEL_ORDER if name in model_names]
    return model_accuracies, ordered_models


def ordered_categories(models: dict[str, dict[str, float]]) -> list[str]:
    known_categories = set(models)
    ordered = [category for category in DEFAULT_CATEGORY_ORDER if category in known_categories]
    extras = sorted(known_categories - set(ordered))
    return ordered + extras


def build_rows(models: dict[str, dict[str, float]], model_names: list[str], categories: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_name in model_names:
        row: dict[str, object] = {"row_name": display_model_name(model_name)}
        for category in categories:
            row[category] = models.get(category, {}).get(model_name)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], categories: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["row_name", *categories]
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
    for category in categories:
        values = [
            float(row[category])
            for row in rows
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
    columns = [("row_name", "Model"), *[(category, CATEGORY_ABBREVIATIONS.get(category, category)) for category in categories]]
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
    for row in rows:
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
    models, model_names = collect_model_accuracies_from_raw(args.raw_root)
    if not models:
        models, model_names = collect_model_accuracies(args.analysis_root)
    categories = ordered_categories(models)
    rows = build_rows(models, model_names, categories)
    write_csv(args.out_csv, rows, categories)
    write_latex(args.out_tex, rows, categories, args.caption, args.label)

    print(f"wrote CSV: {args.out_csv}")
    print(f"wrote LaTeX: {args.out_tex}")
    print(f"rows={len(rows)} categories={len(categories)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create a 4-model in-domain fine-tuning results table."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


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
DEFAULT_EXCLUDED_CATEGORIES: list[str] = []
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
MODEL_DISPLAY_NAMES = {
    "kimi-audio": "Kimi-Audio",
    "mimo-audio": "MiMo-Audio",
    "step-audio-2-mini": "Step-Audio-2 Mini",
    "audioflamingo3": "AudioFlamingo3",
}
METHOD_DISPLAY_NAMES = {
    "lm_head_only": "LM-Head",
    "all_linear": "All-Linear",
}
MODEL_ORDER = ["kimi-audio", "mimo-audio", "step-audio-2-mini", "audioflamingo3"]
SCOPE_ORDER = ["lm_head_only", "all_linear"]
DEFAULT_RECORDS = {
    ("kimi-audio", "lm_head_only"): Path("outputs/analysis/vocalgrad_finetune_eval/_comparisons/epoch1/generalization_records.csv"),
    ("kimi-audio", "all_linear"): Path("outputs/analysis/vocalgrad_finetune_eval/_comparisons/epoch1/generalization_records.csv"),
    ("mimo-audio", "lm_head_only"): Path("outputs/analysis/vocalgrad_finetune_eval/_comparisons/mimo_lm_head_epoch1/generalization_records.csv"),
    ("mimo-audio", "all_linear"): Path("outputs/analysis/vocalgrad_finetune_eval/_comparisons/mimo_all_linear_epoch1/generalization_records.csv"),
    ("step-audio-2-mini", "lm_head_only"): Path("outputs/analysis/vocalgrad_finetune_eval/_comparisons/stepaudio2_lm_head_epoch1/generalization_records.csv"),
    ("step-audio-2-mini", "all_linear"): Path("outputs/analysis/vocalgrad_finetune_eval/_comparisons/stepaudio2_all_linear_epoch1/generalization_records.csv"),
    ("audioflamingo3", "lm_head_only"): Path("outputs/analysis/vocalgrad_finetune_eval/_comparisons/af3_lm_head_epoch1/generalization_records.csv"),
    ("audioflamingo3", "all_linear"): Path("outputs/analysis/vocalgrad_finetune_eval/_comparisons/af3_all_linear_epoch1/generalization_records.csv"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-tex",
        type=Path,
        default=Path("outputs/analysis/vocalgrad_finetune_eval/_comparisons/epoch1/in_domain_accuracy_results_table_4models.tex"),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("outputs/analysis/vocalgrad_finetune_eval/_comparisons/epoch1/in_domain_accuracy_results_table_4models.csv"),
    )
    parser.add_argument("--comparison-root", type=Path, default=None, help="Root containing per-run generalization_records.csv")
    parser.add_argument("--metric", default="accuracy", choices=["accuracy", "balanced_accuracy"])
    parser.add_argument(
        "--exclude-category",
        action="append",
        default=DEFAULT_EXCLUDED_CATEGORIES.copy(),
    )
    return parser.parse_args()


def read_diagonal(path: Path, scope: str, metric: str) -> dict[str, float]:
    values: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("schema_version") != "vocalgrad-all-clips-v1" or row.get("metric") != "accuracy_all_clips":
                raise ValueError("Regenerate generalization_records.csv from current summaries first")
            if row.get("scope") != scope:
                continue
            train_category = row.get("train_category")
            test_category = row.get("test_category")
            if train_category != test_category or not train_category:
                continue
            value = row.get(metric)
            if value not in (None, ""):
                values[train_category] = float(value)
    return values


def build_rows(categories: list[str], metric: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scope in SCOPE_ORDER:
        for model in MODEL_ORDER:
            path = DEFAULT_RECORDS[(model, scope)]
            if not path.exists():
                raise FileNotFoundError(path)
            values = read_diagonal(path, scope, metric)
            row: dict[str, object] = {
                "method": METHOD_DISPLAY_NAMES[scope] if model == "kimi-audio" else "",
                "model": MODEL_DISPLAY_NAMES[model],
            }
            for category in categories:
                row[category] = values.get(category)
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], categories: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["method", "model", *categories]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def fmt_rate(value: object) -> str:
    if value is None:
        return r"\textemdash{}"
    if isinstance(value, float) and math.isnan(value):
        return r"\textemdash{}"
    return f"{float(value) * 100:.1f}"


def best_values_by_method(rows: list[dict[str, object]], categories: list[str]) -> dict[tuple[str, str], float]:
    best_values: dict[tuple[str, str], float] = {}
    current_method = ""
    for row in rows:
        method = str(row.get("method") or current_method)
        if row.get("method"):
            current_method = method
        for category in categories:
            value = row.get(category)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            key = (method, category)
            best_values[key] = max(best_values.get(key, float("-inf")), float(value))
    return best_values


def write_latex(path: Path, rows: list[dict[str, object]], categories: list[str], metric: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    alignment = "ll" + "r" * len(categories)
    best_values = best_values_by_method(rows, categories)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{Four-model 1-epoch in-domain fine-tuning {latex_escape(metric.replace('_', ' '))} by category.}}",
        r"\label{tab:four-model-epoch1-in-domain-results}",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        "Method & Model & " + " & ".join(latex_escape(CATEGORY_ABBREVIATIONS.get(category, category)) for category in categories) + r" \\",
        r"\midrule",
    ]
    current_method = ""
    for row_index, row in enumerate(rows):
        if row_index == len(MODEL_ORDER):
            lines.append(r"\midrule")
        method = str(row.get("method") or current_method)
        if row.get("method"):
            current_method = method
        cells = [latex_escape(row["method"]), latex_escape(row["model"])]
        for category in categories:
            value = row.get(category)
            formatted = fmt_rate(value)
            if (
                value is not None
                and not (isinstance(value, float) and math.isnan(value))
                and abs(float(value) - best_values.get((method, category), float("inf"))) < 1e-12
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
    if args.comparison_root is not None:
        for key, path in list(DEFAULT_RECORDS.items()):
            DEFAULT_RECORDS[key] = args.comparison_root / path.parent.name / path.name
    excluded_categories = set(args.exclude_category)
    categories = [category for category in DEFAULT_CATEGORY_ORDER if category not in excluded_categories]
    rows = build_rows(categories, args.metric)
    write_csv(args.out_csv, rows, categories)
    write_latex(args.out_tex, rows, categories, args.metric)
    print(f"wrote CSV: {args.out_csv}")
    print(f"wrote LaTeX: {args.out_tex}")
    print(f"rows={len(rows)} categories={len(categories)}")


if __name__ == "__main__":
    main()

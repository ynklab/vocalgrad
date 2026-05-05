#!/usr/bin/env python3
"""Create a LaTeX table for in-domain Kimi fine-tuning results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean


DEFAULT_SCOPES = ["lm_head_only", "all_linear"]
DEFAULT_EXCLUDED_CATEGORIES: list[str] = []
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
        "--run-root",
        type=Path,
        default=Path("outputs/analysis/vocalgrad_finetune_eval/kimi-audio/epoch1"),
        help="Root with scope/train_category/test_category.json summaries.",
    )
    parser.add_argument(
        "--base-order-csv",
        type=Path,
        default=Path("outputs/analysis/vocalgrad/default/accuracy_comparison.csv"),
        help="CSV used only to keep category ordering consistent.",
    )
    parser.add_argument(
        "--out-tex",
        type=Path,
        default=Path("outputs/analysis/vocalgrad_finetune_eval/kimi-audio/epoch1/in_domain_accuracy_table.tex"),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("outputs/analysis/vocalgrad_finetune_eval/kimi-audio/epoch1/in_domain_accuracy_table.csv"),
    )
    parser.add_argument(
        "--out-results-tex",
        type=Path,
        default=Path("outputs/analysis/vocalgrad_finetune_eval/kimi-audio/epoch1/in_domain_accuracy_results_table.tex"),
    )
    parser.add_argument(
        "--out-results-csv",
        type=Path,
        default=Path("outputs/analysis/vocalgrad_finetune_eval/kimi-audio/epoch1/in_domain_accuracy_results_table.csv"),
    )
    parser.add_argument("--scope", action="append", default=DEFAULT_SCOPES.copy())
    parser.add_argument(
        "--exclude-category",
        action="append",
        default=DEFAULT_EXCLUDED_CATEGORIES.copy(),
        help="Category to exclude. Can be repeated.",
    )
    parser.add_argument("--metric", default="accuracy", choices=["accuracy", "balanced_accuracy"])
    return parser.parse_args()


def category_order(run_root: Path, base_order_csv: Path, excluded_categories: set[str]) -> list[str]:
    ordered: list[str] = []
    discovered = set()
    for category in DEFAULT_CATEGORY_ORDER:
        if category not in excluded_categories:
            ordered.append(category)
            discovered.add(category)
    if base_order_csv.exists():
        with base_order_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                category = row.get("category")
                if category and category not in excluded_categories and category not in discovered:
                    ordered.append(category)
                    discovered.add(category)
    for scope_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        for train_dir in sorted(path for path in scope_dir.iterdir() if path.is_dir()):
            if train_dir.name not in excluded_categories and train_dir.name not in discovered:
                ordered.append(train_dir.name)
                discovered.add(train_dir.name)
    return ordered


def load_diagonal(
    run_root: Path,
    scopes: list[str],
    categories: list[str],
    metric: str,
) -> list[dict[str, object]]:
    rows = []
    for category in categories:
        row: dict[str, object] = {"category": category}
        for scope in scopes:
            path = run_root / scope / category / f"{category}.json"
            value = None
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                overall = payload.get("overall", {})
                metric_value = overall.get(metric)
                if isinstance(metric_value, (int, float)):
                    value = float(metric_value)
            row[scope] = value
        rows.append(row)
    return rows


def add_average_row(rows: list[dict[str, object]], scopes: list[str]) -> list[dict[str, object]]:
    average_row: dict[str, object] = {"category": "Average"}
    for scope in scopes:
        values = [row[scope] for row in rows if isinstance(row.get(scope), float)]
        average_row[scope] = mean(values) if values else None
    return [*rows, average_row]


def write_csv(path: Path, rows: list[dict[str, object]], scopes: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["category", *scopes]
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


def display_category(category: str) -> str:
    if category == "Average":
        return category
    return category.replace("_", " ")


def display_scope(scope: str) -> str:
    names = {
        "lm_head_only": "LM head only",
        "all_linear": "All linear",
    }
    return names.get(scope, scope.replace("_", " ").title())


def display_scope_results(scope: str) -> str:
    names = {
        "lm_head_only": "LM-Head-Only LoRA",
        "all_linear": "All-Linear LoRA",
    }
    return names.get(scope, scope.replace("_", " ").title())


def display_category_short(category: str) -> str:
    return CATEGORY_ABBREVIATIONS.get(category, category.replace("_", " "))


def fmt_rate(value: object) -> str:
    if value is None:
        return r"\textemdash{}"
    if isinstance(value, float) and math.isnan(value):
        return r"\textemdash{}"
    return f"{float(value) * 100:.1f}"


def write_latex(path: Path, rows: list[dict[str, object]], scopes: list[str], metric: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{Kimi-Audio 1-epoch fine-tuning in-domain {latex_escape(metric.replace('_', ' '))}. Train and test categories are identical for every row.}}",
        r"\label{tab:kimi-epoch1-in-domain}",
        r"\begin{tabular}{l" + "r" * len(scopes) + r"}",
        r"\toprule",
        "Category & " + " & ".join(latex_escape(display_scope(scope)) for scope in scopes) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        if row["category"] == "Average":
            lines.append(r"\midrule")
        cells = [latex_escape(display_category(str(row["category"])))]
        cells.extend(fmt_rate(row.get(scope)) for scope in scopes)
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{0.25em}",
            r"\footnotesize Values are percentages.",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_transposed_rows(rows: list[dict[str, object]], scopes: list[str], categories: list[str]) -> list[dict[str, object]]:
    row_lookup = {
        str(row["category"]): row
        for row in rows
        if str(row["category"]) != "Average"
    }
    transposed_rows: list[dict[str, object]] = []
    for scope in scopes:
        record: dict[str, object] = {"row_name": display_scope_results(scope)}
        for category in categories:
            record[category] = row_lookup.get(category, {}).get(scope)
        transposed_rows.append(record)
    return transposed_rows


def write_transposed_csv(path: Path, rows: list[dict[str, object]], categories: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["row_name", *categories]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_transposed_latex(path: Path, rows: list[dict[str, object]], categories: list[str], metric: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{Kimi-Audio 1-epoch in-domain fine-tuning {latex_escape(metric.replace('_', ' '))} by category.}}",
        r"\label{tab:kimi-epoch1-in-domain-results}",
        r"\begin{tabular}{l" + "r" * len(categories) + r"}",
        r"\toprule",
        "Model & " + " & ".join(latex_escape(display_category_short(category)) for category in categories) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        cells = [latex_escape(str(row["row_name"]))]
        for category in categories:
            value = row.get(category)
            cells.append(fmt_rate(value))
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
    excluded_categories = set(args.exclude_category)
    categories = category_order(args.run_root, args.base_order_csv, excluded_categories)
    rows = load_diagonal(args.run_root, args.scope, categories, args.metric)
    rows_with_average = add_average_row(rows, args.scope)
    write_csv(args.out_csv, rows_with_average, args.scope)
    write_latex(args.out_tex, rows_with_average, args.scope, args.metric)
    transposed_rows = build_transposed_rows(rows, args.scope, categories)
    write_transposed_csv(args.out_results_csv, transposed_rows, categories)
    write_transposed_latex(args.out_results_tex, transposed_rows, categories, args.metric)
    print(f"wrote CSV: {args.out_csv}")
    print(f"wrote LaTeX: {args.out_tex}")
    print(f"wrote Results CSV: {args.out_results_csv}")
    print(f"wrote Results LaTeX: {args.out_results_tex}")
    print(f"categories={len(categories)} scopes={args.scope} excluded={sorted(excluded_categories)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create separate LaTeX tables for VocalGrad parsing-failure analyses."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_MODEL_ORDER = [
    "kimi-audio",
    "gemini-3-flash",
    "mimo-audio",
    "step-audio-2-mini",
    "audioflamingo3",
]

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

MODEL_DISPLAY_NAMES = {
    "audioflamingo3": "AudioFlamingo3",
    "gemini-3-flash": "Gemini 3 Flash",
    "kimi-audio": "Kimi-Audio",
    "mimo-audio": "MiMo-Audio",
    "step-audio-2-mini": "Step-Audio-2 Mini",
}

LABEL_RE = re.compile(r"\b(increase|decrease|increases|decreases)\b")
DEFAULT_RAW_ROOTS = [
    Path("outputs/raw/vocalgrad/default"),
    Path("outputs/raw/vocalgrad/audio-ref"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        action="append",
        type=Path,
        default=None,
        help=(
            "Root containing per-category raw model JSONL files. Can be repeated. "
            "When supplied, replaces the default paper-result roots."
        ),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("outputs/analysis/vocalgrad/default"),
        help="Output directory for separate parsing-failure tables.",
    )
    parser.add_argument(
        "--exclude-category",
        action="append",
        default=["voice_brightness"],
        help="Category to exclude. Can be repeated.",
    )
    args = parser.parse_args()
    args.raw_root = args.raw_root or DEFAULT_RAW_ROOTS
    return args


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def display_model_name(model_name: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_name, model_name.replace("-", " ").title())


def find_raw_file(raw_roots: list[Path], category: str, model_name: str) -> Path | None:
    for root in raw_roots:
        for suffix in (".jsonl", ".json"):
            path = root / category / f"{model_name}{suffix}"
            if path.exists():
                return path
    return None


def normalize_output(value: object) -> str:
    return str(value or "").strip().lower()


def parse_direction_label(raw_text: object) -> str | None:
    tokens = LABEL_RE.findall(normalize_output(raw_text))
    labels = {
        "increase" if token in {"increase", "increases"} else "decrease"
        for token in tokens
    }
    if len(labels) != 1:
        return None
    return next(iter(labels))


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


def percent(value: float | None) -> str:
    if value is None:
        return r"\textemdash{}"
    return f"{value * 100:.1f}"


def collect_rows(raw_roots: list[Path], categories: list[str]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    model_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for model_name in DEFAULT_MODEL_ORDER:
        for category in categories:
            path = find_raw_file(raw_roots, category, model_name)
            if path is None:
                continue
            model_rows.setdefault(model_name, {})[category] = read_jsonl(path)
    return model_rows


def exact_rate(rows: list[dict[str, Any]]) -> float:
    exact = sum(normalize_output(row.get("raw_response")) in {"increase", "decrease"} for row in rows)
    return exact / len(rows) if rows else 0.0


def parser_match_rate(rows: list[dict[str, Any]]) -> float:
    count = 0
    for row in rows:
        count += int(parse_direction_label(row.get("raw_response")) is not None)
    return count / len(rows) if rows else 0.0


def exact_accuracy(rows: list[dict[str, Any]]) -> float:
    correct = 0
    for row in rows:
        raw = normalize_output(row.get("raw_response"))
        gold = normalize_output(row.get("gold_label"))
        correct += int(raw in {"increase", "decrease"} and raw == gold)
    return correct / len(rows) if rows else 0.0


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_latex_table(
    path: Path,
    columns: list[tuple[str, str]],
    rows: list[dict[str, object]],
    caption: str,
    label: str,
    bold_max_numeric: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    alignment = "l" + "r" * (len(columns) - 1)
    maxima: dict[str, float] = {}
    if bold_max_numeric:
        for key, _ in columns:
            if key == "row_name":
                continue
            values: list[float] = []
            for row in rows:
                try:
                    values.append(float(str(row[key])))
                except (KeyError, TypeError, ValueError):
                    continue
            if values:
                maxima[key] = max(values)
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
                continue
            cell = str(row[key])
            if key in maxima:
                try:
                    value = float(cell)
                except ValueError:
                    value = None
                if value is not None and abs(value - maxima[key]) < 1e-12:
                    cell = rf"\textbf{{{cell}}}"
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    excluded = set(args.exclude_category)
    categories = [category for category in DEFAULT_CATEGORY_ORDER if category not in excluded]
    data = collect_rows(args.raw_root, categories)

    output_match_rows: list[dict[str, object]] = []
    for model_name in DEFAULT_MODEL_ORDER:
        category_rows = data.get(model_name, {})
        rows = [row for category in categories for row in category_rows.get(category, [])]
        if not rows:
            continue
        output_match_rows.append(
            {
                "row_name": display_model_name(model_name),
                "exact_match": percent(exact_rate(rows)),
                "parser_match": percent(parser_match_rate(rows)),
            }
        )
    write_csv(args.out_root / "parsing_output_match.csv", output_match_rows, ["row_name", "exact_match", "parser_match"])
    write_latex_table(
        args.out_root / "parsing_output_match.tex",
        [("row_name", "Model"), ("exact_match", "Exact match"), ("parser_match", "Parser match")],
        output_match_rows,
        "Rates of raw model outputs that exactly match the target direction words or match the ambiguity-aware parser on VocalGrad. Parser match requires exactly one of the increase- or decrease-family direction labels to appear. Values are percentages over all model responses.",
        "tab:vocalgrad-parsing-output-match",
    )

    non_exact_rows: list[dict[str, object]] = []
    exact_accuracy_rows: list[dict[str, object]] = []
    for model_name in DEFAULT_MODEL_ORDER:
        category_rows = data.get(model_name, {})
        if not category_rows:
            continue
        non_exact_values: list[float] = []
        exact_accuracy_values: list[float] = []
        non_exact_row: dict[str, object] = {"row_name": display_model_name(model_name)}
        exact_accuracy_row: dict[str, object] = {"row_name": display_model_name(model_name)}
        for category in categories:
            rows = category_rows.get(category, [])
            non_exact = 1.0 - exact_rate(rows) if rows else None
            exact_acc = exact_accuracy(rows) if rows else None
            non_exact_row[category] = percent(non_exact)
            exact_accuracy_row[category] = percent(exact_acc)
            if non_exact is not None:
                non_exact_values.append(non_exact)
            if exact_acc is not None:
                exact_accuracy_values.append(exact_acc)
        non_exact_row["avg"] = percent(mean(non_exact_values) if non_exact_values else None)
        exact_accuracy_row["avg"] = percent(mean(exact_accuracy_values) if exact_accuracy_values else None)
        non_exact_rows.append(non_exact_row)
        exact_accuracy_rows.append(exact_accuracy_row)

    category_columns = [
        ("row_name", "Model"),
        ("avg", "Avg."),
        *[(category, CATEGORY_ABBREVIATIONS.get(category, category)) for category in categories],
    ]
    category_fieldnames = ["row_name", "avg", *categories]
    write_csv(args.out_root / "category_non_exact_match.csv", non_exact_rows, category_fieldnames)
    write_latex_table(
        args.out_root / "category_non_exact_match.tex",
        category_columns,
        non_exact_rows,
        "Category-wise non-exact-match rates for raw model outputs on VocalGrad. Values are percentages of responses whose stripped, lowercased output is not exactly increase or decrease.",
        "tab:vocalgrad-category-non-exact-match",
    )
    write_csv(args.out_root / "exact_match_accuracy.csv", exact_accuracy_rows, category_fieldnames)
    write_latex_table(
        args.out_root / "exact_match_accuracy.tex",
        category_columns,
        exact_accuracy_rows,
        "Model accuracies on VocalGrad when only exact raw outputs are counted as correct. A response is correct only when its stripped, lowercased output exactly matches the gold label.",
        "tab:vocalgrad-exact-match-accuracy",
        bold_max_numeric=True,
    )

    print(f"wrote tables to: {args.out_root}")


if __name__ == "__main__":
    main()

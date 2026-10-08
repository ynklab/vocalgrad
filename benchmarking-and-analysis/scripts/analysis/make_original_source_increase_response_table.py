#!/usr/bin/env python3
"""Summarize increase-response rates for original VocalGrad and source clips."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean


DEFAULT_MODEL_ORDER = [
    "gemini-3-flash",
    "kimi-audio",
    "mimo-audio",
    "step-audio-2-mini",
    "audioflamingo3",
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



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original-analysis-root",
        type=Path,
        default=Path("outputs/analysis/vocalgrad/default"),
        help="Root containing per-category VocalGrad model summary JSON files. Used only as a fallback.",
    )
    parser.add_argument(
        "--original-raw-root",
        action="append",
        type=Path,
        default=[
            Path("outputs/raw/vocalgrad/default"),
            Path("outputs/raw/vocalgrad/audio-ref"),
        ],
        help="Root containing per-category VocalGrad raw model JSONL files. Can be repeated.",
    )
    parser.add_argument(
        "--source-analysis-root",
        type=Path,
        default=Path("outputs/analysis/source_clips_direction_generation"),
        help="Root containing source clip direction-bias model summary JSON files. Used only as a fallback.",
    )
    parser.add_argument(
        "--source-raw-root",
        type=Path,
        default=Path("outputs/raw/source_clips_direction_generation"),
        help="Root containing source clip direction-bias raw model JSONL files.",
    )
    parser.add_argument(
        "--out-tex",
        type=Path,
        default=Path(
            "outputs/analysis/source_clips_direction_generation/"
            "original_vs_source_increase_response_summary.tex"
        ),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path(
            "outputs/analysis/source_clips_direction_generation/"
            "original_vs_source_increase_response_summary.csv"
        ),
    )
    parser.add_argument(
        "--exclude-category",
        action="append",
        default=DEFAULT_EXCLUDED_CATEGORIES.copy(),
        help="Category to exclude. Can be repeated.",
    )
    parser.add_argument(
        "--caption",
        default=(
            "Increase response bias on original VocalGrad data and source clips. "
            "Rates are computed over responses parsed by the ambiguity-aware direction parser."
        ),
    )
    parser.add_argument("--label", default="tab:original-source-increase-response")
    return parser.parse_args()


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


def fmt_rate(value: float | None) -> str:
    if value is None or math.isnan(value):
        return r"\textemdash{}"
    return f"{value * 100:.1f}"


def fmt_category_rate(category: str | None, value: float | None) -> str:
    if category is None or value is None:
        return r"\textemdash{}"
    return f"{latex_escape(category)} ({fmt_rate(value)})"


def display_category_name(category: str | None) -> str:
    if category is None:
        return r"\textemdash{}"
    return latex_escape(CATEGORY_ABBREVIATIONS.get(category, category))


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


from evaluation_common import parse_direction_label, prediction_for_row


def increase_rate_from_rows(rows: list[dict]) -> float | None:
    parsed = [prediction_for_row(row) for row in rows]
    parsed = [label for label in parsed if label is not None]
    if not parsed:
        return None
    return sum(label == "increase" for label in parsed) / len(parsed)


def find_original_raw_file(raw_roots: list[Path], category: str, model_name: str) -> Path | None:
    for root in raw_roots:
        for suffix in (".jsonl", ".json"):
            path = root / category / f"{model_name}{suffix}"
            if path.exists():
                return path
    return None


def original_increase_rate(path: Path) -> float | None:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    overall = payload.get("overall", {})
    total = overall.get("total_evaluable")
    table = overall.get("confusion_matrix", {}).get("table", {})
    pred_increase = table.get("pred_increase", {})
    if total:
        count = sum(float(value) for value in pred_increase.values())
        return count / float(total)
    return None


def collect_original_rates_from_raw(raw_roots: list[Path], excluded_categories: set[str]) -> dict[str, dict[str, float]]:
    rates: dict[str, dict[str, float]] = {}
    categories = [
        category
        for category in CATEGORY_ABBREVIATIONS
        if category not in excluded_categories
    ]
    for model in DEFAULT_MODEL_ORDER:
        for category in categories:
            path = find_original_raw_file(raw_roots, category, model)
            if path is None:
                continue
            rate = increase_rate_from_rows(read_jsonl(path))
            if rate is None:
                continue
            rates.setdefault(model, {})[category] = rate
    return rates


def collect_original_rates(root: Path, excluded_categories: set[str]) -> dict[str, dict[str, float]]:
    rates: dict[str, dict[str, float]] = {}
    for category_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        category = category_dir.name
        if category in excluded_categories:
            continue
        for path in sorted(category_dir.glob("*.json")):
            rate = original_increase_rate(path)
            if rate is None:
                continue
            rates.setdefault(path.stem, {})[category] = rate
    return rates


def collect_source_rates_from_raw(root: Path, excluded_categories: set[str]) -> dict[str, dict[str, float]]:
    rates: dict[str, dict[str, float]] = {}
    for model in DEFAULT_MODEL_ORDER:
        path = root / f"{model}.jsonl"
        if not path.exists():
            continue
        rows_by_category: dict[str, list[dict]] = {}
        for row in read_jsonl(path):
            category = str(row.get("category", ""))
            if not category or category in excluded_categories:
                continue
            rows_by_category.setdefault(category, []).append(row)
        for category, rows in rows_by_category.items():
            rate = increase_rate_from_rows(rows)
            if rate is None:
                continue
            rates.setdefault(model, {})[category] = rate
    return rates


def collect_source_rates(root: Path, excluded_categories: set[str]) -> dict[str, dict[str, float]]:
    rates: dict[str, dict[str, float]] = {}
    for path in sorted(root.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        model = path.stem
        category_summary = payload.get("category_summary", {})
        for category, summary in category_summary.items():
            if category in excluded_categories:
                continue
            rate = summary.get("preferred_increase_rate")
            if rate is None:
                continue
            rates.setdefault(model, {})[category] = float(rate)
    return rates


def summarize(rates: dict[str, float]) -> dict[str, object]:
    if not rates:
        return {
            "avg": None,
            "max_category": None,
            "max_rate": None,
            "min_category": None,
            "min_rate": None,
            "n_categories": 0,
        }
    max_category, max_rate = max(rates.items(), key=lambda item: (item[1], item[0]))
    min_category, min_rate = min(rates.items(), key=lambda item: (item[1], item[0]))
    return {
        "avg": mean(rates.values()),
        "max_category": max_category,
        "max_rate": max_rate,
        "min_category": min_category,
        "min_rate": min_rate,
        "n_categories": len(rates),
    }


def ordered_models(original: dict[str, dict[str, float]], source: dict[str, dict[str, float]]) -> list[str]:
    model_names = set(original) | set(source)
    models = [name for name in DEFAULT_MODEL_ORDER if name in model_names]
    models.extend(sorted(model_names - set(models)))
    return models


def build_rows(
    original: dict[str, dict[str, float]],
    source: dict[str, dict[str, float]],
) -> list[dict[str, object]]:
    rows = []
    for model in ordered_models(original, source):
        original_summary = summarize(original.get(model, {}))
        source_summary = summarize(source.get(model, {}))
        row = {
            "model": model,
            "original_avg": original_summary["avg"],
            "original_max_category": original_summary["max_category"],
            "original_max_rate": original_summary["max_rate"],
            "original_min_category": original_summary["min_category"],
            "original_min_rate": original_summary["min_rate"],
            "original_n_categories": original_summary["n_categories"],
            "source_avg": source_summary["avg"],
            "source_max_category": source_summary["max_category"],
            "source_max_rate": source_summary["max_rate"],
            "source_min_category": source_summary["min_category"],
            "source_min_rate": source_summary["min_rate"],
            "source_n_categories": source_summary["n_categories"],
        }
        rows.append(row)
    return rows


def build_segmented_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    segmented_rows: list[dict[str, object]] = []
    for segment_name, prefix in (("VocalGrad", "original"), ("VCTK source", "source")):
        segment_rows: list[dict[str, object]] = []
        for idx, row in enumerate(rows):
            segment_rows.append(
                {
                    "clips": segment_name if idx == 0 else "",
                    "model": row["model"],
                    "avg": row[f"{prefix}_avg"],
                    "max_category": row[f"{prefix}_max_category"],
                    "max_rate": row[f"{prefix}_max_rate"],
                    "min_category": row[f"{prefix}_min_category"],
                    "min_rate": row[f"{prefix}_min_rate"],
                    "segment": segment_name,
                }
            )
        segmented_rows.extend(segment_rows)
    return segmented_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "clips",
        "model",
        "avg",
        "max_category",
        "max_rate",
        "min_category",
        "min_rate",
        "segment",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_latex(path: Path, rows: list[dict[str, object]], caption: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = latex_table_lines(rows=rows, caption=caption, label=label)
    path.write_text("\n".join(lines), encoding="utf-8")


def latex_table_lines(
    rows: list[dict[str, object]],
    caption: str,
    label: str,
) -> list[str]:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{latex_escape(label)}}}",
        r"\begin{tabular}{llrcrcr}",
        r"\toprule",
        r"Clips & Model & Avg. & Max category & Max value & Min category & Min value \\",
        r"\midrule",
    ]
    current_segment = None
    for row in rows:
        segment = str(row["segment"])
        if current_segment is not None and segment != current_segment:
            lines.append(r"\midrule")
        current_segment = segment
        lines.append(
            " & ".join(
                [
                    latex_escape(str(row["clips"])),
                    latex_escape(display_model_name(str(row["model"]))),
                    fmt_rate(row["avg"]),  # type: ignore[arg-type]
                    display_category_name(row["max_category"]),  # type: ignore[arg-type]
                    fmt_rate(row["max_rate"]),  # type: ignore[arg-type]
                    display_category_name(row["min_category"]),  # type: ignore[arg-type]
                    fmt_rate(row["min_rate"]),  # type: ignore[arg-type]
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return lines


def main() -> None:
    args = parse_args()
    excluded_categories = set(args.exclude_category)
    original = collect_original_rates_from_raw(args.original_raw_root, excluded_categories)
    if not original:
        original = collect_original_rates(args.original_analysis_root, excluded_categories)
    source = collect_source_rates_from_raw(args.source_raw_root, excluded_categories)
    if not source:
        source = collect_source_rates(args.source_analysis_root, excluded_categories)
    rows = build_rows(original, source)
    segmented_rows = build_segmented_rows(rows)
    write_csv(args.out_csv, segmented_rows)
    write_latex(args.out_tex, segmented_rows, args.caption, args.label)
    print(f"wrote CSV: {args.out_csv}")
    print(f"wrote LaTeX: {args.out_tex}")
    print(f"rows={len(segmented_rows)} excluded={sorted(excluded_categories)}")


if __name__ == "__main__":
    main()

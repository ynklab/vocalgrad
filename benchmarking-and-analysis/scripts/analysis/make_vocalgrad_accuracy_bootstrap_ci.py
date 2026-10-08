#!/usr/bin/env python3
"""Create bootstrap confidence intervals for VocalGrad model accuracies."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


DEFAULT_MODEL_ORDER = [
    "gemini-3-flash",
    "kimi-audio",
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
        "--out-csv",
        type=Path,
        default=Path("outputs/analysis/vocalgrad/default/model_accuracy_bootstrap_ci.csv"),
    )
    parser.add_argument(
        "--out-tex",
        type=Path,
        default=Path("outputs/analysis/vocalgrad/default/model_accuracy_bootstrap_ci.tex"),
    )
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--label", default="tab:vocalgrad-model-accuracy-bootstrap-ci")
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


def find_raw_file(raw_roots: list[Path], category: str, model_name: str) -> Path | None:
    for root in raw_roots:
        for suffix in (".jsonl", ".json"):
            path = root / category / f"{model_name}{suffix}"
            if path.exists():
                return path
    return None


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


def parse_direction_label(raw_text: object) -> str | None:
    tokens = LABEL_RE.findall(str(raw_text or "").strip().lower())
    labels = {
        "increase" if token in {"increase", "increases"} else "decrease"
        for token in tokens
    }
    if len(labels) != 1:
        return None
    return next(iter(labels))


def correctness_vector(rows: list[dict[str, Any]]) -> np.ndarray:
    values = []
    for row in rows:
        pred = parse_direction_label(row.get("raw_response"))
        gold = normalize_label(row.get("gold_label"))
        values.append(1 if pred is not None and gold is not None and pred == gold else 0)
    return np.asarray(values, dtype=np.int8)


def bootstrap_ci(values: np.ndarray, n_bootstrap: int, rng: np.random.Generator) -> tuple[float, float, float]:
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    accuracy = float(values.mean())
    # For a binary empirical distribution, resampling rows with replacement is
    # equivalent to drawing Binomial(n, p_hat) correct counts.
    samples = rng.binomial(values.size, accuracy, size=n_bootstrap) / values.size
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return accuracy, float(lo), float(hi)


def display_model_name(model_name: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_name, model_name.replace("-", " ").title())


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


def fmt_ci(point: float, lo: float, hi: float) -> str:
    return f"{point * 100:.1f} [{lo * 100:.1f}, {hi * 100:.1f}]"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "category", "n", "accuracy", "ci_low", "ci_high"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_latex(path: Path, rows: list[dict[str, object]], label: str) -> None:
    categories = DEFAULT_CATEGORY_ORDER
    row_by_model_category = {
        (str(row["model"]), str(row["category"])): row
        for row in rows
    }
    columns = [("model", "Model"), *[(category, CATEGORY_ABBREVIATIONS[category]) for category in categories]]
    alignment = "l" + "c" * (len(columns) - 1)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Model accuracies with bootstrap 95\% confidence intervals on VocalGrad by category. Parsing failures are counted as incorrect.}",
        rf"\label{{{latex_escape(label)}}}",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(latex_escape(header) for _, header in columns) + r" \\",
        r"\midrule",
    ]
    for model in DEFAULT_MODEL_ORDER:
        cells = [latex_escape(display_model_name(model))]
        for category in categories:
            row = row_by_model_category.get((model, category))
            if row is None:
                cells.append(r"\textemdash{}")
                continue
            cells.append(fmt_ci(float(row["accuracy"]), float(row["ci_low"]), float(row["ci_high"])))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    for model in DEFAULT_MODEL_ORDER:
        for category in DEFAULT_CATEGORY_ORDER:
            path = find_raw_file(args.raw_root, category, model)
            if path is None:
                continue
            values = correctness_vector(read_jsonl(path))
            accuracy, lo, hi = bootstrap_ci(values, args.n_bootstrap, rng)
            rows.append(
                {
                    "model": model,
                    "category": category,
                    "n": int(values.size),
                    "accuracy": accuracy,
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
    write_csv(args.out_csv, rows)
    write_latex(args.out_tex, rows, args.label)
    print(f"wrote CSV: {args.out_csv}")
    print(f"wrote LaTeX: {args.out_tex}")
    print(f"rows={len(rows)} n_bootstrap={args.n_bootstrap} seed={args.seed}")


if __name__ == "__main__":
    main()

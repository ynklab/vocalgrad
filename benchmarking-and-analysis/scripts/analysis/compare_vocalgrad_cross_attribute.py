from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate current VocalGrad cross-attribute analysis results."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/analysis/vocalgrad_cross_attribute/default"),
    )
    parser.add_argument("--md-out", type=Path, default=None)
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument("--matrix-csv-out", type=Path, default=None)
    return parser


def fmt_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return mean(values)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    args = build_parser().parse_args()
    root = args.root
    if args.md_out is None:
        args.md_out = root / "comparison.md"
    if args.csv_out is None:
        args.csv_out = root / "comparison.csv"
    if args.matrix_csv_out is None:
        args.matrix_csv_out = root / "matrix_mean_accuracy.csv"

    records: list[dict[str, object]] = []
    for path in sorted(root.glob("*/*/*.json")):
        if "_runs" in path.parts:
            continue
        data = json.loads(path.read_text())
        if data.get("evaluation_type") != "vocalgrad_cross_attribute":
            continue
        actual = str(data["actual_attribute"])
        prompt = str(data["prompt_attribute"])
        model = path.stem
        overall = data["overall"]
        accuracy = float(overall["accuracy"])
        record = {
            "actual_attribute": actual,
            "prompt_attribute": prompt,
            "model": model,
            "accuracy": accuracy,
            "is_diagonal": actual == prompt,
            "path": str(path),
        }
        records.append(record)

    if not records:
        raise SystemExit(f"no cross-attribute analysis files found under {root}")

    actuals = sorted({str(r["actual_attribute"]) for r in records})
    prompts = sorted({str(r["prompt_attribute"]) for r in records})
    models = sorted({str(r["model"]) for r in records})
    total_possible = len(actuals) * len(prompts)

    by_model: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_pair: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in records:
        by_model[str(record["model"])].append(record)
        by_pair[(str(record["actual_attribute"]), str(record["prompt_attribute"]))].append(float(record["accuracy"]))

    overall_rows: list[list[str]] = []
    csv_rows: list[dict[str, object]] = []
    for model in models:
        model_records = by_model[model]
        accuracies = [float(r["accuracy"]) for r in model_records]
        diag = [float(r["accuracy"]) for r in model_records if bool(r["is_diagonal"])]
        offdiag = [float(r["accuracy"]) for r in model_records if not bool(r["is_diagonal"])]
        best = max(model_records, key=lambda r: float(r["accuracy"]))
        worst = min(model_records, key=lambda r: float(r["accuracy"]))
        coverage = len(model_records)
        overall_rows.append([
            model,
            str(coverage),
            str(total_possible),
            fmt_float(safe_mean(accuracies)),
            fmt_float(safe_mean(diag)),
            fmt_float(safe_mean(offdiag)),
            f"{best['actual_attribute']} -> {best['prompt_attribute']} ({fmt_float(float(best['accuracy']))})",
            f"{worst['actual_attribute']} -> {worst['prompt_attribute']} ({fmt_float(float(worst['accuracy']))})",
        ])
        csv_rows.append({
            "model": model,
            "coverage": coverage,
            "total_possible": total_possible,
            "mean_accuracy": safe_mean(accuracies),
            "diagonal_mean_accuracy": safe_mean(diag),
            "off_diagonal_mean_accuracy": safe_mean(offdiag),
            "best_actual_attribute": best["actual_attribute"],
            "best_prompt_attribute": best["prompt_attribute"],
            "best_accuracy": float(best["accuracy"]),
            "worst_actual_attribute": worst["actual_attribute"],
            "worst_prompt_attribute": worst["prompt_attribute"],
            "worst_accuracy": float(worst["accuracy"]),
        })

    actual_counts = defaultdict(int)
    for record in records:
        actual_counts[str(record["actual_attribute"])] += 1
    coverage_rows = []
    per_actual_total_possible = len(prompts) * len(models)
    for actual in actuals:
        coverage_rows.append([actual, str(actual_counts[actual]), str(per_actual_total_possible)])

    matrix_headers = ["actual_attribute", *prompts]
    matrix_rows: list[list[str]] = []
    matrix_csv_rows: list[dict[str, object]] = []
    for actual in actuals:
        row = [actual]
        for prompt in prompts:
            val = safe_mean(by_pair[(actual, prompt)])
            row.append(fmt_float(val))
            matrix_csv_rows.append({
                "actual_attribute": actual,
                "prompt_attribute": prompt,
                "mean_accuracy": val,
                "num_models": len(by_pair[(actual, prompt)]),
            })
        matrix_rows.append(row)

    available_pairs = sorted(
        ((actual, prompt, safe_mean(values), len(values)) for (actual, prompt), values in by_pair.items()),
        key=lambda x: (-(x[2] if x[2] is not None else -1), x[0], x[1]),
    )
    top_pairs_rows = [
        [actual, prompt, fmt_float(acc), str(n)]
        for actual, prompt, acc, n in available_pairs[:12]
    ]
    bottom_pairs_rows = [
        [actual, prompt, fmt_float(acc), str(n)]
        for actual, prompt, acc, n in sorted(available_pairs, key=lambda x: ((x[2] if x[2] is not None else 999), x[0], x[1]))[:12]
    ]

    notes = []
    speaking_speed_cov = actual_counts.get("speaking_speed", 0)
    volume_cov = actual_counts.get("volume", 0)
    if speaking_speed_cov < per_actual_total_possible:
        notes.append(
            f"`speaking_speed` is incomplete: {speaking_speed_cov}/{per_actual_total_possible} files present."
        )
    if volume_cov < per_actual_total_possible:
        notes.append(
            f"`volume` is incomplete: {volume_cov}/{per_actual_total_possible} files present."
        )
    for model in models:
        cov = len(by_model[model])
        if cov < total_possible:
            notes.append(f"`{model}` coverage is {cov}/{total_possible} pairs.")

    md_parts = [
        "# VocalGrad Cross-Attribute Comparison",
        "",
        f"Root: `{root}`",
        "",
        f"Collected `{len(records)}` analysis files across `{len(actuals)}` actual attributes, `{len(prompts)}` prompt attributes, and `{len(models)}` models.",
        "",
        "## Model Summary",
        "",
        markdown_table(
            [
                "Model",
                "Coverage",
                "Possible",
                "Mean Acc",
                "Diag Mean",
                "Off-Diag Mean",
                "Best Pair",
                "Worst Pair",
            ],
            overall_rows,
        ),
        "",
        "## Coverage By Actual Attribute",
        "",
        markdown_table(["Actual Attribute", "Files Present", "Possible"], coverage_rows),
        "",
        "## Mean Accuracy Matrix Across Available Models",
        "",
        markdown_table(matrix_headers, matrix_rows),
        "",
        "## Top Available Pairs",
        "",
        markdown_table(["Actual", "Prompt", "Mean Acc", "Models"], top_pairs_rows),
        "",
        "## Bottom Available Pairs",
        "",
        markdown_table(["Actual", "Prompt", "Mean Acc", "Models"], bottom_pairs_rows),
    ]
    if notes:
        md_parts.extend(["", "## Coverage Notes", ""])
        md_parts.extend([f"- {note}" for note in notes])

    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text("\n".join(md_parts) + "\n")

    with args.csv_out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "coverage",
                "total_possible",
                "mean_accuracy",
                "diagonal_mean_accuracy",
                "off_diagonal_mean_accuracy",
                "best_actual_attribute",
                "best_prompt_attribute",
                "best_accuracy",
                "worst_actual_attribute",
                "worst_prompt_attribute",
                "worst_accuracy",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    with args.matrix_csv_out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["actual_attribute", "prompt_attribute", "mean_accuracy", "num_models"],
        )
        writer.writeheader()
        writer.writerows(matrix_csv_rows)

    print(f"wrote {args.md_out}")
    print(f"wrote {args.csv_out}")
    print(f"wrote {args.matrix_csv_out}")


if __name__ == "__main__":
    main()

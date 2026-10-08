from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT_MODEL_ID = "moonshotai/Kimi-Audio-7B-Instruct"
from plic.kimi_finetune import DEFAULT_KIMI_FINETUNE_ANALYSIS_ROOT, model_stem_for, resolve_categories


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare Kimi fine-tuning domain generalization results across categories and scopes.")
    parser.add_argument("--root", type=Path, default=DEFAULT_KIMI_FINETUNE_ANALYSIS_ROOT)
    parser.add_argument("--run-name", default="default")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--categories", nargs="+", help="Category order for saved-result analysis without audio")
    parser.add_argument("--test-dataset-root", type=Path, default=Path("datasets/vocalgrad/test"))
    parser.add_argument("--base-root", type=Path, default=Path("outputs/analysis/vocalgrad/default"))
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser


def _load_base_metrics(base_root: Path, categories: list[str]) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    base_path = base_root
    for category in categories:
        summary_path = base_path / category / "kimi-audio.json"
        if not summary_path.exists():
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        overall = payload.get("overall")
        if not isinstance(overall, dict):
            continue
        accuracy = overall.get("accuracy")
        if isinstance(accuracy, (int, float)):
            metrics[category] = {"accuracy": float(accuracy)}
    return metrics

def load_records(root: Path, run_name: str, model_stem: str, categories: list[str], base_metrics: dict[str, dict[str, float]]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    run_root = root / model_stem / run_name
    if not run_root.exists():
        raise SystemExit(f"run root not found: {run_root}")

    for scope_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        scope = scope_dir.name
        for train_dir in sorted(path for path in scope_dir.iterdir() if path.is_dir()):
            train_category = train_dir.name
            for test_category in categories:
                summary_path = train_dir / f"{test_category}.json"
                if not summary_path.exists():
                    continue
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
                overall = payload.get("overall")
                if not isinstance(overall, dict):
                    continue
                accuracy = overall.get("accuracy")
                balanced_accuracy = overall.get("balanced_accuracy")
                accuracy_up = overall.get("accuracy_up")
                accuracy_down = overall.get("accuracy_down")
                if not isinstance(accuracy, (int, float)) or not isinstance(balanced_accuracy, (int, float)):
                    continue
                base_accuracy = None
                if test_category in base_metrics:
                    base_accuracy = base_metrics[test_category].get("accuracy")
                records.append(
                    {
                        "schema_version": "vocalgrad-all-clips-v1",
                        "metric": "accuracy_all_clips",
                        "scope": scope,
                        "train_category": train_category,
                        "test_category": test_category,
                        "accuracy": float(accuracy),
                        "balanced_accuracy": float(balanced_accuracy),
                        "accuracy_up": float(accuracy_up) if isinstance(accuracy_up, (int, float)) else None,
                        "accuracy_down": float(accuracy_down) if isinstance(accuracy_down, (int, float)) else None,
                        "is_in_domain": train_category == test_category,
                        "base_accuracy": float(base_accuracy) if isinstance(base_accuracy, (int, float)) else None,
                        "delta_vs_base_accuracy": float(accuracy - base_accuracy) if isinstance(base_accuracy, (int, float)) else None,
                    }
                )
    if not records:
        raise SystemExit(f"no fine-tuning summaries found under {run_root}")
    return pd.DataFrame.from_records(records)


def write_markdown(df: pd.DataFrame, out_path: Path) -> None:
    def format_table(table_df: pd.DataFrame) -> list[str]:
        headers = [str(column) for column in table_df.columns]
        rows: list[list[str]] = []
        for _, row in table_df.iterrows():
            values: list[str] = []
            for column in table_df.columns:
                value = row[column]
                if isinstance(value, float):
                    values.append(f"{value:.4f}")
                else:
                    values.append(str(value))
            rows.append(values)

        widths = [len(header) for header in headers]
        for row_values in rows:
            widths = [max(width, len(value)) for width, value in zip(widths, row_values)]

        def render(values: list[str]) -> str:
            padded = [value.ljust(width) for value, width in zip(values, widths)]
            return "| " + " | ".join(padded) + " |"

        separator = "| " + " | ".join("-" * width for width in widths) + " |"
        return [render(headers), separator] + [render(row_values) for row_values in rows]

    lines: list[str] = []
    lines.append("# Kimi Fine-Tuning Domain Generalization")
    lines.append("")
    lines.append(f"Rows summarized: `{len(df)}`")
    lines.append("")

    scope_summary = (
        df.groupby(["scope", "is_in_domain"], as_index=False)[["accuracy", "balanced_accuracy", "delta_vs_base_accuracy"]]
        .mean(numeric_only=True)
        .sort_values(["scope", "is_in_domain"])
    )
    lines.append("## Scope Summary")
    lines.append("")
    lines.extend(format_table(scope_summary))
    lines.append("")

    best_runs = (
        df.groupby(["scope", "train_category"], as_index=False)[["accuracy", "balanced_accuracy"]]
        .mean(numeric_only=True)
        .sort_values(["scope", "balanced_accuracy"], ascending=[True, False])
    )
    lines.append("## Mean Across Test Categories")
    lines.append("")
    lines.extend(format_table(best_runs))
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    model_stem = model_stem_for(args.model_id)
    categories = args.categories or resolve_categories(args.test_dataset_root, None)
    out_dir = args.out_dir or (args.root / "_comparisons" / args.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_metrics = _load_base_metrics(args.base_root, categories)
    df = load_records(args.root, args.run_name, model_stem, categories, base_metrics)
    df = df.sort_values(["scope", "train_category", "test_category"]).reset_index(drop=True)
    df.to_csv(out_dir / "generalization_records.csv", index=False)

    aggregate = (
        df.groupby(["scope", "train_category", "is_in_domain"], as_index=False)[["accuracy", "balanced_accuracy", "delta_vs_base_accuracy"]]
        .mean(numeric_only=True)
        .sort_values(["scope", "train_category", "is_in_domain"])
        .reset_index(drop=True)
    )
    aggregate.to_csv(out_dir / "generalization_summary.csv", index=False)
    write_markdown(df, out_dir / "comparison.md")
    print(f"wrote Kimi fine-tuning generalization comparison to {out_dir}")


if __name__ == "__main__":
    main()

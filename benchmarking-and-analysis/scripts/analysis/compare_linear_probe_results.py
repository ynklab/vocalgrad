from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

LINEAR_PROBE_STAGES = ("mel", "pre_lm_audio", "lm_audio", "lm_text")
DEFAULT_ROOT = REPO_ROOT / "outputs" / "analysis" / "linear_probe"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate category-specific linear-probe results across models and stages."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--run-name", default="default")
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def _discover_results(root: Path, run_name: str) -> dict[str, dict[str, Any]]:
    discovered: dict[str, dict[str, Any]] = {}
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("_"):
            continue
        run_dir = model_dir / run_name
        summary_path = run_dir / "results.json"
        if not summary_path.exists():
            continue
        categories: dict[str, dict[str, Any]] = {}
        for category_path in sorted(run_dir.glob("*.json")):
            if category_path.name == "results.json":
                continue
            payload = _load_json(category_path)
            category = payload.get("category")
            if isinstance(category, str):
                categories[category] = payload
        summary_payload = _load_json(summary_path)
        summary_payload["category_results"] = categories
        discovered[model_dir.name] = summary_payload
    if not discovered:
        raise FileNotFoundError(f"no linear probe results found under {root} for run-name={run_name}")
    return discovered


def _metric(payload: dict[str, Any], stage: str, key: str) -> float | None:
    stages = payload.get("stages")
    if not isinstance(stages, dict):
        return None
    stage_section = stages.get(stage)
    if not isinstance(stage_section, dict):
        return None
    value = stage_section.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _category_metric(results: dict[str, dict[str, Any]], model_stem: str, category: str, stage: str, key: str) -> float | None:
    payload = results.get(model_stem, {})
    category_results = payload.get("category_results", {})
    if not isinstance(category_results, dict):
        return None
    category_payload = category_results.get(category)
    if not isinstance(category_payload, dict):
        return None
    return _metric(category_payload, stage, key)


def _write_csv(out_path: Path, results: dict[str, dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model_stem",
            "category",
            "stage",
            "accuracy",
            "balanced_accuracy",
            "accuracy_up",
            "accuracy_down",
            "n_train",
            "n_test",
        ])
        for model_stem, payload in sorted(results.items()):
            category_results = payload.get("category_results", {})
            if not isinstance(category_results, dict):
                continue
            for category, category_payload in sorted(category_results.items()):
                for stage in LINEAR_PROBE_STAGES:
                    stage_data = category_payload.get("stages", {}).get(stage, {})
                    writer.writerow(
                        [
                            model_stem,
                            category,
                            stage,
                            stage_data.get("accuracy"),
                            stage_data.get("balanced_accuracy"),
                            stage_data.get("accuracy_up"),
                            stage_data.get("accuracy_down"),
                            stage_data.get("n_train"),
                            stage_data.get("n_test"),
                        ]
                    )


def _overall_stage_means(results: dict[str, dict[str, Any]]) -> list[tuple[str, float | None, float | None]]:
    rows: list[tuple[str, float | None, float | None]] = []
    for stage in LINEAR_PROBE_STAGES:
        acc_values = [_metric(payload, stage, "accuracy") for payload in results.values()]
        bal_values = [_metric(payload, stage, "balanced_accuracy") for payload in results.values()]
        acc_clean = [v for v in acc_values if v is not None]
        bal_clean = [v for v in bal_values if v is not None]
        rows.append(
            (
                stage,
                mean(acc_clean) if acc_clean else None,
                mean(bal_clean) if bal_clean else None,
            )
        )
    return rows


def _within_model_rankings(results: dict[str, dict[str, Any]]) -> dict[str, list[tuple[str, float | None]]]:
    rankings: dict[str, list[tuple[str, float | None]]] = {}
    for model_stem, payload in sorted(results.items()):
        rows = [(stage, _metric(payload, stage, "balanced_accuracy")) for stage in LINEAR_PROBE_STAGES]
        rows.sort(key=lambda item: (item[1] is not None, item[1]), reverse=True)
        rankings[model_stem] = rows
    return rankings


def _shared_categories(results: dict[str, dict[str, Any]]) -> list[str]:
    category_sets: list[set[str]] = []
    for payload in results.values():
        category_results = payload.get("category_results", {})
        if isinstance(category_results, dict):
            category_sets.append(set(category_results))
    if not category_sets:
        return []
    return sorted(set.intersection(*category_sets))


def _write_md(out_path: Path, results: dict[str, dict[str, Any]], run_name: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_stems = sorted(results)
    stage_means = _overall_stage_means(results)
    rankings = _within_model_rankings(results)
    shared_categories = _shared_categories(results)

    lines: list[str] = []
    lines.append("# Linear Probe Comparison")
    lines.append("")
    lines.append(f"Run name: `{run_name}`")
    lines.append("")
    lines.append("Metrics below are means over category-specific probes.")
    lines.append("")
    lines.append("## Mean Balanced Accuracy By Stage")
    lines.append("")
    lines.append("| Stage | " + " | ".join(model_stems) + " |")
    lines.append("| --- | " + " | ".join(["---"] * len(model_stems)) + " |")
    for stage in LINEAR_PROBE_STAGES:
        cells = [_format_float(_metric(results[model_stem], stage, "balanced_accuracy")) for model_stem in model_stems]
        lines.append("| " + " | ".join([stage, *cells]) + " |")

    lines.append("")
    lines.append("## Mean Accuracy By Stage")
    lines.append("")
    lines.append("| Stage | " + " | ".join(model_stems) + " |")
    lines.append("| --- | " + " | ".join(["---"] * len(model_stems)) + " |")
    for stage in LINEAR_PROBE_STAGES:
        cells = [_format_float(_metric(results[model_stem], stage, "accuracy")) for model_stem in model_stems]
        lines.append("| " + " | ".join([stage, *cells]) + " |")

    lines.append("")
    lines.append("## Mean By Stage")
    lines.append("")
    lines.append("| Stage | Mean Accuracy | Mean Balanced Accuracy |")
    lines.append("| --- | --- | --- |")
    for stage, mean_acc, mean_bal in stage_means:
        lines.append(f"| {stage} | {_format_float(mean_acc)} | {_format_float(mean_bal)} |")

    lines.append("")
    lines.append("## Within-Model Stage Ranking")
    lines.append("")
    for model_stem in model_stems:
        lines.append(f"### {model_stem}")
        lines.append("")
        lines.append("| Rank | Stage | Mean Balanced Accuracy |")
        lines.append("| --- | --- | --- |")
        for idx, (stage, value) in enumerate(rankings[model_stem], start=1):
            lines.append(f"| {idx} | {stage} | {_format_float(value)} |")
        lines.append("")

    if shared_categories:
        lines.append("## Per-Category Balanced Accuracy")
        lines.append("")
        for category in shared_categories:
            lines.append(f"### {category}")
            lines.append("")
            lines.append("| Stage | " + " | ".join(model_stems) + " |")
            lines.append("| --- | " + " | ".join(["---"] * len(model_stems)) + " |")
            for stage in LINEAR_PROBE_STAGES:
                cells = [
                    _format_float(_category_metric(results, model_stem, category, stage, "balanced_accuracy"))
                    for model_stem in model_stems
                ]
                lines.append("| " + " | ".join([stage, *cells]) + " |")
            lines.append("")

    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    results = _discover_results(args.root, args.run_name)
    out_dir = args.out_dir or (args.root / "_comparisons" / args.run_name)
    _write_csv(out_dir / "comparison.csv", results)
    _write_md(out_dir / "comparison.md", results, args.run_name)
    print(f"wrote linear probe comparison to {out_dir}")


if __name__ == "__main__":
    main()

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

DEFAULT_ROOT = REPO_ROOT / "outputs" / "analysis" / "linear_probe_lm_text_layers"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate layer-wise lm_text linear-probe results across models."
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
        raise FileNotFoundError(f"no layer-wise lm_text probe results found under {root} for run-name={run_name}")
    return discovered


def _layer_metric(payload: dict[str, Any], layer_idx: int, key: str) -> float | None:
    layers = payload.get("layers")
    if not isinstance(layers, dict):
        return None
    section = layers.get(str(layer_idx))
    if not isinstance(section, dict):
        return None
    value = section.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _category_metric(
    results: dict[str, dict[str, Any]],
    model_stem: str,
    category: str,
    layer_idx: int,
    key: str,
) -> float | None:
    payload = results.get(model_stem, {})
    category_results = payload.get("category_results", {})
    if not isinstance(category_results, dict):
        return None
    category_payload = category_results.get(category)
    if not isinstance(category_payload, dict):
        return None
    return _layer_metric(category_payload, layer_idx, key)


def _all_layers(results: dict[str, dict[str, Any]]) -> list[int]:
    found: set[int] = set()
    for payload in results.values():
        layers = payload.get("layers", {})
        if not isinstance(layers, dict):
            continue
        for key in layers:
            if str(key).isdigit():
                found.add(int(key))
    return sorted(found)


def _write_csv(out_path: Path, results: dict[str, dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model_stem",
                "category",
                "layer",
                "accuracy",
                "balanced_accuracy",
                "accuracy_up",
                "accuracy_down",
                "n_train",
                "n_test",
            ]
        )
        for model_stem, payload in sorted(results.items()):
            category_results = payload.get("category_results", {})
            if not isinstance(category_results, dict):
                continue
            for category, category_payload in sorted(category_results.items()):
                layers = category_payload.get("layers", {})
                if not isinstance(layers, dict):
                    continue
                for layer_key, layer_section in sorted(layers.items(), key=lambda item: int(item[0])):
                    if not isinstance(layer_section, dict):
                        continue
                    writer.writerow(
                        [
                            model_stem,
                            category,
                            int(layer_key),
                            layer_section.get("accuracy"),
                            layer_section.get("balanced_accuracy"),
                            layer_section.get("accuracy_up"),
                            layer_section.get("accuracy_down"),
                            layer_section.get("n_train"),
                            layer_section.get("n_test"),
                        ]
                    )


def _best_layer(payload: dict[str, Any], key: str) -> tuple[int | None, float | None]:
    layers = payload.get("layers", {})
    if not isinstance(layers, dict):
        return None, None
    best: tuple[int | None, float | None] = (None, None)
    for layer_key, layer_section in layers.items():
        if not str(layer_key).isdigit() or not isinstance(layer_section, dict):
            continue
        value = layer_section.get(key)
        if not isinstance(value, (int, float)):
            continue
        layer_idx = int(layer_key)
        if best[1] is None or float(value) > float(best[1]):
            best = (layer_idx, float(value))
    return best


def _shared_categories(results: dict[str, dict[str, Any]]) -> list[str]:
    category_sets: list[set[str]] = []
    for payload in results.values():
        category_results = payload.get("category_results", {})
        if isinstance(category_results, dict):
            category_sets.append(set(category_results))
    if not category_sets:
        return []
    return sorted(set.intersection(*category_sets))


def _mean_over_models(results: dict[str, dict[str, Any]], layer_idx: int, key: str) -> float | None:
    values = [_layer_metric(payload, layer_idx, key) for payload in results.values()]
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return mean(clean)


def _write_md(out_path: Path, results: dict[str, dict[str, Any]], run_name: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_stems = sorted(results)
    all_layers = _all_layers(results)
    shared_categories = _shared_categories(results)

    lines: list[str] = []
    lines.append("# Layer-Wise LM-Text Linear Probe Comparison")
    lines.append("")
    lines.append(f"Run name: `{run_name}`")
    lines.append("")
    lines.append("Metrics below are means over category-specific probes.")
    lines.append("")

    lines.append("## Best Layer By Model (Balanced Accuracy)")
    lines.append("")
    lines.append("| Model | Best Layer | Balanced Accuracy |")
    lines.append("| --- | --- | --- |")
    for model_stem in model_stems:
        best_layer, best_value = _best_layer(results[model_stem], "balanced_accuracy")
        layer_text = str(best_layer) if best_layer is not None else "-"
        lines.append(f"| {model_stem} | {layer_text} | {_format_float(best_value)} |")

    lines.append("")
    lines.append("## Mean Balanced Accuracy By Layer")
    lines.append("")
    lines.append("| Layer | " + " | ".join(model_stems) + " | Mean |")
    lines.append("| --- | " + " | ".join(["---"] * len(model_stems)) + " | --- |")
    for layer_idx in all_layers:
        per_model = [_format_float(_layer_metric(results[model_stem], layer_idx, "balanced_accuracy")) for model_stem in model_stems]
        mean_text = _format_float(_mean_over_models(results, layer_idx, "balanced_accuracy"))
        lines.append("| " + " | ".join([str(layer_idx), *per_model, mean_text]) + " |")

    lines.append("")
    lines.append("## Mean Accuracy By Layer")
    lines.append("")
    lines.append("| Layer | " + " | ".join(model_stems) + " | Mean |")
    lines.append("| --- | " + " | ".join(["---"] * len(model_stems)) + " | --- |")
    for layer_idx in all_layers:
        per_model = [_format_float(_layer_metric(results[model_stem], layer_idx, "accuracy")) for model_stem in model_stems]
        mean_text = _format_float(_mean_over_models(results, layer_idx, "accuracy"))
        lines.append("| " + " | ".join([str(layer_idx), *per_model, mean_text]) + " |")

    if shared_categories:
        lines.append("")
        lines.append("## Per-Category Balanced Accuracy")
        lines.append("")
        for category in shared_categories:
            lines.append(f"### {category}")
            lines.append("")
            lines.append("| Layer | " + " | ".join(model_stems) + " |")
            lines.append("| --- | " + " | ".join(["---"] * len(model_stems)) + " |")
            for layer_idx in all_layers:
                cells = [
                    _format_float(_category_metric(results, model_stem, category, layer_idx, "balanced_accuracy"))
                    for model_stem in model_stems
                ]
                lines.append("| " + " | ".join([str(layer_idx), *cells]) + " |")
            lines.append("")

    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    results = _discover_results(args.root, args.run_name)
    out_dir = args.out_dir or (args.root / "_comparisons" / args.run_name)
    _write_csv(out_dir / "comparison.csv", results)
    _write_md(out_dir / "comparison.md", results, args.run_name)
    print(f"wrote layer-wise lm_text comparison to {out_dir}")


if __name__ == "__main__":
    main()

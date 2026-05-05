from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

import numpy as np
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from plic.linear_probe import (  # noqa: E402
    DEFAULT_LINEAR_PROBE_ANALYSIS_ROOT,
    LINEAR_PROBE_STAGES,
    evaluate_probe,
    train_logistic_probe,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate category-specific PyTorch linear probes on cached VocalGrad features."
    )
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--model-stem", required=True)
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--analysis-root", type=Path, default=REPO_ROOT / DEFAULT_LINEAR_PROBE_ANALYSIS_ROOT)
    parser.add_argument("--run-name", default="default")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--max-iter", type=int, default=None, help="Deprecated alias for --epochs.")
    parser.add_argument("--c-value", type=float, default=None, help="Deprecated sklearn-era argument; ignored.")
    return parser


def _load_category_npz(root: Path, model_stem: str, category: str) -> dict[str, np.ndarray]:
    path = root / model_stem / f"{category}.npz"
    if not path.exists():
        raise FileNotFoundError(f"feature cache not found: {path}")
    loaded = np.load(path, allow_pickle=False)
    return {key: loaded[key] for key in loaded.files}


def _resolve_categories(train_root: Path, test_root: Path, model_stem: str, categories: list[str] | None) -> list[str]:
    if categories:
        return categories
    train_dir = train_root / model_stem
    test_dir = test_root / model_stem
    train_categories = {path.stem for path in train_dir.glob("*.npz")}
    test_categories = {path.stem for path in test_dir.glob("*.npz")}
    resolved = sorted(train_categories & test_categories)
    if not resolved:
        raise FileNotFoundError(
            f"no overlapping category feature caches under {train_dir} and {test_dir}"
        )
    return resolved


def _stage_array(record: dict[str, np.ndarray], stage: str) -> np.ndarray:
    return record[f"{stage}_features"]


def _mean_metric(category_results: dict[str, dict[str, dict[str, float | int | None]]], stage: str, key: str) -> float | None:
    values: list[float] = []
    for payload in category_results.values():
        stage_section = payload.get("stages", {}).get(stage, {})
        value = stage_section.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    if not values:
        return None
    return mean(values)


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()
    categories = _resolve_categories(args.train_root, args.test_root, args.model_stem, args.categories)
    epochs = args.max_iter if args.max_iter is not None else args.epochs

    analysis_dir = args.analysis_root / args.model_stem / args.run_name
    analysis_dir.mkdir(parents=True, exist_ok=True)

    category_results: dict[str, dict[str, object]] = {}
    for category in categories:
        train_record = _load_category_npz(args.train_root, args.model_stem, category)
        test_record = _load_category_npz(args.test_root, args.model_stem, category)

        stage_results: dict[str, dict[str, float | int | None]] = {}
        for stage in LINEAR_PROBE_STAGES:
            train_x = _stage_array(train_record, stage)
            train_y = train_record["label"].astype(np.int64)
            test_x = _stage_array(test_record, stage)
            test_y = test_record["label"].astype(np.int64)
            probe = train_logistic_probe(
                train_x,
                train_y,
                random_state=args.seed,
                max_iter=epochs,
                device=args.device,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                show_progress=args.show_progress,
                progress_label=f"{args.model_stem}:{category}:{stage}",
            )
            metrics = evaluate_probe(probe, train_x, train_y, test_x, test_y)
            stage_results[stage] = metrics.to_dict()

        category_payload: dict[str, object] = {
            "model_stem": args.model_stem,
            "category": category,
            "train_root": str(args.train_root),
            "test_root": str(args.test_root),
            "seed": args.seed,
            "epochs": epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "device": args.device,
            "stages": stage_results,
        }
        category_path = analysis_dir / f"{category}.json"
        category_path.write_text(json.dumps(category_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        category_results[category] = category_payload

    summary_stages: dict[str, dict[str, float | int | None]] = {}
    for stage in LINEAR_PROBE_STAGES:
        summary_stages[stage] = {
            "accuracy": _mean_metric(category_results, stage, "accuracy"),
            "balanced_accuracy": _mean_metric(category_results, stage, "balanced_accuracy"),
            "accuracy_up": _mean_metric(category_results, stage, "accuracy_up"),
            "accuracy_down": _mean_metric(category_results, stage, "accuracy_down"),
            "n_train": None,
            "n_test": None,
        }

    out_path = analysis_dir / "results.json"
    payload = {
        "model_stem": args.model_stem,
        "categories": categories,
        "train_root": str(args.train_root),
        "test_root": str(args.test_root),
        "seed": args.seed,
        "epochs": epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "device": args.device,
        "training_backend": "pytorch_linear",
        "aggregation": "mean_over_category_specific_probes",
        "stages": summary_stages,
        "per_category_files": {category: f"{category}.json" for category in categories},
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote linear-probe results to {out_path}")


if __name__ == "__main__":
    main()

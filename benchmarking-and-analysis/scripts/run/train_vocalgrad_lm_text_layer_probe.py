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

from plic.linear_probe import evaluate_probe, train_logistic_probe  # noqa: E402


DEFAULT_LAYER_ANALYSIS_ROOT = REPO_ROOT / "outputs" / "analysis" / "linear_probe_lm_text_layers"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate layer-wise lm_text linear probes on cached VocalGrad features."
    )
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--model-stem", required=True)
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_LAYER_ANALYSIS_ROOT)
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
        raise FileNotFoundError(f"no overlapping category feature caches under {train_dir} and {test_dir}")
    return resolved


def _resolve_layer_indices(record: dict[str, np.ndarray], n_layers: int) -> list[int]:
    layer_index = record.get("layer_index")
    if layer_index is None:
        return list(range(n_layers))
    values = np.asarray(layer_index).astype(np.int64, copy=False).tolist()
    if len(values) != n_layers:
        raise ValueError(f"layer_index length mismatch: expected {n_layers}, got {len(values)}")
    return [int(value) for value in values]


def _mean_layer_metric(
    category_results: dict[str, dict[str, object]],
    layer_idx: int,
    key: str,
) -> float | None:
    values: list[float] = []
    for payload in category_results.values():
        layers = payload.get("layers", {})
        if not isinstance(layers, dict):
            continue
        section = layers.get(str(layer_idx), {})
        if not isinstance(section, dict):
            continue
        value = section.get(key)
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
    expected_layers: int | None = None
    expected_layer_indices: list[int] | None = None

    for category in categories:
        train_record = _load_category_npz(args.train_root, args.model_stem, category)
        test_record = _load_category_npz(args.test_root, args.model_stem, category)

        train_y = train_record["label"].astype(np.int64)
        test_y = test_record["label"].astype(np.int64)
        train_layers = train_record["lm_text_layer_features"].astype(np.float32, copy=False)
        test_layers = test_record["lm_text_layer_features"].astype(np.float32, copy=False)

        if train_layers.ndim != 3 or test_layers.ndim != 3:
            raise ValueError(
                f"expected lm_text_layer_features shape [N, L, D] for category={category}, "
                f"got train={train_layers.shape} test={test_layers.shape}"
            )

        n_layers = int(train_layers.shape[1])
        if test_layers.shape[1] != n_layers:
            raise ValueError(
                f"layer count mismatch for category={category}: train={train_layers.shape[1]} test={test_layers.shape[1]}"
            )
        train_layer_indices = _resolve_layer_indices(train_record, n_layers)
        test_layer_indices = _resolve_layer_indices(test_record, n_layers)
        if train_layer_indices != test_layer_indices:
            raise ValueError(
                f"layer indices mismatch for category={category}: train={train_layer_indices} test={test_layer_indices}"
            )
        if expected_layers is None:
            expected_layers = n_layers
        elif expected_layers != n_layers:
            raise ValueError(
                f"layer count mismatch across categories: expected={expected_layers} got={n_layers} ({category})"
            )
        if expected_layer_indices is None:
            expected_layer_indices = train_layer_indices
        elif expected_layer_indices != train_layer_indices:
            raise ValueError(
                f"layer indices mismatch across categories: expected={expected_layer_indices} got={train_layer_indices} ({category})"
            )

        layer_results: dict[str, dict[str, float | int | None]] = {}
        for layer_pos, layer_idx in enumerate(train_layer_indices):
            train_x = train_layers[:, layer_pos, :]
            test_x = test_layers[:, layer_pos, :]
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
                progress_label=f"{args.model_stem}:{category}:layer{layer_idx}",
            )
            metrics = evaluate_probe(probe, train_x, train_y, test_x, test_y)
            layer_results[str(layer_idx)] = metrics.to_dict()

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
            "n_layers": n_layers,
            "layer_indices": train_layer_indices,
            "layers": layer_results,
        }
        category_path = analysis_dir / f"{category}.json"
        category_path.write_text(json.dumps(category_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        category_results[category] = category_payload

    if expected_layers is None:
        raise RuntimeError("no categories were processed")
    if expected_layer_indices is None:
        raise RuntimeError("no layer indices were resolved")

    summary_layers: dict[str, dict[str, float | int | None]] = {}
    for layer_idx in expected_layer_indices:
        summary_layers[str(layer_idx)] = {
            "accuracy": _mean_layer_metric(category_results, layer_idx, "accuracy"),
            "balanced_accuracy": _mean_layer_metric(category_results, layer_idx, "balanced_accuracy"),
            "accuracy_up": _mean_layer_metric(category_results, layer_idx, "accuracy_up"),
            "accuracy_down": _mean_layer_metric(category_results, layer_idx, "accuracy_down"),
            "n_train": None,
            "n_test": None,
        }

    out_path = analysis_dir / "results.json"
    payload = {
        "model_stem": args.model_stem,
        "feature_type": "lm_text_layer_features",
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
        "n_layers": expected_layers,
        "layer_indices": expected_layer_indices,
        "layers": summary_layers,
        "per_category_files": {category: f"{category}.json" for category in categories},
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote layer-wise lm_text probe results to {out_path}")


if __name__ == "__main__":
    main()

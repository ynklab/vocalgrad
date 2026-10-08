from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from plic.linear_probe import evaluate_probe, train_logistic_probe  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Kimi rebuttal probes for audio segments and text positions."
    )
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--model-stem", default="kimi-audio")
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument(
        "--input-order", choices=("query_to_audio", "audio_to_query"), required=True
    )
    parser.add_argument("--run-name", default="default")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--rebuild-summary-only",
        action="store_true",
        help="Rebuild results.json from completed category JSON files without training.",
    )
    return parser


def _load(root: Path, stem: str, category: str) -> dict[str, np.ndarray]:
    path = root / stem / f"{category}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=False)
    return {key: data[key] for key in data.files}


def _categories(
    train_root: Path, test_root: Path, stem: str, requested: list[str] | None
) -> list[str]:
    if requested:
        return requested
    return sorted(
        {path.stem for path in (train_root / stem).glob("*.npz")}
        & {path.stem for path in (test_root / stem).glob("*.npz")}
    )


def _metrics(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    args: argparse.Namespace,
    label: str,
) -> dict[str, float | int | None]:
    probe = train_logistic_probe(
        train_x,
        train_y,
        random_state=args.seed,
        max_iter=args.epochs,
        device=args.device,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        progress_label=label,
    )
    return evaluate_probe(probe, train_x, train_y, test_x, test_y).to_dict()


def _mean_payload(
    category_payloads: dict[str, dict[str, object]], stream: str
) -> dict[str, object]:
    # Preserve the full position/layer grid while averaging category-specific probe scores.
    values: dict[tuple[str, str], list[float]] = {}
    for payload in category_payloads.values():
        section = payload[stream]
        assert isinstance(section, dict)
        for unit, layers in section.items():
            assert isinstance(layers, dict)
            for layer, metrics in layers.items():
                assert isinstance(metrics, dict)
                value = metrics.get("balanced_accuracy")
                if isinstance(value, (int, float)):
                    values.setdefault((str(unit), str(layer)), []).append(float(value))
    return {
        unit: {
            layer: {"balanced_accuracy": mean(values[(unit, layer)])}
            for unit2, layer in values
            if unit2 == unit
        }
        for unit in sorted({unit for unit, _ in values})
    }


def main() -> None:
    args = build_parser().parse_args()
    available_categories = _categories(
        args.train_root, args.test_root, args.model_stem, None
    )
    categories = _categories(
        args.train_root, args.test_root, args.model_stem, args.categories
    )
    if not categories:
        raise ValueError("no overlapping feature categories")
    if args.rebuild_summary_only:
        categories = []
    out_dir = args.analysis_root / args.input_order / args.model_stem / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    category_payloads: dict[str, dict[str, object]] = {}
    # A category may be probed in a separate job. Reuse its completed payload so
    # each invocation rewrites results.json as the aggregate over all available
    # categories, not merely the categories requested in this invocation.
    for category in available_categories:
        if category in categories:
            continue
        existing_path = out_dir / f"{category}.json"
        if existing_path.exists():
            category_payloads[category] = json.loads(
                existing_path.read_text(encoding="utf-8")
            )
    for category in categories:
        train, test = (
            _load(args.train_root, args.model_stem, category),
            _load(args.test_root, args.model_stem, category),
        )
        train_y, test_y = (
            train["label"].astype(np.int64),
            test["label"].astype(np.int64),
        )
        text_train, text_test = (
            train["text_layer_features"],
            test["text_layer_features"],
        )
        if text_train.shape[1:] != text_test.shape[1:]:
            raise ValueError(f"train/test shape mismatch for {category}")
        shared_indices = train["shared_audio_layer_index"].astype(np.int64).tolist()
        text_indices = train["text_layer_index"].astype(np.int64).tolist()
        if (
            shared_indices != test["shared_audio_layer_index"].astype(np.int64).tolist()
            or text_indices != test["text_layer_index"].astype(np.int64).tolist()
        ):
            raise ValueError(f"layer indices mismatch for {category}")
        names = [str(name) for name in train["text_position_name"].tolist()]
        if names != [str(name) for name in test["text_position_name"].tolist()]:
            raise ValueError(f"text positions mismatch for {category}")
        audio_results: dict[
            str, dict[str, dict[str, dict[str, float | int | None]]]
        ] = {}
        for stream, feature_key, layer_indices in (
            ("shared_audio", "shared_audio_segment_layer_features", shared_indices),
        ):
            audio_train, audio_test = train[feature_key], test[feature_key]
            if audio_train.shape[1:] != audio_test.shape[1:]:
                raise ValueError(f"{stream} train/test shape mismatch for {category}")
            stream_results: dict[str, dict[str, dict[str, float | int | None]]] = {}
            for segment in range(audio_train.shape[2]):
                layer_results: dict[str, dict[str, float | int | None]] = {}
                for layer_pos, layer_index in enumerate(layer_indices):
                    layer_results[str(layer_index)] = _metrics(
                        audio_train[:, layer_pos, segment],
                        train_y,
                        audio_test[:, layer_pos, segment],
                        test_y,
                        args,
                        f"{stream}:{category}:s{segment}:l{layer_index}",
                    )
                stream_results[str(segment)] = layer_results
            audio_results[stream] = stream_results
        text_results: dict[str, dict[str, dict[str, float | int | None]]] = {}
        for position, name in enumerate(names):
            layer_results = {}
            for layer_pos, layer_index in enumerate(text_indices):
                layer_results[str(layer_index)] = _metrics(
                    text_train[:, layer_pos, position],
                    train_y,
                    text_test[:, layer_pos, position],
                    test_y,
                    args,
                    f"text:{category}:{name}:l{layer_index}",
                )
            text_results[name] = layer_results
        payload: dict[str, object] = {
            "category": category,
            "input_order": args.input_order,
            "model_stem": args.model_stem,
            "n_train": int(train_y.size),
            "n_test": int(test_y.size),
            **audio_results,
            "text": text_results,
        }
        (out_dir / f"{category}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        category_payloads[category] = payload
    summary = {
        "input_order": args.input_order,
        "model_stem": args.model_stem,
        "categories": sorted(category_payloads),
        "aggregation": "mean balanced accuracy over category-specific probes",
        "shared_audio": _mean_payload(category_payloads, "shared_audio"),
        "text": _mean_payload(category_payloads, "text"),
    }
    (out_dir / "results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote Kimi rebuttal probe results to {out_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from plic.kimi_common import DEFAULT_MODEL_ID
from plic.kimi_finetune import (
    DEFAULT_KIMI_FINETUNE_DATA_ROOT,
    collect_category_examples,
    model_stem_for,
    resolve_categories,
    stratified_split_examples,
    write_json,
    write_jsonl,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build per-category fine-tuning datasets for VocalGrad.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/vocalgrad/train"))
    parser.add_argument("--out-root", type=Path, default=DEFAULT_KIMI_FINETUNE_DATA_ROOT)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-samples-per-category", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _label_counts(examples) -> dict[str, int]:
    return dict(Counter(example.gold_label for example in examples))


def main() -> None:
    args = build_parser().parse_args()
    model_stem = model_stem_for(args.model_id)
    categories = resolve_categories(args.dataset_root, args.categories)

    for category in categories:
        out_dir = args.out_root / model_stem / category
        manifest_path = out_dir / "manifest.json"
        if manifest_path.exists() and not args.overwrite:
            print(f"[build] skip category={category} manifest={manifest_path}")
            continue

        examples = collect_category_examples(
            args.dataset_root,
            category,
            max_samples=args.max_samples_per_category,
        )
        train_examples, val_examples = stratified_split_examples(
            examples,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )

        write_jsonl(out_dir / "train.jsonl", [example.to_dict() for example in train_examples])
        write_jsonl(out_dir / "val.jsonl", [example.to_dict() for example in val_examples])
        manifest = {
            "model_id": args.model_id,
            "model_stem": model_stem,
            "dataset_root": str(args.dataset_root),
            "category": category,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "max_samples_per_category": args.max_samples_per_category,
            "n_total": len(examples),
            "n_train": len(train_examples),
            "n_val": len(val_examples),
            "train_label_counts": _label_counts(train_examples),
            "val_label_counts": _label_counts(val_examples),
        }
        write_json(manifest_path, manifest)
        print(
            f"[build] category={category} total={len(examples)} train={len(train_examples)} "
            f"val={len(val_examples)} out_dir={out_dir}"
        )


if __name__ == "__main__":
    main()

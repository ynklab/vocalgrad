from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from plic.kimi_common import resolve_backend, resolve_model_id  # noqa: E402
from plic.linear_probe import (  # noqa: E402
    DEFAULT_LINEAR_PROBE_FEATURE_ROOT,
    LINEAR_PROBE_STAGES,
    collect_vocalgrad_samples,
    label_to_int,
    sample_prompt,
    slugify_model_id,
)
from plic.linear_probe_extractors import make_linear_probe_extractor  # noqa: E402
from plic.vocalgrad import DEFAULT_VOCALGRAD_DATASET_ROOT, list_vocalgrad_categories  # noqa: E402


LINEAR_PROBE_BACKEND_CHOICES = ["kimia", "audioflamingo3", "stepaudio2", "mimoaudio"]


def _summarize_existing_feature_file(path: Path, category: str) -> dict[str, object]:
    loaded = np.load(path, allow_pickle=False)
    arrays = {key: loaded[key] for key in loaded.files}
    label = arrays.get("label")
    if label is None:
        raise KeyError(f"existing feature file is missing label array: {path}")
    stage_dims: dict[str, int] = {}
    for stage in LINEAR_PROBE_STAGES:
        key = f"{stage}_features"
        value = arrays.get(key)
        if value is None:
            raise KeyError(f"existing feature file is missing {key}: {path}")
        stage_dims[stage] = int(value.shape[1])
    return {
        "category": category,
        "n_samples": int(label.shape[0]),
        "path": str(path),
        "stage_dims": stage_dims,
        "skipped_existing": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract 4-stage representations for VocalGrad and save feature caches for linear probing."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / DEFAULT_VOCALGRAD_DATASET_ROOT)
    parser.add_argument("--split-name", default=None, help="Optional split label used under outputs/features/linear_probe/.")
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--max-samples-per-category", type=int, default=None)
    parser.add_argument("--backend", required=True, choices=LINEAR_PROBE_BACKEND_CHOICES)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--adapter-path", type=Path, default=None, help="Optional LoRA adapter path for Kimi extraction.")
    parser.add_argument(
        "--model-stem-override",
        default=None,
        help="Optional output model stem override, useful for treating a fine-tuned adapter as a separate model.",
    )
    parser.add_argument("--out-root", type=Path, default=REPO_ROOT / DEFAULT_LINEAR_PROBE_FEATURE_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing per-category .npz files instead of failing when they already exist.",
    )
    return parser


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()

    model_id = resolve_model_id(args.backend, args.model_id)
    backend = resolve_backend(args.backend, model_id)
    model_stem = args.model_stem_override or slugify_model_id(model_id)
    split_name = args.split_name or args.dataset_root.name
    categories = args.categories or list_vocalgrad_categories(args.dataset_root)

    extractor = make_linear_probe_extractor(
        backend=backend,
        model_id=model_id,
        adapter_path=args.adapter_path,
    )
    out_dir = args.out_root / split_name / model_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[dict[str, object]] = []
    for category in categories:
        samples = collect_vocalgrad_samples(
            dataset_root=args.dataset_root,
            categories=[category],
            max_samples_per_category=args.max_samples_per_category,
        )
        out_path = out_dir / f"{category}.npz"
        if out_path.exists() and not args.overwrite:
            if args.skip_existing:
                manifest_entries.append(_summarize_existing_feature_file(out_path, category))
                continue
            raise FileExistsError(
                f"feature file already exists: {out_path}; pass --overwrite to replace it or --skip-existing to reuse it"
            )
        if not samples:
            raise ValueError(f"no VocalGrad samples found for category={category} under {args.dataset_root}")

        ids: list[str] = []
        labels: list[int] = []
        prompt_texts: list[str] = []
        feature_rows = {stage: [] for stage in LINEAR_PROBE_STAGES}

        for sample in tqdm(samples, desc=f"extract {category}"):
            prompt = sample_prompt(sample)
            features = extractor.extract(audio_path=sample.audio_path, prompt=prompt)
            ids.append(sample.benchmark_clip_id)
            labels.append(label_to_int(sample.gold_label))
            prompt_texts.append(prompt)
            for stage, value in features.as_dict().items():
                feature_rows[stage].append(np.asarray(value, dtype=np.float32))

        arrays = {
            "benchmark_clip_id": np.asarray(ids, dtype="U64"),
            "category": np.asarray([category] * len(ids), dtype="U64"),
            "prompt": np.asarray(prompt_texts, dtype="U512"),
            "label": np.asarray(labels, dtype=np.int64),
        }
        stage_dims: dict[str, int] = {}
        for stage, rows in feature_rows.items():
            arrays[f"{stage}_features"] = np.stack(rows).astype(np.float32, copy=False)
            stage_dims[stage] = int(arrays[f"{stage}_features"].shape[1])

        np.savez_compressed(out_path, **arrays)
        manifest_entries.append(
            {
                "category": category,
                "n_samples": len(ids),
                "path": str(out_path),
                "stage_dims": stage_dims,
            }
        )

    manifest = {
        "dataset_root": str(args.dataset_root),
        "split_name": split_name,
        "backend": backend,
        "model_id": model_id,
        "model_stem": model_stem,
        "adapter_path": str(args.adapter_path) if args.adapter_path is not None else None,
        "stages": list(LINEAR_PROBE_STAGES),
        "categories": categories,
        "files": manifest_entries,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote linear-probe features to {out_dir}")


if __name__ == "__main__":
    main()

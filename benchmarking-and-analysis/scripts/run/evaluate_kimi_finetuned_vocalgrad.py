from __future__ import annotations

import argparse
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from plic.cli_vocalgrad import default_temperature, run_category
from plic.kimi_common import DEFAULT_MODEL_ID, resolve_backend
from plic.kimi_finetune import (
    DEFAULT_KIMI_FINETUNE_ANALYSIS_ROOT,
    DEFAULT_KIMI_FINETUNE_CHECKPOINT_ROOT,
    DEFAULT_KIMI_FINETUNE_RAW_EVAL_ROOT,
    KIMI_FINETUNE_SCOPE_CHOICES,
    load_kimi_runtime,
    model_stem_for,
    resolve_categories,
    summarize_prediction_file,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned Kimi-Audio adapter across VocalGrad categories.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/vocalgrad/test"))
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_KIMI_FINETUNE_CHECKPOINT_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_KIMI_FINETUNE_RAW_EVAL_ROOT)
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_KIMI_FINETUNE_ANALYSIS_ROOT)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--run-name", default="default")
    parser.add_argument("--train-category", required=True)
    parser.add_argument("--scope", required=True, choices=KIMI_FINETUNE_SCOPE_CHOICES)
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--retry-max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _adapter_dir(args: argparse.Namespace) -> Path:
    model_stem = model_stem_for(args.model_id)
    return args.checkpoint_root / model_stem / args.run_name / args.scope / args.train_category / "adapter"


def _raw_dir(args: argparse.Namespace) -> Path:
    model_stem = model_stem_for(args.model_id)
    return args.raw_root / model_stem / args.run_name / args.scope / args.train_category


def _analysis_dir(args: argparse.Namespace) -> Path:
    model_stem = model_stem_for(args.model_id)
    return args.analysis_root / model_stem / args.run_name / args.scope / args.train_category


def _done_path(analysis_dir: Path, category: str) -> Path:
    return analysis_dir / f"done.eval.{category}.json"


def _run_args(args: argparse.Namespace, category: str) -> Namespace:
    return Namespace(
        dataset_root=args.dataset_root,
        category=category,
        max_samples=args.max_samples,
        out=None,
        model_id=args.model_id,
        backend="kimia",
        max_new_tokens=args.max_new_tokens,
        retry_max_new_tokens=args.retry_max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        swap_direction_order=False,
        explicit_audio_clip_reference=False,
        concurrency=1,
        cleanup_uploaded_files=False,
        max_retries=5,
        initial_backoff_sec=1.0,
        max_backoff_sec=30.0,
        upload_timeout_sec=180.0,
        gemini_input_usd_per_1m=None,
        gemini_output_usd_per_1m=None,
        raise_on_error=False,
    )


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()
    backend = resolve_backend("kimia", args.model_id)
    args.temperature = args.temperature if args.temperature is not None else default_temperature(backend, args.model_id)
    categories = resolve_categories(args.dataset_root, args.categories)
    adapter_dir = _adapter_dir(args)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"adapter directory not found: {adapter_dir}")

    raw_dir = _raw_dir(args)
    analysis_dir = _analysis_dir(args)
    raw_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    runtime = load_kimi_runtime(args.model_id, adapter_path=adapter_dir, trainable_adapter=False)
    try:
        for category in categories:
            raw_path = raw_dir / f"{category}.jsonl"
            summary_path = analysis_dir / f"{category}.json"
            done_path = _done_path(analysis_dir, category)
            if done_path.exists() and summary_path.exists() and not args.overwrite:
                print(f"[eval] skip train_category={args.train_category} test_category={category} done={done_path}")
                continue

            print(
                f"[eval] train_category={args.train_category} scope={args.scope} "
                f"test_category={category} adapter={adapter_dir}"
            )
            summary = run_category(
                runtime,
                args=_run_args(args, category),
                backend="kimia",
                category=category,
                out_path=raw_path,
            )
            detailed_summary = summarize_prediction_file(
                raw_path,
                summary_path,
                extra={
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "model_id": args.model_id,
                    "model_stem": model_stem_for(args.model_id),
                    "run_name": args.run_name,
                    "scope": args.scope,
                    "train_category": args.train_category,
                    "test_category": category,
                    "adapter_dir": str(adapter_dir),
                    "generation_summary": summary,
                },
            )
            write_json(
                done_path,
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "train_category": args.train_category,
                    "test_category": category,
                    "scope": args.scope,
                    "summary_path": str(summary_path),
                    "raw_path": str(raw_path),
                    "overall": detailed_summary["overall"],
                },
            )
    finally:
        runtime.close()


if __name__ == "__main__":
    main()

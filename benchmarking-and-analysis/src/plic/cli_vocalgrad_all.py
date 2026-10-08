from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from .cli_summarize_vocalgrad import summarize_vocalgrad_file
from .cli_vocalgrad import (
    BACKEND_CHOICES,
    close_runtime,
    default_temperature,
    print_gemini_cost,
    run_category,
)
from .kimi_common import resolve_backend, resolve_model_id
from .vocalgrad import DEFAULT_VOCALGRAD_DATASET_ROOT, list_vocalgrad_categories


DEFAULT_RAW_ROOT = Path("outputs/raw/vocalgrad/default")
DEFAULT_ANALYSIS_ROOT = Path("outputs/analysis/vocalgrad/default")
DEFAULT_DESCRIPTION = "Run VocalGrad evaluation across multiple categories in one command."


def build_parser(
    *,
    description: str = DEFAULT_DESCRIPTION,
    default_dataset_root: Path = DEFAULT_VOCALGRAD_DATASET_ROOT,
    default_raw_root: Path = DEFAULT_RAW_ROOT,
    default_analysis_root: Path = DEFAULT_ANALYSIS_ROOT,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=default_dataset_root,
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="Optional explicit category list. If omitted, all detected VocalGrad categories are used.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=default_raw_root,
        help="Root directory for per-category raw jsonl outputs.",
    )
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=default_analysis_root,
        help="Root directory for per-category analysis json outputs.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run identifier used under outputs/*/vocalgrad/_runs/.",
    )
    parser.add_argument(
        "--out-stem",
        default=None,
        help="Optional file stem for per-category outputs. Defaults to a model-based slug.",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip per-category analyze-vocalgrad summaries after raw generation.",
    )
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--backend", default="auto", choices=BACKEND_CHOICES)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--retry-max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--swap-direction-order",
        action="store_true",
        help="Ask about `decrease or increase` instead of `increase or decrease`.",
    )
    parser.add_argument(
        "--explicit-audio-clip-reference",
        action="store_true",
        help="Explicitly say that the target attribute is being judged for the current audio clip.",
    )
    parser.add_argument(
        "--audio-before-text",
        action="store_true",
        help=(
            "Send each evaluation clip before its question text through the "
            "interleaved multimodal input path."
        ),
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--cleanup-uploaded-files", action="store_true")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--initial-backoff-sec", type=float, default=1.0)
    parser.add_argument("--max-backoff-sec", type=float, default=30.0)
    parser.add_argument("--upload-timeout-sec", type=float, default=180.0)
    parser.add_argument("--gemini-input-usd-per-1m", type=float, default=None)
    parser.add_argument("--gemini-output-usd-per-1m", type=float, default=None)
    parser.add_argument("--raise-on-error", action="store_true")
    return parser


def _slugify_model_id(model_id: str) -> str:
    lowered = model_id.lower()
    if lowered == "moonshotai/kimi-audio-7b-instruct":
        return "kimi-audio"
    if lowered == "gemini-3-flash-preview":
        return "gemini-3-flash"
    if lowered == "nvidia/audio-flamingo-3-hf":
        return "audioflamingo3"
    if lowered == "xiaomimimo/mimo-audio-7b-instruct":
        return "mimo-audio"

    stem = model_id.split("/")[-1].lower()
    stem = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
    return stem or "model"


def _resolve_categories(dataset_root: Path, categories: list[str] | None) -> list[str]:
    discovered = list_vocalgrad_categories(dataset_root)
    if categories is None or len(categories) == 0:
        return discovered

    missing = [category for category in categories if category not in discovered]
    if missing:
        raise ValueError(
            "unknown VocalGrad categories requested: "
            f"{missing}; available={discovered}"
        )
    return categories


def _run_manifest_path(root: Path, run_id: str) -> Path:
    return root / "_runs" / run_id / "manifest.json"


def _summary_totals(summaries: list[dict[str, object]]) -> dict[str, float | int]:
    total = sum(int(summary["total"]) for summary in summaries)
    parsed = sum(int(summary["parsed"]) for summary in summaries)
    correct = sum(int(summary["correct"]) for summary in summaries)
    parsed_accuracy = (correct / parsed) if parsed else 0.0
    strict_accuracy = (correct / total) if total else 0.0
    return {
        "total": total,
        "parsed": parsed,
        "correct": correct,
        "parsed_accuracy": parsed_accuracy,
        "strict_accuracy": strict_accuracy,
    }


def _analysis_summary_record(category: str, out_path: Path, summary: dict[str, object]) -> dict[str, object]:
    overall = summary["overall"]
    return {
        "category": category,
        "out_path": str(out_path),
        "accuracy": overall["accuracy"],
        "precision_increase": overall["precision_increase"],
        "recall_increase": overall["recall_increase"],
        "f1_increase": overall["f1_increase"],
        "total_evaluable": overall["total_evaluable"],
    }


def run_from_args(args: argparse.Namespace) -> None:
    args.model_id = resolve_model_id(args.backend, args.model_id)
    backend = resolve_backend(args.backend, args.model_id)
    args.temperature = (
        args.temperature
        if args.temperature is not None
        else default_temperature(backend, args.model_id)
    )

    categories = _resolve_categories(args.dataset_root, args.categories)
    model_stem = args.out_stem or _slugify_model_id(args.model_id)
    run_id = args.run_id or (
        f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{model_stem}"
    )

    from .kimi_common import AudioModelRuntime

    runtime = AudioModelRuntime(
        backend=backend,
        model_id=args.model_id,
        cleanup_uploaded_files=args.cleanup_uploaded_files,
        upload_timeout_sec=args.upload_timeout_sec,
        max_retries=args.max_retries,
        initial_backoff_sec=args.initial_backoff_sec,
        max_backoff_sec=args.max_backoff_sec,
    )

    generation_summaries: list[dict[str, object]] = []
    analysis_summaries: list[dict[str, object]] = []
    try:
        print(
            f"running vocalgrad-all backend={backend} model={args.model_id} "
            f"categories={len(categories)}"
        )
        for idx, category in enumerate(categories, start=1):
            print(f"[{idx}/{len(categories)}] category={category}")
            raw_out_path = args.raw_root / category / f"{model_stem}.jsonl"
            generation_summary = run_category(
                runtime,
                args=args,
                backend=backend,
                category=category,
                out_path=raw_out_path,
            )
            generation_summaries.append(generation_summary)

            if not args.skip_analysis:
                analysis_out_path = args.analysis_root / category / f"{model_stem}.json"
                summary = summarize_vocalgrad_file(raw_out_path, analysis_out_path)
                analysis_record = _analysis_summary_record(category, analysis_out_path, summary)
                analysis_summaries.append(analysis_record)
                print(
                    f"analysis done: category={category} "
                    f"accuracy={analysis_record['accuracy']:.4f} out={analysis_out_path}"
                )
    finally:
        if backend == "gemini":
            print_gemini_cost(runtime, args)
        close_runtime(runtime)

    totals = _summary_totals(generation_summaries)
    raw_manifest_path = _run_manifest_path(args.raw_root, run_id)
    raw_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    raw_manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "backend": backend,
        "model_id": args.model_id,
        "model_stem": model_stem,
        "dataset_root": str(args.dataset_root),
        "raw_root": str(args.raw_root),
        "analysis_root": str(args.analysis_root),
        "categories": categories,
        "max_samples": args.max_samples,
        "max_new_tokens": args.max_new_tokens,
        "retry_max_new_tokens": args.retry_max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "swap_direction_order": args.swap_direction_order,
        "explicit_audio_clip_reference": args.explicit_audio_clip_reference,
        "audio_before_text": args.audio_before_text,
        "concurrency": args.concurrency,
        "skip_analysis": args.skip_analysis,
        "generation_summaries": generation_summaries,
        "analysis_summaries": analysis_summaries,
        "totals": totals,
    }
    raw_manifest_path.write_text(json.dumps(raw_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.skip_analysis:
        analysis_manifest_path = _run_manifest_path(args.analysis_root, run_id)
        analysis_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        analysis_manifest = {
            "timestamp": raw_manifest["timestamp"],
            "run_id": run_id,
            "backend": backend,
            "model_id": args.model_id,
            "model_stem": model_stem,
            "analysis_root": str(args.analysis_root),
            "analysis_summaries": analysis_summaries,
            "totals": totals,
        }
        analysis_manifest_path.write_text(
            json.dumps(analysis_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"analysis manifest: {analysis_manifest_path}")

    print(
        "all categories done: "
        f"total={totals['total']} parsed={totals['parsed']} correct={totals['correct']} "
        f"parsed_accuracy={totals['parsed_accuracy']:.4f} "
        f"strict_accuracy={totals['strict_accuracy']:.4f}"
    )
    print(f"run manifest: {raw_manifest_path}")


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()
    run_from_args(args)


if __name__ == "__main__":
    main()

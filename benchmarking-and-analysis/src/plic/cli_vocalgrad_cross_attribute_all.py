from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from .cli_vocalgrad import BACKEND_CHOICES, close_runtime, default_temperature, print_gemini_cost
from .cli_vocalgrad_cross_attribute import (
    DEFAULT_ANALYSIS_ROOT,
    DEFAULT_RAW_ROOT,
    summarize_cross_attribute_file,
    run_attribute_pair,
)
from .kimi_common import AudioModelRuntime, resolve_backend, resolve_model_id
from .vocalgrad import DEFAULT_VOCALGRAD_DATASET_ROOT, list_vocalgrad_categories


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run cross-attribute VocalGrad evaluation across multiple actual/prompt attribute pairs."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_VOCALGRAD_DATASET_ROOT)
    parser.add_argument(
        "--actual-attributes",
        nargs="*",
        default=None,
        help="Optional explicit list of actual attributes. If omitted, all VocalGrad attributes are used.",
    )
    parser.add_argument(
        "--prompt-attributes",
        nargs="*",
        default=None,
        help="Optional explicit list of prompt attributes. If omitted, all VocalGrad attributes are used.",
    )
    parser.add_argument(
        "--off-diagonal-only",
        action="store_true",
        help="Skip pairs where actual attribute and prompt attribute are the same.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out-stem", default=None)
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--backend", default="auto", choices=BACKEND_CHOICES)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--retry-max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--swap-direction-order", action="store_true")
    parser.add_argument("--explicit-audio-clip-reference", action="store_true")
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
    if lowered == "stepfun-ai/step-audio-2-mini":
        return "step-audio-2-mini"
    stem = model_id.split("/")[-1].lower()
    stem = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
    return stem or "model"


def _resolve_attributes(dataset_root: Path, attrs: list[str] | None, *, flag_name: str) -> list[str]:
    discovered = list_vocalgrad_categories(dataset_root)
    if attrs is None or len(attrs) == 0:
        return discovered
    missing = [attr for attr in attrs if attr not in discovered]
    if missing:
        raise ValueError(f"unknown {flag_name}: {missing}; available={discovered}")
    return attrs


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


def _analysis_summary_record(actual_attribute: str, prompt_attribute: str, out_path: Path, summary: dict[str, object]) -> dict[str, object]:
    overall = summary["overall"]
    return {
        "actual_attribute": actual_attribute,
        "prompt_attribute": prompt_attribute,
        "out_path": str(out_path),
        "accuracy": overall["accuracy"],
        "precision_increase": overall["precision_increase"],
        "recall_increase": overall["recall_increase"],
        "f1_increase": overall["f1_increase"],
        "total_evaluable": overall["total_evaluable"],
    }


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()

    args.model_id = resolve_model_id(args.backend, args.model_id)
    backend = resolve_backend(args.backend, args.model_id)
    args.temperature = args.temperature if args.temperature is not None else default_temperature(backend, args.model_id)

    actual_attributes = _resolve_attributes(args.dataset_root, args.actual_attributes, flag_name="actual_attributes")
    prompt_attributes = _resolve_attributes(args.dataset_root, args.prompt_attributes, flag_name="prompt_attributes")
    pairs = [
        (actual_attribute, prompt_attribute)
        for actual_attribute in actual_attributes
        for prompt_attribute in prompt_attributes
        if not (args.off_diagonal_only and actual_attribute == prompt_attribute)
    ]
    if not pairs:
        raise ValueError("no cross-attribute pairs selected")

    model_stem = args.out_stem or _slugify_model_id(args.model_id)
    run_id = args.run_id or f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{model_stem}"

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
            f"running vocalgrad-cross-attribute-all backend={backend} model={args.model_id} "
            f"pairs={len(pairs)}"
        )
        for idx, (actual_attribute, prompt_attribute) in enumerate(pairs, start=1):
            print(f"[{idx}/{len(pairs)}] actual={actual_attribute} prompt={prompt_attribute}")
            raw_out_path = args.raw_root / actual_attribute / prompt_attribute / f"{model_stem}.jsonl"
            generation_summary = run_attribute_pair(
                runtime,
                args=args,
                backend=backend,
                actual_attribute=actual_attribute,
                prompt_attribute=prompt_attribute,
                out_path=raw_out_path,
            )
            generation_summaries.append(generation_summary)
            if not args.skip_analysis:
                analysis_out_path = args.analysis_root / actual_attribute / prompt_attribute / f"{model_stem}.json"
                summary = summarize_cross_attribute_file(
                    raw_out_path,
                    analysis_out_path,
                    actual_attribute=actual_attribute,
                    prompt_attribute=prompt_attribute,
                    args=args,
                    backend=backend,
                )
                analysis_record = _analysis_summary_record(actual_attribute, prompt_attribute, analysis_out_path, summary)
                analysis_summaries.append(analysis_record)
                print(
                    f"analysis done: actual={actual_attribute} prompt={prompt_attribute} "
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
        "actual_attributes": actual_attributes,
        "prompt_attributes": prompt_attributes,
        "pairs": [{"actual_attribute": actual_attribute, "prompt_attribute": prompt_attribute} for actual_attribute, prompt_attribute in pairs],
        "off_diagonal_only": args.off_diagonal_only,
        "max_samples": args.max_samples,
        "max_new_tokens": args.max_new_tokens,
        "retry_max_new_tokens": args.retry_max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "swap_direction_order": args.swap_direction_order,
        "explicit_audio_clip_reference": args.explicit_audio_clip_reference,
        "concurrency": args.concurrency,
        "skip_analysis": args.skip_analysis,
        "generation_summaries": generation_summaries,
        "analysis_summaries": analysis_summaries,
        "totals": totals,
    }
    raw_manifest_path.write_text(json.dumps(raw_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

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
            "actual_attributes": actual_attributes,
            "prompt_attributes": prompt_attributes,
            "pairs": raw_manifest["pairs"],
            "analysis_summaries": analysis_summaries,
            "totals": totals,
        }
        analysis_manifest_path.write_text(json.dumps(analysis_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"analysis manifest: {analysis_manifest_path}")

    print(
        "all pairs done: "
        f"total={totals['total']} parsed={totals['parsed']} correct={totals['correct']} "
        f"parsed_accuracy={totals['parsed_accuracy']:.4f} "
        f"strict_accuracy={totals['strict_accuracy']:.4f}"
    )
    print(f"run manifest: {raw_manifest_path}")


if __name__ == "__main__":
    main()

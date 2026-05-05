from __future__ import annotations

import argparse
import asyncio
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from tqdm.auto import tqdm

from .cli_summarize_vocalgrad import summarize_vocalgrad_file
from .cli_vocalgrad import (
    BACKEND_CHOICES,
    build_prompt,
    close_runtime,
    default_temperature,
    maybe_retry_same_prompt,
    maybe_retry_same_prompt_async,
    print_gemini_cost,
    prompt_settings_dict,
)
from .kimi_common import AudioModelRuntime, resolve_backend, resolve_model_id
from .vocalgrad import (
    DEFAULT_VOCALGRAD_DATASET_ROOT,
    VocalGradSample,
    list_vocalgrad_categories,
    load_local_vocalgrad_samples,
)


DEFAULT_RAW_ROOT = Path("outputs/raw/vocalgrad_cross_attribute/default")
DEFAULT_ANALYSIS_ROOT = Path("outputs/analysis/vocalgrad_cross_attribute/default")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run cross-attribute VocalGrad evaluation. "
            "Audio clips are drawn from one actual attribute, while the prompt asks about another attribute."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_VOCALGRAD_DATASET_ROOT)
    parser.add_argument(
        "--actual-attribute",
        default=None,
        help="The attribute that actually changes in the evaluated audio clips, e.g. `echo`. If omitted, all VocalGrad attributes are used.",
    )
    parser.add_argument(
        "--prompt-attribute",
        default=None,
        help="The attribute named in the prompt, e.g. `volume`. If omitted, all VocalGrad attributes are used.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--analysis-out", type=Path, default=None)
    parser.add_argument(
        "--off-diagonal-only",
        action="store_true",
        help="When evaluating multiple pairs, skip pairs where actual attribute and prompt attribute are the same.",
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


def default_output_path(actual_attribute: str, prompt_attribute: str) -> Path:
    return DEFAULT_RAW_ROOT / actual_attribute / prompt_attribute / "results.jsonl"


def default_analysis_path(actual_attribute: str, prompt_attribute: str) -> Path:
    return DEFAULT_ANALYSIS_ROOT / actual_attribute / prompt_attribute / "summary.json"


def build_cross_attribute_prompt(prompt_attribute: str, args: argparse.Namespace) -> str:
    return build_prompt(
        prompt_attribute,
        swap_direction_order=args.swap_direction_order,
        explicit_audio_clip_reference=args.explicit_audio_clip_reference,
    )


def _resolve_attributes(dataset_root: Path, attributes: list[str] | None, *, flag_name: str) -> list[str]:
    available = list_vocalgrad_categories(dataset_root)
    if attributes is None or len(attributes) == 0:
        return available
    missing = [attribute for attribute in attributes if attribute not in available]
    if missing:
        raise ValueError(f"unknown {flag_name}: {missing}; available={available}")
    return attributes


def _summary_metadata(actual_attribute: str, prompt_attribute: str, args: argparse.Namespace, backend: str) -> dict[str, object]:
    return {
        "evaluation_type": "vocalgrad_cross_attribute",
        "actual_attribute": actual_attribute,
        "prompt_attribute": prompt_attribute,
        "backend": backend,
        "model_id": args.model_id,
        "prompt": build_cross_attribute_prompt(prompt_attribute, args),
        "prompt_settings": prompt_settings_dict(args),
        "dataset_root": str(args.dataset_root),
    }


def summarize_cross_attribute_file(
    input_path: Path,
    out_path: Path,
    *,
    actual_attribute: str,
    prompt_attribute: str,
    args: argparse.Namespace,
    backend: str,
) -> dict[str, object]:
    summary = summarize_vocalgrad_file(input_path, out_path)
    summary.update(_summary_metadata(actual_attribute, prompt_attribute, args, backend))
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


async def _run_async_batch_cross(
    runtime: AudioModelRuntime,
    samples: list[VocalGradSample],
    args: argparse.Namespace,
    backend: str,
    actual_attribute: str,
    prompt_attribute: str,
) -> tuple[list[dict[str, object]], int, int]:
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    prompt = build_cross_attribute_prompt(prompt_attribute, args)
    settings = prompt_settings_dict(args)
    rows: list[dict[str, object]] = []
    parsed = 0
    correct = 0

    async def worker(sample: VocalGradSample) -> dict[str, object]:
        async with semaphore:
            try:
                raw_text = await runtime.generate_text_async(
                    prompt=prompt,
                    audio_path=sample.audio_path,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                )
                pred_label, final_raw_text = await maybe_retry_same_prompt_async(
                    runtime=runtime,
                    sample=sample,
                    raw_text=raw_text,
                    retry_max_new_tokens=args.retry_max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    prompt=prompt,
                )
                return {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "backend": backend,
                    "model_id": args.model_id,
                    "category": actual_attribute,
                    "actual_attribute": actual_attribute,
                    "prompt_attribute": prompt_attribute,
                    "benchmark_clip_id": sample.benchmark_clip_id,
                    "audio_path": str(sample.audio_path),
                    "gold_label": sample.gold_label,
                    "prediction_label": pred_label,
                    "parsed_ok": pred_label is not None,
                    "is_correct": pred_label == sample.gold_label if pred_label is not None else None,
                    "raw_response": final_raw_text,
                    "prompt": prompt,
                    "prompt_settings": settings,
                    "metadata": sample.metadata,
                }
            except Exception as exc:
                if args.raise_on_error:
                    raise
                return {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "backend": backend,
                    "model_id": args.model_id,
                    "category": actual_attribute,
                    "actual_attribute": actual_attribute,
                    "prompt_attribute": prompt_attribute,
                    "benchmark_clip_id": sample.benchmark_clip_id,
                    "audio_path": str(sample.audio_path),
                    "gold_label": sample.gold_label,
                    "prediction_label": None,
                    "parsed_ok": False,
                    "is_correct": None,
                    "raw_response": str(exc),
                    "error_type": type(exc).__name__,
                    "traceback": traceback.format_exc(),
                    "prompt": prompt,
                    "prompt_settings": settings,
                    "metadata": sample.metadata,
                    "error": True,
                }

    tasks = [asyncio.create_task(worker(sample)) for sample in samples]
    with tqdm(total=len(tasks), desc=f"xattr:{actual_attribute}->{prompt_attribute}", unit="sample") as pbar:
        for task in asyncio.as_completed(tasks):
            row = await task
            rows.append(row)
            if row.get("prediction_label") is not None:
                parsed += 1
            if row.get("is_correct") is True:
                correct += 1
            pbar.update(1)

    return rows, parsed, correct


def run_attribute_pair(
    runtime: AudioModelRuntime,
    *,
    args: argparse.Namespace,
    backend: str,
    actual_attribute: str,
    prompt_attribute: str,
    out_path: Path,
) -> dict[str, object]:
    samples = load_local_vocalgrad_samples(args.dataset_root, actual_attribute)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    parsed = 0
    correct = 0
    prompt = build_cross_attribute_prompt(prompt_attribute, args)
    settings = prompt_settings_dict(args)

    print(
        "evaluating vocalgrad-cross-attribute "
        f"actual={actual_attribute} prompt={prompt_attribute} "
        f"samples={len(samples)} backend={backend}"
    )

    if backend == "gemini":
        rows, parsed, correct = asyncio.run(
            _run_async_batch_cross(
                runtime=runtime,
                samples=samples,
                args=args,
                backend=backend,
                actual_attribute=actual_attribute,
                prompt_attribute=prompt_attribute,
            )
        )
        total = len(rows)
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        with out_path.open("w", encoding="utf-8") as f, tqdm(
            samples,
            desc=f"xattr:{actual_attribute}->{prompt_attribute}",
            unit="sample",
        ) as pbar:
            for sample in pbar:
                total += 1
                try:
                    raw_text = runtime.generate_text(
                        prompt=prompt,
                        audio_path=sample.audio_path,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_k=args.top_k,
                    )
                    pred_label, final_raw_text = maybe_retry_same_prompt(
                        runtime=runtime,
                        sample=sample,
                        raw_text=raw_text,
                        retry_max_new_tokens=args.retry_max_new_tokens,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        prompt=prompt,
                    )
                    if pred_label is not None:
                        parsed += 1
                    is_correct = pred_label == sample.gold_label if pred_label is not None else None
                    if is_correct:
                        correct += 1
                    row = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "backend": backend,
                        "model_id": args.model_id,
                        "category": actual_attribute,
                        "actual_attribute": actual_attribute,
                        "prompt_attribute": prompt_attribute,
                        "benchmark_clip_id": sample.benchmark_clip_id,
                        "audio_path": str(sample.audio_path),
                        "gold_label": sample.gold_label,
                        "prediction_label": pred_label,
                        "parsed_ok": pred_label is not None,
                        "is_correct": is_correct,
                        "raw_response": final_raw_text,
                        "prompt": prompt,
                        "prompt_settings": settings,
                        "metadata": sample.metadata,
                    }
                except Exception as exc:
                    row = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "backend": backend,
                        "model_id": args.model_id,
                        "category": actual_attribute,
                        "actual_attribute": actual_attribute,
                        "prompt_attribute": prompt_attribute,
                        "benchmark_clip_id": sample.benchmark_clip_id,
                        "audio_path": str(sample.audio_path),
                        "gold_label": sample.gold_label,
                        "prediction_label": None,
                        "parsed_ok": False,
                        "is_correct": None,
                        "raw_response": str(exc),
                        "error_type": type(exc).__name__,
                        "traceback": traceback.format_exc(),
                        "prompt": prompt,
                        "prompt_settings": settings,
                        "metadata": sample.metadata,
                        "error": True,
                    }
                    if args.raise_on_error:
                        raise
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    parsed_acc = (correct / parsed) if parsed else 0.0
    strict_acc = (correct / total) if total else 0.0
    print(
        f"done: total={total} parsed={parsed} correct={correct} "
        f"parsed_accuracy={parsed_acc:.4f} strict_accuracy={strict_acc:.4f}"
    )
    print(f"raw output: {out_path}")

    return {
        "actual_attribute": actual_attribute,
        "prompt_attribute": prompt_attribute,
        "total": total,
        "parsed": parsed,
        "correct": correct,
        "parsed_accuracy": parsed_acc,
        "strict_accuracy": strict_acc,
        "out_path": str(out_path),
        "prompt_settings": settings,
    }


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()

    actual_attributes = _resolve_attributes(
        args.dataset_root,
        [args.actual_attribute] if args.actual_attribute is not None else None,
        flag_name="actual_attribute",
    )
    prompt_attributes = _resolve_attributes(
        args.dataset_root,
        [args.prompt_attribute] if args.prompt_attribute is not None else None,
        flag_name="prompt_attribute",
    )
    pairs = [
        (actual_attribute, prompt_attribute)
        for actual_attribute in actual_attributes
        for prompt_attribute in prompt_attributes
        if not (args.off_diagonal_only and actual_attribute == prompt_attribute)
    ]
    if not pairs:
        raise ValueError("no cross-attribute pairs selected")

    if len(pairs) > 1 and (args.out is not None or args.analysis_out is not None):
        raise ValueError(
            "--out and --analysis-out only support a single actual/prompt pair; omit them when expanding to multiple attributes"
        )

    args.model_id = resolve_model_id(args.backend, args.model_id)
    backend = resolve_backend(args.backend, args.model_id)
    args.temperature = args.temperature if args.temperature is not None else default_temperature(backend, args.model_id)

    runtime = AudioModelRuntime(
        backend=backend,
        model_id=args.model_id,
        cleanup_uploaded_files=args.cleanup_uploaded_files,
        upload_timeout_sec=args.upload_timeout_sec,
        max_retries=args.max_retries,
        initial_backoff_sec=args.initial_backoff_sec,
        max_backoff_sec=args.max_backoff_sec,
    )

    try:
        for actual_attribute, prompt_attribute in pairs:
            out_path = args.out or default_output_path(actual_attribute, prompt_attribute)
            analysis_out = args.analysis_out or default_analysis_path(actual_attribute, prompt_attribute)
            run_attribute_pair(
                runtime,
                args=args,
                backend=backend,
                actual_attribute=actual_attribute,
                prompt_attribute=prompt_attribute,
                out_path=out_path,
            )
            summary = summarize_cross_attribute_file(
                out_path,
                analysis_out,
                actual_attribute=actual_attribute,
                prompt_attribute=prompt_attribute,
                args=args,
                backend=backend,
            )
            print(f"analysis output: {analysis_out}")
            print(f"analysis accuracy: {summary['overall']['accuracy']:.4f}")
        if backend == "gemini":
            print_gemini_cost(runtime, args)
    finally:
        close_runtime(runtime)


if __name__ == "__main__":
    main()

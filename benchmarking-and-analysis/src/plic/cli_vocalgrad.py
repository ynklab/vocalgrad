from __future__ import annotations

import argparse
import asyncio
import json
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from tqdm.auto import tqdm

from .kimi_common import (
    AudioModelRuntime,
    default_gemini_pricing,
    resolve_backend,
    resolve_model_id,
)
from .vocalgrad import DEFAULT_VOCALGRAD_DATASET_ROOT, VocalGradSample, load_local_vocalgrad_samples


BACKEND_CHOICES = ["auto", "kimia", "gemini", "audioflamingo3", "stepaudio2", "mimoaudio"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run single-audio direction classification on VocalGrad. "
            "Each sample asks whether the target attribute increases or decreases over time."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_VOCALGRAD_DATASET_ROOT,
    )
    parser.add_argument(
        "--category",
        default="volume",
        help="VocalGrad category name such as `volume`, `voice_pitch`, or `background_noise`.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Evaluate first N manifest rows only.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
    )
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--backend", default="auto", choices=BACKEND_CHOICES)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--retry-max-new-tokens",
        type=int,
        default=256,
        help="If the first response is unparsable, retry once with the same prompt and this max token budget.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature. If omitted, backend/model-specific defaults are used.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--swap-direction-order",
        action="store_true",
        help="Ask about `decrease or increase` instead of `increase or decrease`. The answer instruction order is swapped to match.",
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
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Number of concurrent requests for Gemini backend.",
    )
    parser.add_argument("--cleanup-uploaded-files", action="store_true")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--initial-backoff-sec", type=float, default=1.0)
    parser.add_argument("--max-backoff-sec", type=float, default=30.0)
    parser.add_argument("--upload-timeout-sec", type=float, default=180.0)
    parser.add_argument("--gemini-input-usd-per-1m", type=float, default=None)
    parser.add_argument("--gemini-output-usd-per-1m", type=float, default=None)
    parser.add_argument(
        "--raise-on-error",
        action="store_true",
        help="Raise immediately on first sample error.",
    )
    return parser


def default_output_path(category: str) -> Path:
    return Path(f"outputs/raw/vocalgrad/default/{category}/results.jsonl")


def build_prompt(
    category: str,
    *,
    swap_direction_order: bool = False,
    explicit_audio_clip_reference: bool = False,
    audio_before_text: bool = False,
) -> str:
    # The flag controls the multimodal part order, not the question wording.
    # Keeping it here lets saved prompt settings be replayed with build_prompt.
    del audio_before_text
    prompt_attribute = {
        "volume": "volume",
        "voice_pitch": "voice pitch",
    }.get(category, category.replace("_", " "))

    if explicit_audio_clip_reference:
        subject = f"the {prompt_attribute} of this audio clip"
    else:
        subject = f"the {prompt_attribute}"

    if swap_direction_order:
        question = f"Does {subject} decrease or increase over time?"
        answer_instruction = 'Answer with only one word: "decrease" or "increase".'
    else:
        question = f"Does {subject} increase or decrease over time?"
        answer_instruction = 'Answer with only one word: "increase" or "decrease".'

    return f"{question}\n\n{answer_instruction}"


def prompt_settings_dict(args: argparse.Namespace) -> dict[str, bool]:
    return {
        "swap_direction_order": args.swap_direction_order,
        "explicit_audio_clip_reference": args.explicit_audio_clip_reference,
        "audio_before_text": getattr(args, "audio_before_text", False),
    }


def build_sample_prompt(category: str, args: argparse.Namespace) -> str:
    return build_prompt(category, **prompt_settings_dict(args))


def prompt_for_sample(sample: VocalGradSample, args: argparse.Namespace) -> str:
    prompt_builder = getattr(args, "sample_prompt_builder", None)
    if callable(prompt_builder):
        return str(prompt_builder(sample, args))
    return build_sample_prompt(sample.category, args)


def audio_path_for_sample(sample: VocalGradSample, args: argparse.Namespace) -> Path:
    audio_path_builder = getattr(args, "sample_audio_path_builder", None)
    if callable(audio_path_builder):
        return Path(audio_path_builder(sample, args))
    return sample.audio_path


def extra_fields_for_sample(sample: VocalGradSample, args: argparse.Namespace) -> dict[str, object]:
    extra_fields_builder = getattr(args, "sample_extra_fields_builder", None)
    if callable(extra_fields_builder):
        return dict(extra_fields_builder(sample, args))
    return {}


def interleaved_parts_for_sample(sample: VocalGradSample, args: argparse.Namespace) -> list[dict[str, object]] | None:
    parts_builder = getattr(args, "sample_interleaved_parts_builder", None)
    if callable(parts_builder):
        return list(parts_builder(sample, args))
    if getattr(args, "audio_before_text", False):
        return [
            {"type": "audio", "path": str(sample.audio_path)},
            {"type": "text", "text": prompt_for_sample(sample, args)},
        ]
    return None


def default_temperature(backend: str, model_id: str) -> float:
    if backend == "gemini" and "gemini-3" in model_id.lower():
        return 1.0
    return 0.0


def parse_direction_label(raw_text: str) -> str | None:
    lowered = raw_text.strip().lower()
    if lowered in {"increase", "decrease"}:
        return lowered
    if lowered in {"increases", "decreases"}:
        return "increase" if lowered == "increases" else "decrease"

    tokens = re.findall(r"\b(increase|decrease|increases|decreases)\b", lowered)
    labels = {
        "increase" if token in {"increase", "increases"} else "decrease"
        for token in tokens
    }
    if len(labels) != 1:
        return None
    return next(iter(labels))


def maybe_retry_same_prompt(
    runtime: AudioModelRuntime,
    sample: VocalGradSample,
    raw_text: str,
    retry_max_new_tokens: int,
    temperature: float,
    top_k: int,
    prompt: str,
    audio_path: Path | None = None,
    interleaved_parts: list[dict[str, object]] | None = None,
) -> tuple[str | None, str]:
    pred = parse_direction_label(raw_text)
    if pred is not None:
        return pred, raw_text

    if interleaved_parts is not None:
        retry_raw_text = runtime.generate_text_interleaved(
            parts=interleaved_parts,
            max_new_tokens=retry_max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
    else:
        retry_raw_text = runtime.generate_text(
            prompt=prompt,
            audio_path=audio_path or sample.audio_path,
            max_new_tokens=retry_max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
    return parse_direction_label(retry_raw_text), retry_raw_text


async def maybe_retry_same_prompt_async(
    runtime: AudioModelRuntime,
    sample: VocalGradSample,
    raw_text: str,
    retry_max_new_tokens: int,
    temperature: float,
    top_k: int,
    prompt: str,
    audio_path: Path | None = None,
    interleaved_parts: list[dict[str, object]] | None = None,
) -> tuple[str | None, str]:
    pred = parse_direction_label(raw_text)
    if pred is not None:
        return pred, raw_text

    if interleaved_parts is not None:
        retry_raw_text = await runtime.generate_text_interleaved_async(
            parts=interleaved_parts,
            max_new_tokens=retry_max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
    else:
        retry_raw_text = await runtime.generate_text_async(
            prompt=prompt,
            audio_path=audio_path or sample.audio_path,
            max_new_tokens=retry_max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
    return parse_direction_label(retry_raw_text), retry_raw_text


def run_one(
    runtime: AudioModelRuntime,
    sample: VocalGradSample,
    max_new_tokens: int,
    retry_max_new_tokens: int,
    temperature: float,
    top_k: int,
    prompt: str,
    audio_path: Path | None = None,
    interleaved_parts: list[dict[str, object]] | None = None,
) -> tuple[str | None, str]:
    resolved_audio_path = audio_path or sample.audio_path
    if interleaved_parts is not None:
        raw_text = runtime.generate_text_interleaved(
            parts=interleaved_parts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
    else:
        raw_text = runtime.generate_text(
            prompt=prompt,
            audio_path=resolved_audio_path,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
    return maybe_retry_same_prompt(
        runtime=runtime,
        sample=sample,
        raw_text=raw_text,
        retry_max_new_tokens=retry_max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        prompt=prompt,
        audio_path=resolved_audio_path,
        interleaved_parts=interleaved_parts,
    )


async def run_one_async(
    runtime: AudioModelRuntime,
    sample: VocalGradSample,
    max_new_tokens: int,
    retry_max_new_tokens: int,
    temperature: float,
    top_k: int,
    prompt: str,
    audio_path: Path | None = None,
    interleaved_parts: list[dict[str, object]] | None = None,
) -> tuple[str | None, str]:
    resolved_audio_path = audio_path or sample.audio_path
    if interleaved_parts is not None:
        raw_text = await runtime.generate_text_interleaved_async(
            parts=interleaved_parts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
    else:
        raw_text = await runtime.generate_text_async(
            prompt=prompt,
            audio_path=resolved_audio_path,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
    return await maybe_retry_same_prompt_async(
        runtime=runtime,
        sample=sample,
        raw_text=raw_text,
        retry_max_new_tokens=retry_max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        prompt=prompt,
        audio_path=resolved_audio_path,
        interleaved_parts=interleaved_parts,
    )


def print_gemini_cost(runtime: AudioModelRuntime, args: argparse.Namespace) -> None:
    default_input_price, default_output_price = default_gemini_pricing(args.model_id)
    input_price = (
        args.gemini_input_usd_per_1m
        if args.gemini_input_usd_per_1m is not None
        else default_input_price
    )
    output_price = (
        args.gemini_output_usd_per_1m
        if args.gemini_output_usd_per_1m is not None
        else default_output_price
    )
    estimated_cost = runtime.stats.estimate_cost_usd(input_price, output_price)
    print(
        "gemini usage: "
        f"requests={runtime.stats.request_count} retries={runtime.stats.retry_count} "
        f"input_tokens={runtime.stats.input_tokens} output_tokens={runtime.stats.output_tokens}"
    )
    print(
        "gemini estimated cost (USD): "
        f"{estimated_cost:.6f} "
        f"(input_rate={input_price}/1M, output_rate={output_price}/1M)"
    )


async def _run_async_batch(
    runtime: AudioModelRuntime,
    samples: list[VocalGradSample],
    args: argparse.Namespace,
    backend: str,
    category: str,
) -> tuple[list[dict[str, object]], int, int]:
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    rows: list[dict[str, object]] = []
    parsed = 0
    correct = 0

    async def worker(sample: VocalGradSample) -> dict[str, object]:
        prompt = prompt_for_sample(sample, args)
        audio_path = audio_path_for_sample(sample, args)
        extra_fields = extra_fields_for_sample(sample, args)
        interleaved_parts = interleaved_parts_for_sample(sample, args)
        settings = prompt_settings_dict(args)
        async with semaphore:
            try:
                pred_label, raw_text = await run_one_async(
                    runtime=runtime,
                    sample=sample,
                    max_new_tokens=args.max_new_tokens,
                    retry_max_new_tokens=args.retry_max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    prompt=prompt,
                    audio_path=audio_path,
                    interleaved_parts=interleaved_parts,
                )
                return {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "backend": backend,
                    "model_id": args.model_id,
                    "category": sample.category,
                    "benchmark_clip_id": sample.benchmark_clip_id,
                    "audio_path": str(audio_path),
                    "original_audio_path": str(sample.audio_path),
                    "gold_label": sample.gold_label,
                    "prediction_label": pred_label,
                    "parsed_ok": pred_label is not None,
                    "is_correct": pred_label == sample.gold_label if pred_label is not None else None,
                    "raw_response": raw_text,
                    "prompt": prompt,
                    "prompt_settings": settings,
                    "metadata": sample.metadata,
                    **extra_fields,
                }
            except Exception as exc:
                if args.raise_on_error:
                    raise
                return {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "backend": backend,
                    "model_id": args.model_id,
                    "category": sample.category,
                    "benchmark_clip_id": sample.benchmark_clip_id,
                    "audio_path": str(audio_path),
                    "original_audio_path": str(sample.audio_path),
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
                    **extra_fields,
                }

    tasks = [asyncio.create_task(worker(sample)) for sample in samples]
    with tqdm(total=len(tasks), desc=f"vocalgrad:{category}", unit="sample") as pbar:
        for task in asyncio.as_completed(tasks):
            row = await task
            rows.append(row)
            if row.get("prediction_label") is not None:
                parsed += 1
            if row.get("is_correct") is True:
                correct += 1
            pbar.update(1)

    return rows, parsed, correct


def run_category(
    runtime: AudioModelRuntime,
    *,
    args: argparse.Namespace,
    backend: str,
    category: str,
    out_path: Path,
) -> dict[str, object]:
    samples = load_local_vocalgrad_samples(args.dataset_root, category)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    parsed = 0
    correct = 0

    print(
        f"evaluating vocalgrad category={category} "
        f"samples={len(samples)} backend={backend}"
    )

    if backend == "gemini":
        rows, parsed, correct = asyncio.run(
            _run_async_batch(
                runtime=runtime,
                samples=samples,
                args=args,
                backend=backend,
                category=category,
            )
        )
        total = len(rows)
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        with out_path.open("w", encoding="utf-8") as f, tqdm(
            samples,
            desc=f"vocalgrad:{category}",
            unit="sample",
        ) as pbar:
            for sample in pbar:
                total += 1
                prompt = prompt_for_sample(sample, args)
                audio_path = audio_path_for_sample(sample, args)
                extra_fields = extra_fields_for_sample(sample, args)
                interleaved_parts = interleaved_parts_for_sample(sample, args)
                settings = prompt_settings_dict(args)
                try:
                    pred_label, raw_text = run_one(
                        runtime=runtime,
                        sample=sample,
                        max_new_tokens=args.max_new_tokens,
                        retry_max_new_tokens=args.retry_max_new_tokens,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        prompt=prompt,
                        audio_path=audio_path,
                        interleaved_parts=interleaved_parts,
                    )
                    if pred_label is not None:
                        parsed += 1
                    is_correct = (
                        pred_label == sample.gold_label if pred_label is not None else None
                    )
                    if is_correct:
                        correct += 1

                    row = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "backend": backend,
                        "model_id": args.model_id,
                        "category": sample.category,
                        "benchmark_clip_id": sample.benchmark_clip_id,
                        "audio_path": str(audio_path),
                        "original_audio_path": str(sample.audio_path),
                        "gold_label": sample.gold_label,
                        "prediction_label": pred_label,
                        "parsed_ok": pred_label is not None,
                        "is_correct": is_correct,
                        "raw_response": raw_text,
                        "prompt": prompt,
                        "prompt_settings": settings,
                        "metadata": sample.metadata,
                        **extra_fields,
                    }
                except Exception as exc:
                    row = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "backend": backend,
                        "model_id": args.model_id,
                        "category": sample.category,
                        "benchmark_clip_id": sample.benchmark_clip_id,
                        "audio_path": str(audio_path),
                        "original_audio_path": str(sample.audio_path),
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
                        **extra_fields,
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
        "category": category,
        "total": total,
        "parsed": parsed,
        "correct": correct,
        "parsed_accuracy": parsed_acc,
        "strict_accuracy": strict_acc,
        "out_path": str(out_path),
        "prompt_settings": prompt_settings_dict(args),
    }


def close_runtime(runtime: AudioModelRuntime) -> None:
    if runtime.backend == "gemini":
        asyncio.run(runtime.aclose())
    else:
        runtime.close()


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()
    if args.out is None:
        args.out = default_output_path(args.category)

    args.model_id = resolve_model_id(args.backend, args.model_id)
    backend = resolve_backend(args.backend, args.model_id)
    args.temperature = (
        args.temperature
        if args.temperature is not None
        else default_temperature(backend, args.model_id)
    )
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
        run_category(
            runtime,
            args=args,
            backend=backend,
            category=args.category,
            out_path=args.out,
        )
        if backend == "gemini":
            print_gemini_cost(runtime, args)
    finally:
        close_runtime(runtime)


if __name__ == "__main__":
    main()

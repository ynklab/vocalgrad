"""Evaluate reviewer-requested paraphrases on their matching VocalGrad clips."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm.auto import tqdm

from .cli_summarize_vocalgrad import summarize_vocalgrad_file
from .cli_vocalgrad import (
    BACKEND_CHOICES,
    close_runtime,
    default_temperature,
    print_gemini_cost,
)
from .cli_vocalgrad_ternary_all import _model_stem
from .kimi_common import AudioModelRuntime, resolve_backend, resolve_model_id
from .vocalgrad import (
    DEFAULT_VOCALGRAD_DATASET_ROOT,
    VocalGradSample,
    load_local_vocalgrad_samples,
)


DEFAULT_CATEGORIES = ["volume", "voice_pitch", "speaking_speed"]
DEFAULT_RAW_ROOT = Path("outputs/raw/rebuttal/paraphrase_questions/default")
DEFAULT_ANALYSIS_ROOT = Path("outputs/analysis/rebuttal/paraphrase_questions/default")


@dataclass(frozen=True)
class PromptVariant:
    name: str
    question: str
    answer_words: tuple[str, str]

    @property
    def prompt(self) -> str:
        increase, decrease = self.answer_words
        return f'{self.question}\n\nAnswer with only one word: "{increase}" or "{decrease}".'


PROMPTS: dict[str, list[PromptVariant]] = {
    "volume": [
        PromptVariant(
            "rise_fall",
            "Does the loudness rise or fall as the audio progresses?",
            ("rise", "fall"),
        ),
        PromptVariant(
            "intensity_increase_decrease",
            "Does the sound intensity increase or decrease throughout the recording?",
            ("increase", "decrease"),
        ),
        PromptVariant(
            "intensity_increasing_decreasing",
            "Is the audio intensity increasing or decreasing over time?",
            ("increasing", "decreasing"),
        ),
    ],
    "voice_pitch": [
        PromptVariant(
            "tone_rise_fall", "Does the tone rise or fall over time?", ("rise", "fall")
        ),
        PromptVariant(
            "frequency_upward_downward",
            "Does the sound frequency change upward or downward over time?",
            ("upward", "downward"),
        ),
        PromptVariant(
            "frequency_rise_decline",
            "Over time, does the frequency of the voice rise or decline?",
            ("rise", "decline"),
        ),
    ],
    "speaking_speed": [
        PromptVariant(
            "pace_faster_slower",
            "Does the pace of speech become faster or slower over time?",
            ("faster", "slower"),
        ),
        PromptVariant(
            "tempo_increase_decrease",
            "Does the speech tempo increase or decrease with time?",
            ("increase", "decrease"),
        ),
        PromptVariant(
            "delivery_accelerate_decelerate",
            "Does the pace of delivery accelerate or decelerate over time?",
            ("accelerate", "decelerate"),
        ),
    ],
}

from .direction_evaluation import INCREASE_WORDS, DECREASE_WORDS, ANSWER_RE, parse_paraphrase_label


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run candidate question paraphrases on their matching VocalGrad categories."
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=DEFAULT_VOCALGRAD_DATASET_ROOT
    )
    parser.add_argument("--categories", nargs="*", default=DEFAULT_CATEGORIES.copy())
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out-stem", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--backend", default="auto", choices=BACKEND_CHOICES)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--retry-max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=5)
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


def _row(
    sample: VocalGradSample,
    variant: PromptVariant,
    backend: str,
    model_id: str,
    raw_text: str,
    prediction: str | None,
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "model_id": model_id,
        "category": sample.category,
        "benchmark_clip_id": sample.benchmark_clip_id,
        "audio_path": str(sample.audio_path),
        "gold_label": sample.gold_label,
        "prediction_label": prediction,
        "parsed_ok": prediction is not None,
        "is_correct": prediction == sample.gold_label
        if prediction is not None
        else None,
        "raw_response": raw_text,
        "prompt": variant.prompt,
        "prompt_variant": variant.name,
        "question": variant.question,
        "answer_words": {
            "increase": variant.answer_words[0],
            "decrease": variant.answer_words[1],
        },
        "metadata": sample.metadata,
    }


def _generate(
    runtime: AudioModelRuntime,
    sample: VocalGradSample,
    variant: PromptVariant,
    args: argparse.Namespace,
) -> tuple[str | None, str]:
    raw = runtime.generate_text(
        prompt=variant.prompt,
        audio_path=sample.audio_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    prediction = parse_paraphrase_label(raw)
    if prediction is not None:
        return prediction, raw
    retry = runtime.generate_text(
        prompt=variant.prompt,
        audio_path=sample.audio_path,
        max_new_tokens=args.retry_max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    return parse_paraphrase_label(retry), retry


async def _generate_async(
    runtime: AudioModelRuntime,
    sample: VocalGradSample,
    variant: PromptVariant,
    args: argparse.Namespace,
) -> tuple[str | None, str]:
    raw = await runtime.generate_text_async(
        prompt=variant.prompt,
        audio_path=sample.audio_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    prediction = parse_paraphrase_label(raw)
    if prediction is not None:
        return prediction, raw
    retry = await runtime.generate_text_async(
        prompt=variant.prompt,
        audio_path=sample.audio_path,
        max_new_tokens=args.retry_max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    return parse_paraphrase_label(retry), retry


def _run_sync(
    runtime: AudioModelRuntime,
    samples: list[VocalGradSample],
    variant: PromptVariant,
    backend: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in tqdm(
        samples, desc=f"paraphrase:{samples[0].category}:{variant.name}", unit="sample"
    ):
        try:
            prediction, raw = _generate(runtime, sample, variant, args)
            rows.append(_row(sample, variant, backend, args.model_id, raw, prediction))
        except Exception as exc:
            if args.raise_on_error:
                raise
            row = _row(sample, variant, backend, args.model_id, str(exc), None)
            row.update(
                {
                    "error": True,
                    "error_type": type(exc).__name__,
                    "traceback": traceback.format_exc(),
                }
            )
            rows.append(row)
    return rows


async def _run_async(
    runtime: AudioModelRuntime,
    samples: list[VocalGradSample],
    variant: PromptVariant,
    backend: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(1, args.concurrency))

    async def worker(sample: VocalGradSample) -> dict[str, Any]:
        async with semaphore:
            try:
                prediction, raw = await _generate_async(runtime, sample, variant, args)
                return _row(sample, variant, backend, args.model_id, raw, prediction)
            except Exception as exc:
                if args.raise_on_error:
                    raise
                row = _row(sample, variant, backend, args.model_id, str(exc), None)
                row.update(
                    {
                        "error": True,
                        "error_type": type(exc).__name__,
                        "traceback": traceback.format_exc(),
                    }
                )
                return row

    tasks = [asyncio.create_task(worker(sample)) for sample in samples]
    rows: list[dict[str, Any]] = []
    with tqdm(
        total=len(tasks),
        desc=f"paraphrase:{samples[0].category}:{variant.name}",
        unit="sample",
    ) as bar:
        for task in asyncio.as_completed(tasks):
            rows.append(await task)
            bar.update(1)
    return rows


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()
    unknown = sorted(set(args.categories) - set(PROMPTS))
    if not args.categories or unknown:
        raise ValueError(f"categories must be {DEFAULT_CATEGORIES}; unknown={unknown}")
    args.model_id = resolve_model_id(args.backend, args.model_id)
    backend = resolve_backend(args.backend, args.model_id)
    args.temperature = (
        args.temperature
        if args.temperature is not None
        else default_temperature(backend, args.model_id)
    )
    model_stem = args.out_stem or _model_stem(args.model_id)
    run_id = (
        args.run_id
        or f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{model_stem}"
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
    summaries: list[dict[str, Any]] = []
    try:
        for category in args.categories:
            samples = load_local_vocalgrad_samples(args.dataset_root, category)
            if args.max_samples is not None:
                samples = samples[: args.max_samples]
            for variant in PROMPTS[category]:
                rows = (
                    asyncio.run(_run_async(runtime, samples, variant, backend, args))
                    if backend == "gemini"
                    else _run_sync(runtime, samples, variant, backend, args)
                )
                raw_path = (
                    args.raw_root / category / variant.name / f"{model_stem}.jsonl"
                )
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_text(
                    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8",
                )
                analysis_path = (
                    args.analysis_root / category / variant.name / f"{model_stem}.json"
                )
                summary = summarize_vocalgrad_file(raw_path, analysis_path)
                summary.update(
                    {
                        "category": category,
                        "prompt_variant": variant.name,
                        "prompt": variant.prompt,
                        "answer_words": {
                            "increase": variant.answer_words[0],
                            "decrease": variant.answer_words[1],
                        },
                    }
                )
                analysis_path.write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                summaries.append(
                    {
                        "category": category,
                        "variant": variant.name,
                        "raw_path": str(raw_path),
                        "analysis_path": str(analysis_path),
                        "accuracy": summary["overall"]["accuracy"],
                    }
                )
                print(
                    f"done: category={category} variant={variant.name} accuracy={summary['overall']['accuracy']:.4f}"
                )
    finally:
        if backend == "gemini":
            print_gemini_cost(runtime, args)
        close_runtime(runtime)
    manifest = {
        "run_id": run_id,
        "backend": backend,
        "model_id": args.model_id,
        "dataset_root": str(args.dataset_root),
        "categories": args.categories,
        "max_samples": args.max_samples,
        "variants": summaries,
    }
    path = args.analysis_root / "_runs" / run_id / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"run manifest: {path}")


if __name__ == "__main__":
    main()

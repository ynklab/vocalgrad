"""Evaluate whether each VocalGrad attribute varies over time."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm.auto import tqdm

from .cli_vocalgrad import (
    BACKEND_CHOICES,
    close_runtime,
    default_temperature,
    print_gemini_cost,
)
from .cli_vocalgrad_ternary_all import (
    PAPER_CATEGORIES,
    TernarySample,
    _load_source_samples,
    _model_stem,
)
from .kimi_common import AudioModelRuntime, resolve_backend, resolve_model_id
from .vocalgrad import DEFAULT_VOCALGRAD_DATASET_ROOT, load_local_vocalgrad_samples


DEFAULT_SOURCE_ROOT = DEFAULT_VOCALGRAD_DATASET_ROOT / "source_clips"
DEFAULT_RAW_ROOT = Path("outputs/raw/rebuttal/binary_variation_question/default")
DEFAULT_ANALYSIS_ROOT = Path(
    "outputs/analysis/rebuttal/binary_variation_question/default"
)
YES_NO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the binary VocalGrad variation question on augmented and source clips."
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=DEFAULT_VOCALGRAD_DATASET_ROOT
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--source-manifest", type=Path, default=None)
    parser.add_argument("--categories", nargs="*", default=PAPER_CATEGORIES.copy())
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out-stem", default=None)
    parser.add_argument("--max-augmented-samples", type=int, default=None)
    parser.add_argument("--max-source-clips", type=int, default=None)
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


def build_binary_variation_prompt(category: str) -> str:
    attribute = {"voice_pitch": "voice pitch"}.get(category, category.replace("_", " "))
    return f'Does the {attribute} vary with time?\n\nAnswer with only one word: "yes" or "no".'


def parse_yes_no(raw_text: str) -> str | None:
    text = raw_text.strip().lower()
    if text in {"yes", "no"}:
        return text
    match = YES_NO_RE.search(text)
    return match.group(1).lower() if match else None


def _category_samples(
    dataset_root: Path,
    category: str,
    source_samples: list[TernarySample],
    maximum: int | None,
) -> list[TernarySample]:
    augmented = load_local_vocalgrad_samples(dataset_root, category)
    if maximum is not None:
        augmented = augmented[:maximum]
    return [
        TernarySample(
            category,
            sample.benchmark_clip_id,
            sample.audio_path,
            "yes",
            "augmented",
            sample.metadata,
        )
        for sample in augmented
    ] + [
        TernarySample(
            category,
            sample.benchmark_clip_id,
            sample.audio_path,
            "no",
            "source",
            sample.metadata,
        )
        for sample in source_samples
    ]


def _row(
    sample: TernarySample,
    backend: str,
    model_id: str,
    prompt: str,
    raw: str,
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
        "sample_origin": sample.sample_origin,
        "prediction_label": prediction,
        "parsed_ok": prediction is not None,
        "is_correct": prediction == sample.gold_label if prediction else None,
        "raw_response": raw,
        "prompt": prompt,
        "metadata": sample.metadata,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    table = {pred: {gold: 0 for gold in ("yes", "no")} for pred in ("yes", "no")}
    origins: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "parsed": 0, "correct": 0}
    )
    parsed = correct = 0
    for row in rows:
        origin = origins[row["sample_origin"]]
        origin["total"] += 1
        prediction = row.get("prediction_label")
        if prediction not in table:
            continue
        parsed += 1
        origin["parsed"] += 1
        table[prediction][row["gold_label"]] += 1
        if prediction == row["gold_label"]:
            correct += 1
            origin["correct"] += 1
    for values in origins.values():
        values["strict_accuracy"] = (
            values["correct"] / values["total"] if values["total"] else 0.0
        )
        values["parsed_accuracy"] = (
            values["correct"] / values["parsed"] if values["parsed"] else 0.0
        )
    return {
        "total": len(rows),
        "parsed": parsed,
        "correct": correct,
        "strict_accuracy": correct / len(rows) if rows else 0.0,
        "parsed_accuracy": correct / parsed if parsed else 0.0,
        "gold_label_counts": dict(Counter(row["gold_label"] for row in rows)),
        "by_origin": dict(origins),
        "confusion_matrix": {"labels": ["yes", "no"], "table": table},
    }


def _generate(
    runtime: AudioModelRuntime,
    sample: TernarySample,
    prompt: str,
    args: argparse.Namespace,
) -> tuple[str | None, str]:
    raw = runtime.generate_text(
        prompt=prompt,
        audio_path=sample.audio_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    prediction = parse_yes_no(raw)
    if prediction is not None:
        return prediction, raw
    raw = runtime.generate_text(
        prompt=prompt,
        audio_path=sample.audio_path,
        max_new_tokens=args.retry_max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    return parse_yes_no(raw), raw


async def _generate_async(
    runtime: AudioModelRuntime,
    sample: TernarySample,
    prompt: str,
    args: argparse.Namespace,
) -> tuple[str | None, str]:
    raw = await runtime.generate_text_async(
        prompt=prompt,
        audio_path=sample.audio_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    prediction = parse_yes_no(raw)
    if prediction is not None:
        return prediction, raw
    raw = await runtime.generate_text_async(
        prompt=prompt,
        audio_path=sample.audio_path,
        max_new_tokens=args.retry_max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    return parse_yes_no(raw), raw


def _run_sync(
    runtime: AudioModelRuntime,
    samples: list[TernarySample],
    backend: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    prompt = build_binary_variation_prompt(samples[0].category)
    rows: list[dict[str, Any]] = []
    for sample in tqdm(
        samples, desc=f"binary-variation:{samples[0].category}", unit="sample"
    ):
        try:
            prediction, raw = _generate(runtime, sample, prompt, args)
            rows.append(_row(sample, backend, args.model_id, prompt, raw, prediction))
        except Exception as exc:
            if args.raise_on_error:
                raise
            row = _row(sample, backend, args.model_id, prompt, str(exc), None)
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
    samples: list[TernarySample],
    backend: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    prompt = build_binary_variation_prompt(samples[0].category)
    semaphore = asyncio.Semaphore(max(1, args.concurrency))

    async def worker(sample: TernarySample) -> dict[str, Any]:
        async with semaphore:
            try:
                prediction, raw = await _generate_async(runtime, sample, prompt, args)
                return _row(sample, backend, args.model_id, prompt, raw, prediction)
            except Exception as exc:
                if args.raise_on_error:
                    raise
                row = _row(sample, backend, args.model_id, prompt, str(exc), None)
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
        total=len(tasks), desc=f"binary-variation:{samples[0].category}", unit="sample"
    ) as bar:
        for task in asyncio.as_completed(tasks):
            rows.append(await task)
            bar.update(1)
    return rows


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()
    unknown = sorted(set(args.categories) - set(PAPER_CATEGORIES))
    if not args.categories or unknown:
        raise ValueError(f"categories must be paper categories; unknown={unknown}")
    source_root = args.source_root.resolve()
    source_manifest = (args.source_manifest or source_root / "manifest.csv").resolve()
    original_sources = _load_source_samples(
        source_root, source_manifest, max_clips=args.max_source_clips
    )
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
    summaries: dict[str, dict[str, Any]] = {}
    try:
        for category in args.categories:
            samples = _category_samples(
                args.dataset_root,
                category,
                original_sources,
                args.max_augmented_samples,
            )
            rows = (
                asyncio.run(_run_async(runtime, samples, backend, args))
                if backend == "gemini"
                else _run_sync(runtime, samples, backend, args)
            )
            raw_path = args.raw_root / category / f"{model_stem}.jsonl"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            summary = _summary(rows)
            summary.update(
                {
                    "category": category,
                    "prompt": build_binary_variation_prompt(category),
                    "raw_path": str(raw_path),
                }
            )
            analysis_path = args.analysis_root / category / f"{model_stem}.json"
            analysis_path.parent.mkdir(parents=True, exist_ok=True)
            analysis_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            summaries[category] = summary
            print(
                f"done: category={category} strict_accuracy={summary['strict_accuracy']:.4f} raw={raw_path}"
            )
    finally:
        if backend == "gemini":
            print_gemini_cost(runtime, args)
        close_runtime(runtime)
    manifest = {
        "run_id": run_id,
        "backend": backend,
        "model_id": args.model_id,
        "categories": args.categories,
        "dataset_root": str(args.dataset_root),
        "source_root": str(source_root),
        "source_manifest": str(source_manifest),
        "prompt_template": 'Does the {attribute} vary with time?\\n\\nAnswer with only one word: "yes" or "no".',
        "max_augmented_samples": args.max_augmented_samples,
        "max_source_clips": args.max_source_clips,
        "summaries": summaries,
    }
    path = args.analysis_root / "_runs" / run_id / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"run manifest: {path}")


if __name__ == "__main__":
    main()

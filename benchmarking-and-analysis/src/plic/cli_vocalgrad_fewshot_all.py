from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from .cli_summarize_vocalgrad import summarize_vocalgrad_file
from .cli_vocalgrad import (
    BACKEND_CHOICES,
    build_prompt,
    close_runtime,
    default_temperature,
    print_gemini_cost,
    prompt_settings_dict,
    run_category,
)
from .kimi_common import AudioModelRuntime, resolve_backend, resolve_model_id
from .vocalgrad import (
    DEFAULT_VOCALGRAD_DATASET_ROOT,
    VocalGradSample,
    list_vocalgrad_categories,
    load_local_vocalgrad_samples,
)


DEFAULT_TRAIN_DATASET_ROOT = Path("datasets/vocalgrad/train")
DEFAULT_RAW_ROOT = Path("outputs/raw/vocalgrad_fewshot/default")
DEFAULT_ANALYSIS_ROOT = Path("outputs/analysis/vocalgrad_fewshot/default")


@dataclass(frozen=True)
class FewShotContext:
    exemplars: list[VocalGradSample]


class FewShotContextBuilder:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.train_by_category = {
            category: load_local_vocalgrad_samples(args.train_dataset_root, category)
            for category in args.categories
        }
        self.cache: dict[tuple[str, str], FewShotContext] = {}

    def context_for(self, sample: VocalGradSample) -> FewShotContext:
        key = (sample.category, sample.benchmark_clip_id)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        context = FewShotContext(exemplars=self._sample_exemplars(sample))
        self.cache[key] = context
        return context

    def _sample_exemplars(self, sample: VocalGradSample) -> list[VocalGradSample]:
        train_samples = self.train_by_category[sample.category]
        by_label = {
            "increase": [candidate for candidate in train_samples if candidate.gold_label == "increase"],
            "decrease": [candidate for candidate in train_samples if candidate.gold_label == "decrease"],
        }
        for label, candidates in by_label.items():
            if len(candidates) < self.args.shots_per_label:
                raise ValueError(
                    f"not enough {label} training clips for category={sample.category}: "
                    f"need={self.args.shots_per_label} found={len(candidates)}"
                )

        rng = random.Random(_stable_seed(self.args.seed, sample.category, sample.benchmark_clip_id))
        exemplars = [
            *rng.sample(by_label["increase"], self.args.shots_per_label),
            *rng.sample(by_label["decrease"], self.args.shots_per_label),
        ]
        rng.shuffle(exemplars)
        return exemplars


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run VocalGrad few-shot prompting. For each query clip, the audio context "
            "contains shuffled labeled training examples from the same category followed by the query."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_VOCALGRAD_DATASET_ROOT)
    parser.add_argument("--train-dataset-root", type=Path, default=DEFAULT_TRAIN_DATASET_ROOT)
    parser.add_argument("--categories", nargs="*", default=None)
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
    parser.add_argument("--shots-per-label", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    return parser


def _slugify_model_id(model_id: str) -> str:
    lowered = model_id.lower()
    if lowered == "moonshotai/kimi-audio-7b-instruct":
        return "kimi-audio"
    if lowered == "gemini-3-flash-preview":
        return "gemini-3-flash"
    if lowered == "nvidia/audio-flamingo-3-hf":
        return "audioflamingo3"
    if lowered == "stepfun-ai/step-audio-2-mini":
        return "step-audio-2-mini"
    if lowered == "xiaomimimo/mimo-audio-7b-instruct":
        return "mimo-audio"
    stem = model_id.split("/")[-1].lower()
    stem = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
    return stem or "model"


def _resolve_categories(dataset_root: Path, categories: list[str] | None) -> list[str]:
    discovered = list_vocalgrad_categories(dataset_root)
    if not categories:
        return discovered
    missing = [category for category in categories if category not in discovered]
    if missing:
        raise ValueError(f"unknown VocalGrad categories requested: {missing}; available={discovered}")
    return categories


def _stable_seed(seed: int, category: str, benchmark_clip_id: str) -> int:
    payload = f"{seed}:{category}:{benchmark_clip_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _display_attribute(category: str) -> str:
    return {"voice_pitch": "voice pitch"}.get(category, category.replace("_", " "))


def build_fewshot_prompt(sample: VocalGradSample, args: argparse.Namespace) -> str:
    context = args.fewshot_context_builder.context_for(sample)
    attribute = _display_attribute(sample.category)
    lines = [
        f"You will receive {len(context.exemplars)} labeled audio examples for {attribute}, followed by one query audio clip.",
        "",
        "Labeled examples:",
    ]
    for idx, exemplar in enumerate(context.exemplars, start=1):
        lines.append(f"Example {idx}: audio -> {exemplar.gold_label}")
    lines.extend(
        [
            "",
            "For the query audio, answer the same question:",
            build_prompt(sample.category, **prompt_settings_dict(args)),
        ]
    )
    return "\n".join(lines)


def build_fewshot_interleaved_parts(sample: VocalGradSample, args: argparse.Namespace) -> list[dict[str, object]]:
    context = args.fewshot_context_builder.context_for(sample)
    attribute = _display_attribute(sample.category)
    parts: list[dict[str, object]] = [
        {
            "type": "text",
            "text": (
                f"You will receive {len(context.exemplars)} labeled audio examples for {attribute}, "
                "followed by one query audio clip.\n\n"
            ),
        }
    ]
    for idx, exemplar in enumerate(context.exemplars, start=1):
        parts.extend(
            [
                {"type": "text", "text": f"Example {idx} audio:\n"},
                {"type": "audio", "path": str(exemplar.audio_path)},
                {"type": "text", "text": f"\nExample {idx} label: {exemplar.gold_label}\n\n"},
            ]
        )
    parts.extend(
        [
            {"type": "text", "text": "Query audio:\n"},
            {"type": "audio", "path": str(sample.audio_path)},
            {
                "type": "text",
                "text": (
                    "\nFor the query audio, answer the same question:\n"
                    f"{build_prompt(sample.category, **prompt_settings_dict(args))}"
                ),
            },
        ]
    )
    return parts


def build_fewshot_extra_fields(sample: VocalGradSample, args: argparse.Namespace) -> dict[str, object]:
    context = args.fewshot_context_builder.context_for(sample)
    return {
        "fewshot": {
            "shots_per_label": args.shots_per_label,
            "seed": args.seed,
            "context_format": "interleaved_audio_text_parts",
            "exemplars": [
                {
                    "clip_index": idx,
                    "category": exemplar.category,
                    "benchmark_clip_id": exemplar.benchmark_clip_id,
                    "audio_path": str(exemplar.audio_path),
                    "gold_label": exemplar.gold_label,
                    "metadata": exemplar.metadata,
                }
                for idx, exemplar in enumerate(context.exemplars, start=1)
            ],
            "query_clip_index": len(context.exemplars) + 1,
            "query_audio_path": str(sample.audio_path),
        }
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


def _summary_totals(summaries: list[dict[str, object]]) -> dict[str, float | int]:
    total = sum(int(summary["total"]) for summary in summaries)
    parsed = sum(int(summary["parsed"]) for summary in summaries)
    correct = sum(int(summary["correct"]) for summary in summaries)
    return {
        "total": total,
        "parsed": parsed,
        "correct": correct,
        "parsed_accuracy": (correct / parsed) if parsed else 0.0,
        "strict_accuracy": (correct / total) if total else 0.0,
    }


def _run_manifest_path(root: Path, run_id: str) -> Path:
    return root / "_runs" / run_id / "manifest.json"


def run_from_args(args: argparse.Namespace) -> None:
    args.model_id = resolve_model_id(args.backend, args.model_id)
    backend = resolve_backend(args.backend, args.model_id)
    args.temperature = args.temperature if args.temperature is not None else default_temperature(backend, args.model_id)
    args.categories = _resolve_categories(args.dataset_root, args.categories)
    model_stem = args.out_stem or _slugify_model_id(args.model_id)
    run_id = args.run_id or f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{model_stem}_fewshot"
    args.fewshot_context_builder = FewShotContextBuilder(args)
    args.sample_prompt_builder = build_fewshot_prompt
    args.sample_interleaved_parts_builder = build_fewshot_interleaved_parts
    args.sample_extra_fields_builder = build_fewshot_extra_fields

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
            f"running vocalgrad-fewshot-all backend={backend} model={args.model_id} "
            f"categories={len(args.categories)} shots_per_label={args.shots_per_label}"
        )
        for idx, category in enumerate(args.categories, start=1):
            print(f"[{idx}/{len(args.categories)}] category={category}")
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
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "backend": backend,
        "model_id": args.model_id,
        "model_stem": model_stem,
        "dataset_root": str(args.dataset_root),
        "train_dataset_root": str(args.train_dataset_root),
        "raw_root": str(args.raw_root),
        "analysis_root": str(args.analysis_root),
        "context_format": "interleaved_audio_text_parts",
        "categories": args.categories,
        "max_samples": args.max_samples,
        "max_new_tokens": args.max_new_tokens,
        "retry_max_new_tokens": args.retry_max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "swap_direction_order": args.swap_direction_order,
        "explicit_audio_clip_reference": args.explicit_audio_clip_reference,
        "shots_per_label": args.shots_per_label,
        "seed": args.seed,
        "generation_summaries": generation_summaries,
        "analysis_summaries": analysis_summaries,
        "totals": totals,
    }
    raw_manifest_path = _run_manifest_path(args.raw_root, run_id)
    raw_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    raw_manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.skip_analysis:
        analysis_manifest_path = _run_manifest_path(args.analysis_root, run_id)
        analysis_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        analysis_manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    load_dotenv()
    run_from_args(build_parser().parse_args())


if __name__ == "__main__":
    main()

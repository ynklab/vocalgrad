from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from plic.cli_vocalgrad import (  # noqa: E402
    BACKEND_CHOICES,
    build_prompt,
    default_temperature,
    parse_direction_label,
    print_gemini_cost,
)
from plic.kimi_common import AudioModelRuntime, resolve_backend, resolve_model_id  # noqa: E402
from plic.vocalgrad import (  # noqa: E402
    DEFAULT_VOCALGRAD_DATASET_ROOT,
    DEFAULT_VOCALGRAD_SOURCE_ROOT,
    list_vocalgrad_categories,
)

DEFAULT_SOURCE_ROOT = REPO_ROOT / DEFAULT_VOCALGRAD_SOURCE_ROOT
DEFAULT_MANIFEST_PATH = DEFAULT_SOURCE_ROOT / "manifest.csv"
DEFAULT_RAW_OUT_DIR = REPO_ROOT / "outputs" / "raw" / "source_clips_direction_generation"
DEFAULT_ANALYSIS_OUT_DIR = REPO_ROOT / "outputs" / "analysis" / "source_clips_direction_generation"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run VocalGrad-style generation-based direction classification on source clips. "
            "Each source clip is evaluated against every VocalGrad category prompt and the "
            "predicted increase/decrease ratio is summarized per category."
        )
    )
    parser.add_argument("--backend", required=True, choices=BACKEND_CHOICES)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / DEFAULT_VOCALGRAD_DATASET_ROOT)
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--speaker-ids", nargs="*", default=None)
    parser.add_argument(
        "--raw-out",
        type=Path,
        default=None,
        help="Per-prompt raw JSONL output path. Defaults to outputs/raw/source_clips_direction_generation/<model>.jsonl",
    )
    parser.add_argument(
        "--analysis-out",
        type=Path,
        default=None,
        help="Summary JSON output path. Defaults to outputs/analysis/source_clips_direction_generation/<model>.json",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--retry-max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--swap-direction-order",
        action="store_true",
        help="Ask about `decrease or increase` instead of `increase or decrease`.",
    )
    parser.add_argument(
        "--explicit-audio-clip-reference",
        action="store_true",
        help="Explicitly refer to the current clip in the prompt, matching VocalGrad options.",
    )
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
    if lowered == "stepfun-ai/step-audio-2-mini":
        return "step-audio-2-mini"
    if lowered == "xiaomimimo/mimo-audio-7b-instruct":
        return "mimo-audio"
    return model_id.split("/")[-1].lower().replace("/", "-")


def _prompt_settings_dict(args: argparse.Namespace) -> dict[str, bool]:
    return {
        "swap_direction_order": args.swap_direction_order,
        "explicit_audio_clip_reference": args.explicit_audio_clip_reference,
    }


def _build_source_prompt(category: str, args: argparse.Namespace) -> str:
    return build_prompt(category, **_prompt_settings_dict(args))


def _resolve_categories(dataset_root: Path, categories: list[str] | None) -> list[str]:
    discovered = list_vocalgrad_categories(dataset_root)
    if not categories:
        return discovered
    missing = [category for category in categories if category not in discovered]
    if missing:
        raise ValueError(
            f"unknown VocalGrad categories requested: {missing}; available={discovered}"
        )
    return categories


def _load_manifest_rows(
    manifest_path: Path,
    source_root: Path,
    *,
    max_clips: int | None,
    speaker_ids: set[str] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if speaker_ids and row["speaker_id"] not in speaker_ids:
                continue
            selected_rel = Path(row["selected_audio_path"])
            audio_path = source_root / selected_rel
            if not audio_path.exists():
                raise FileNotFoundError(f"source clip not found: {audio_path}")
            item = dict(row)
            item["audio_path"] = str(audio_path.resolve())
            rows.append(item)
            if max_clips is not None and len(rows) >= max_clips:
                break
    if not rows:
        raise ValueError("no source clips selected from manifest")
    return rows


def _count_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    parsed_rows = [row for row in rows if row.get("prediction_label") in {"increase", "decrease"}]
    increase_rows = [row for row in parsed_rows if row["prediction_label"] == "increase"]
    decrease_rows = [row for row in parsed_rows if row["prediction_label"] == "decrease"]

    raw_response_counter: Counter[str] = Counter(str(row.get("raw_response", "")) for row in rows)

    return {
        "n_records": total,
        "parsed_count": len(parsed_rows),
        "unparsed_count": total - len(parsed_rows),
        "parsed_rate": (len(parsed_rows) / total) if total else 0.0,
        "preferred_increase_count": len(increase_rows),
        "preferred_decrease_count": len(decrease_rows),
        "preferred_increase_rate": (len(increase_rows) / total) if total else 0.0,
        "preferred_decrease_rate": (len(decrease_rows) / total) if total else 0.0,
        "parsed_preferred_increase_rate": (len(increase_rows) / len(parsed_rows)) if parsed_rows else 0.0,
        "parsed_preferred_decrease_rate": (len(decrease_rows) / len(parsed_rows)) if parsed_rows else 0.0,
        "raw_response_top20": dict(raw_response_counter.most_common(20)),
    }


def _row_payload(
    *,
    backend: str,
    model_id: str,
    category: str,
    manifest_row: dict[str, Any],
    prompt: str,
    raw_text: str,
    pred_label: str | None,
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "model_id": model_id,
        "category": category,
        "test_item_id": manifest_row["test_item_id"],
        "clip_id": manifest_row["clip_id"],
        "speaker_id": manifest_row["speaker_id"],
        "sentence_id": manifest_row["sentence_id"],
        "audio_path": manifest_row["audio_path"],
        "prediction_label": pred_label,
        "parsed_ok": pred_label is not None,
        "raw_response": raw_text,
        "prompt": prompt,
        "prompt_settings": None,
        "text": manifest_row.get("text"),
        "metadata": manifest_row,
    }


def _run_one(
    runtime: AudioModelRuntime,
    *,
    prompt: str,
    audio_path: Path,
    max_new_tokens: int,
    retry_max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> tuple[str | None, str]:
    raw_text = runtime.generate_text(
        prompt=prompt,
        audio_path=audio_path,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    pred = parse_direction_label(raw_text)
    if pred is not None:
        return pred, raw_text

    retry_raw_text = runtime.generate_text(
        prompt=prompt,
        audio_path=audio_path,
        max_new_tokens=retry_max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    return parse_direction_label(retry_raw_text), retry_raw_text


async def _run_one_async(
    runtime: AudioModelRuntime,
    *,
    prompt: str,
    audio_path: Path,
    max_new_tokens: int,
    retry_max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> tuple[str | None, str]:
    raw_text = await runtime.generate_text_async(
        prompt=prompt,
        audio_path=audio_path,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    pred = parse_direction_label(raw_text)
    if pred is not None:
        return pred, raw_text

    retry_raw_text = await runtime.generate_text_async(
        prompt=prompt,
        audio_path=audio_path,
        max_new_tokens=retry_max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    return parse_direction_label(retry_raw_text), retry_raw_text


async def _run_async_batch(
    runtime: AudioModelRuntime,
    *,
    backend: str,
    model_id: str,
    tasks: list[tuple[dict[str, Any], str, str]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    rows: list[dict[str, Any]] = []

    async def worker(manifest_row: dict[str, Any], category: str, prompt: str) -> dict[str, Any]:
        audio_path = Path(manifest_row["audio_path"])
        async with semaphore:
            try:
                pred_label, raw_text = await _run_one_async(
                    runtime,
                    prompt=prompt,
                    audio_path=audio_path,
                    max_new_tokens=args.max_new_tokens,
                    retry_max_new_tokens=args.retry_max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                )
                row = _row_payload(
                    backend=backend,
                    model_id=model_id,
                    category=category,
                    manifest_row=manifest_row,
                    prompt=prompt,
                    raw_text=raw_text,
                    pred_label=pred_label,
                )
            except Exception as exc:
                if args.raise_on_error:
                    raise
                row = _row_payload(
                    backend=backend,
                    model_id=model_id,
                    category=category,
                    manifest_row=manifest_row,
                    prompt=prompt,
                    raw_text=str(exc),
                    pred_label=None,
                )
                row.update(
                    {
                        "error_type": type(exc).__name__,
                        "traceback": traceback.format_exc(),
                        "error": True,
                    }
                )
            row["prompt_settings"] = _prompt_settings_dict(args)
            return row

    pending = [asyncio.create_task(worker(manifest_row, category, prompt)) for manifest_row, category, prompt in tasks]
    with tqdm(total=len(pending), desc="source_clips", unit="prompt") as pbar:
        for task in asyncio.as_completed(pending):
            rows.append(await task)
            pbar.update(1)
    return rows


def _run_sync(
    runtime: AudioModelRuntime,
    *,
    backend: str,
    model_id: str,
    tasks: list[tuple[dict[str, Any], str, str]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with tqdm(total=len(tasks), desc="source_clips", unit="prompt") as pbar:
        for manifest_row, category, prompt in tasks:
            audio_path = Path(manifest_row["audio_path"])
            try:
                pred_label, raw_text = _run_one(
                    runtime,
                    prompt=prompt,
                    audio_path=audio_path,
                    max_new_tokens=args.max_new_tokens,
                    retry_max_new_tokens=args.retry_max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                )
                row = _row_payload(
                    backend=backend,
                    model_id=model_id,
                    category=category,
                    manifest_row=manifest_row,
                    prompt=prompt,
                    raw_text=raw_text,
                    pred_label=pred_label,
                )
            except Exception as exc:
                if args.raise_on_error:
                    raise
                row = _row_payload(
                    backend=backend,
                    model_id=model_id,
                    category=category,
                    manifest_row=manifest_row,
                    prompt=prompt,
                    raw_text=str(exc),
                    pred_label=None,
                )
                row.update(
                    {
                        "error_type": type(exc).__name__,
                        "traceback": traceback.format_exc(),
                        "error": True,
                    }
                )
            row["prompt_settings"] = _prompt_settings_dict(args)
            rows.append(row)
            pbar.update(1)
    return rows


def _close_runtime(runtime: AudioModelRuntime) -> None:
    if runtime.backend == "gemini":
        asyncio.run(runtime.aclose())
    else:
        runtime.close()


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()

    model_id = resolve_model_id(args.backend, args.model_id)
    backend = resolve_backend(args.backend, model_id)
    args.model_id = model_id
    args.temperature = (
        args.temperature if args.temperature is not None else default_temperature(backend, model_id)
    )

    source_root = args.source_root.resolve()
    manifest_path = args.manifest.resolve()
    if not source_root.exists():
        raise SystemExit(f"source root not found: {source_root}")
    if not manifest_path.exists():
        raise SystemExit(f"manifest not found: {manifest_path}")

    speaker_ids = set(args.speaker_ids) if args.speaker_ids else None
    categories = _resolve_categories(args.dataset_root, args.categories)
    manifest_rows = _load_manifest_rows(
        manifest_path,
        source_root,
        max_clips=args.max_clips,
        speaker_ids=speaker_ids,
    )

    model_stem = _slugify_model_id(model_id)
    raw_out = args.raw_out or (DEFAULT_RAW_OUT_DIR / f"{model_stem}.jsonl")
    analysis_out = args.analysis_out or (DEFAULT_ANALYSIS_OUT_DIR / f"{model_stem}.json")
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    analysis_out.parent.mkdir(parents=True, exist_ok=True)

    prompt_tasks: list[tuple[dict[str, Any], str, str]] = []
    for manifest_row in manifest_rows:
        for category in categories:
            prompt_tasks.append((manifest_row, category, _build_source_prompt(category, args)))

    runtime = AudioModelRuntime(
        backend=backend,
        model_id=model_id,
        cleanup_uploaded_files=args.cleanup_uploaded_files,
        upload_timeout_sec=args.upload_timeout_sec,
        max_retries=args.max_retries,
        initial_backoff_sec=args.initial_backoff_sec,
        max_backoff_sec=args.max_backoff_sec,
    )
    try:
        if backend == "gemini":
            rows = asyncio.run(
                _run_async_batch(
                    runtime,
                    backend=backend,
                    model_id=model_id,
                    tasks=prompt_tasks,
                    args=args,
                )
            )
        else:
            rows = _run_sync(
                runtime,
                backend=backend,
                model_id=model_id,
                tasks=prompt_tasks,
                args=args,
            )
    finally:
        if backend == "gemini":
            print_gemini_cost(runtime, args)
        _close_runtime(runtime)

    with raw_out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    category_summary = {
        category: _count_summary([row for row in rows if row["category"] == category])
        for category in categories
    }
    overall_summary = _count_summary(rows)

    payload = {
        "backend": backend,
        "model_id": model_id,
        "mode": "generation",
        "source_root": str(source_root),
        "manifest_path": str(manifest_path),
        "raw_output_path": str(raw_out),
        "n_source_clips": len(manifest_rows),
        "n_prompt_evaluations": len(rows),
        "categories": categories,
        "prompt_settings": _prompt_settings_dict(args),
        "max_new_tokens": args.max_new_tokens,
        "retry_max_new_tokens": args.retry_max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "overall_summary": overall_summary,
        "category_summary": category_summary,
    }
    analysis_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"wrote raw JSONL: {raw_out}")
    print(f"wrote summary JSON: {analysis_out}")
    print(
        "overall: "
        f"parsed={overall_summary['parsed_count']}/{overall_summary['n_records']} "
        f"pred_increase_rate={overall_summary['preferred_increase_rate']:.4f} "
        f"pred_decrease_rate={overall_summary['preferred_decrease_rate']:.4f}"
    )


if __name__ == "__main__":
    main()

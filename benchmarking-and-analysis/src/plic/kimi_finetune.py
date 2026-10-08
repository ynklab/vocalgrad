from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .vocalgrad import VocalGradSample, list_vocalgrad_categories, load_local_vocalgrad_samples

DEFAULT_KIMI_FINETUNE_DATA_ROOT = Path("outputs/data/vocalgrad_finetune")
DEFAULT_KIMI_FINETUNE_CHECKPOINT_ROOT = Path("outputs/checkpoints/vocalgrad_finetune")
DEFAULT_KIMI_FINETUNE_RAW_EVAL_ROOT = Path("outputs/raw/vocalgrad_finetune_eval")
DEFAULT_KIMI_FINETUNE_ANALYSIS_ROOT = Path("outputs/analysis/vocalgrad_finetune_eval")
KIMI_FINETUNE_SCOPE_CHOICES = ("lm_head_only", "all_linear")


@dataclass(frozen=True)
class FineTuneExample:
    category: str
    benchmark_clip_id: str
    audio_path: Path
    gold_label: str
    prompt: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "benchmark_clip_id": self.benchmark_clip_id,
            "audio_path": str(self.audio_path),
            "gold_label": self.gold_label,
            "prompt": self.prompt,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FineTuneExample":
        return cls(
            category=str(payload["category"]),
            benchmark_clip_id=str(payload["benchmark_clip_id"]),
            audio_path=Path(payload["audio_path"]),
            gold_label=str(payload["gold_label"]),
            prompt=str(payload["prompt"]),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class DirectionMetrics:
    accuracy: float
    balanced_accuracy: float
    accuracy_up: float | None
    accuracy_down: float | None
    total_evaluable: int
    total_rows: int
    parsed_rows: int
    correct_rows: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "vocalgrad-all-clips-v1",
            "metric": "accuracy_all_clips",
            "accuracy_all": self.accuracy,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "accuracy_up": self.accuracy_up,
            "accuracy_down": self.accuracy_down,
            "total_evaluable": self.total_evaluable,
            "total_rows": self.total_rows,
            "parsed_rows": self.parsed_rows,
            "correct_rows": self.correct_rows,
        }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_direction_label(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    lowered = value.strip().lower()
    if lowered in {"increase", "up", "upward"}:
        return "increase"
    if lowered in {"decrease", "down", "downward"}:
        return "decrease"
    return None


def summarize_direction_rows(rows: list[dict[str, Any]]) -> DirectionMetrics:
    from .direction_evaluation import binary_metrics, validate_rows
    validate_rows(rows)
    metrics = binary_metrics(rows)
    return DirectionMetrics(**{key: metrics[key] for key in DirectionMetrics.__dataclass_fields__})


def summarize_prediction_file(input_path: Path, out_path: Path, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    rows = read_jsonl(input_path)
    from .cli_summarize_vocalgrad import summarize_rows
    payload = summarize_rows(rows, str(input_path))
    if extra:
        payload.update(extra)
    write_json(out_path, payload)
    return payload


def resolve_categories(dataset_root: Path, categories: list[str] | None) -> list[str]:
    discovered = list_vocalgrad_categories(dataset_root)
    if not categories:
        return discovered
    missing = [category for category in categories if category not in discovered]
    if missing:
        raise ValueError(f"unknown VocalGrad categories requested: {missing}; available={discovered}")
    return categories


def make_finetune_examples(samples: Iterable[VocalGradSample]) -> list[FineTuneExample]:
    from .cli_vocalgrad import build_prompt
    return [
        FineTuneExample(
            category=sample.category,
            benchmark_clip_id=sample.benchmark_clip_id,
            audio_path=sample.audio_path,
            gold_label=sample.gold_label,
            prompt=build_prompt(sample.category),
            metadata=sample.metadata,
        )
        for sample in samples
    ]


def collect_category_examples(
    dataset_root: Path,
    category: str,
    *,
    max_samples: int | None = None,
) -> list[FineTuneExample]:
    samples = load_local_vocalgrad_samples(dataset_root=dataset_root, category=category)
    if max_samples is not None:
        samples = samples[:max_samples]
    return make_finetune_examples(samples)


def stratified_split_examples(
    examples: list[FineTuneExample],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[FineTuneExample], list[FineTuneExample]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
    if val_ratio == 0.0 or len(examples) <= 1:
        return list(examples), []

    grouped: dict[str, list[FineTuneExample]] = {"increase": [], "decrease": []}
    for example in examples:
        grouped.setdefault(example.gold_label, []).append(example)

    train_examples: list[FineTuneExample] = []
    val_examples: list[FineTuneExample] = []
    rng = random.Random(seed)
    for label, group in grouped.items():
        if not group:
            continue
        shuffled = list(group)
        rng.shuffle(shuffled)
        if len(shuffled) == 1:
            train_examples.extend(shuffled)
            continue
        n_val = max(1, int(round(len(shuffled) * val_ratio)))
        if n_val >= len(shuffled):
            n_val = len(shuffled) - 1
        val_examples.extend(shuffled[:n_val])
        train_examples.extend(shuffled[n_val:])

    def _sort_key(example: FineTuneExample) -> tuple[str, str]:
        return (example.category, example.benchmark_clip_id)

    train_examples.sort(key=_sort_key)
    val_examples.sort(key=_sort_key)
    return train_examples, val_examples


def build_kimi_supervised_messages(example: FineTuneExample) -> list[dict[str, Any]]:
    return [
        {"role": "user", "message_type": "text", "content": example.prompt},
        {"role": "user", "message_type": "audio", "content": str(example.audio_path)},
        {"role": "assistant", "message_type": "text", "content": example.gold_label},
    ]


def load_finetune_examples(path: Path) -> list[FineTuneExample]:
    return [FineTuneExample.from_dict(row) for row in read_jsonl(path)]


def load_kimi_runtime(
    model_id: str,
    *,
    adapter_path: Path | None = None,
    trainable_adapter: bool = False,
):
    import torch
    from kimia_infer.api.kimia import KimiAudio

    runtime = KimiAudio(model_path=model_id, load_detokenizer=False)
    if adapter_path is not None:
        from peft import PeftModel

        try:
            runtime.alm = PeftModel.from_pretrained(
                runtime.alm,
                str(adapter_path),
                is_trainable=trainable_adapter,
            )
        except TypeError:
            runtime.alm = PeftModel.from_pretrained(runtime.alm, str(adapter_path))
            if trainable_adapter:
                runtime.alm.train()
                for name, param in runtime.alm.named_parameters():
                    if "lora_" in name:
                        param.requires_grad_(True)
    runtime.alm = runtime.alm.to(torch.cuda.current_device())
    runtime.alm.eval()
    if not hasattr(runtime, "generate_text"):
        def _generate_text(
            *,
            prompt: str,
            audio_path: Path,
            max_new_tokens: int,
            temperature: float,
            top_k: int,
            response_schema: Any | None = None,
            response_mime_type: str | None = None,
        ) -> str:
            del response_schema, response_mime_type
            messages = [
                {"role": "user", "message_type": "text", "content": prompt},
                {"role": "user", "message_type": "audio", "content": str(audio_path)},
            ]
            _, raw_text = runtime.generate(
                messages,
                output_type="text",
                max_new_tokens=max_new_tokens,
                text_temperature=temperature,
                text_top_k=top_k,
                text_repetition_penalty=1.0,
                text_repetition_window_size=16,
                audio_temperature=0.0,
                audio_top_k=5,
                audio_repetition_penalty=1.0,
                audio_repetition_window_size=64,
            )
            return str(raw_text).strip()

        runtime.generate_text = _generate_text  # type: ignore[attr-defined]
    if not hasattr(runtime, "close"):
        runtime.close = lambda: None  # type: ignore[attr-defined]
    return runtime


def collect_lora_target_modules(model: object, scope: str) -> list[str]:
    import torch.nn as nn

    normalized = scope.strip().lower()
    if normalized not in KIMI_FINETUNE_SCOPE_CHOICES:
        raise ValueError(f"unsupported LoRA scope: {scope}")
    if normalized == "lm_head_only":
        return ["lm_head"]

    target_modules: list[str] = []
    for name, module in model.named_modules():
        if not name:
            continue
        if isinstance(module, nn.Linear):
            target_modules.append(name)
    return sorted(set(target_modules))


def model_stem_for(model_id: str) -> str:
    from .model_names import slugify_model_id
    return slugify_model_id(model_id)

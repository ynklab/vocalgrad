from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from .kimi_common import DEFAULT_MIMOAUDIO_MODEL_ID, DEFAULT_MIMOAUDIO_TOKENIZER_ID

MIMO_FINETUNE_SCOPE_CHOICES = ("lm_head_only", "all_linear")


def ensure_mimo_import_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    mimo_root = repo_root / "third_party" / "MiMo-Audio"
    if not mimo_root.exists():
        raise RuntimeError(
            "MiMo-Audio checkout not found at "
            f"{mimo_root}. Place the upstream repo under third_party/MiMo-Audio."
        )
    if str(mimo_root) not in sys.path:
        sys.path.insert(0, str(mimo_root))
    return mimo_root


def default_mimo_tokenizer_id() -> str:
    return os.getenv("MIMOAUDIO_TOKENIZER_ID", DEFAULT_MIMOAUDIO_TOKENIZER_ID)


def load_mimo_runtime(
    model_id: str = DEFAULT_MIMOAUDIO_MODEL_ID,
    *,
    tokenizer_id: str | None = None,
    adapter_path: Path | None = None,
    trainable_adapter: bool = False,
):
    ensure_mimo_import_path()

    from peft import PeftModel
    from src.mimo_audio.mimo_audio import MimoAudio

    runtime = MimoAudio(model_id, tokenizer_id or default_mimo_tokenizer_id())
    import torch

    runtime.torch = torch
    if adapter_path is not None:
        try:
            runtime.model = PeftModel.from_pretrained(
                runtime.model,
                str(adapter_path),
                is_trainable=trainable_adapter,
            )
        except TypeError:
            runtime.model = PeftModel.from_pretrained(runtime.model, str(adapter_path))
            if trainable_adapter:
                runtime.model.train()
                for name, param in runtime.model.named_parameters():
                    if "lora_" in name:
                        param.requires_grad_(True)
    runtime.model = runtime.model.to(runtime.device)
    runtime.model.eval()
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
            del temperature, top_k, response_schema, response_mime_type
            from src.mimo_audio.modeling_mimo_audio import MiMoStopper

            stopping_criteria = [
                MiMoStopper(
                    stop_tokens=[runtime.tokenizer.eos_token_id, runtime.im_end_idx],
                    group_size=runtime.group_size,
                    audio_channels=runtime.audio_channels,
                )
            ]
            input_ids = runtime.get_audio_understanding_sft_prompt(
                str(audio_path),
                prompt,
                thinking=False,
            )
            raw_text = runtime.forward(
                input_ids,
                stopping_criteria=stopping_criteria,
                max_new_tokens=max_new_tokens,
                task_name="audio_understanding",
            )
            return str(raw_text).strip()

        runtime.generate_text = _generate_text  # type: ignore[attr-defined]
    if not hasattr(runtime, "close"):
        runtime.close = lambda: None  # type: ignore[attr-defined]
    return runtime


def build_mimo_audio_understanding_ids(runtime: Any, audio_path: Path, prompt: str, answer: str | None = None):
    from src.mimo_audio.process_speechdata import InputSegment

    input_ids = runtime.get_audio_understanding_sft_prompt(
        str(audio_path),
        prompt,
        thinking=False,
    )
    if answer is None:
        return input_ids

    answer_ids = runtime.get_input_ids(
        [
            InputSegment(
                text=f"{answer}<|im_end|>\n",
                speech_zeroemb_idx=runtime.speech_zeroemb_idx,
                text_zeroemb_idx=runtime.empty_token,
            )
        ]
    )
    return runtime.torch.cat([input_ids, answer_ids], dim=1)


def collect_mimo_lora_target_modules(model: object, scope: str) -> list[str]:
    import torch.nn as nn

    normalized = scope.strip().lower()
    if normalized not in MIMO_FINETUNE_SCOPE_CHOICES:
        raise ValueError(f"unsupported MiMo LoRA scope: {scope}")
    if normalized == "lm_head_only":
        return ["lm_head"]

    target_modules: list[str] = []
    for name, module in model.named_modules():
        if not name:
            continue
        if isinstance(module, nn.Linear):
            target_modules.append(name)
    return sorted(set(target_modules))


def compute_mimo_text_loss(runtime: Any, input_ids: Any, prompt_token_count: int):
    torch = runtime.torch
    model = runtime.model
    base_model = getattr(model, "base_model", model)
    if hasattr(base_model, "model") and hasattr(base_model.model, "_prepare_input_embeds"):
        mimo_model = base_model.model
    else:
        mimo_model = base_model

    flat_input_ids = input_ids.T.reshape(1, -1).to(runtime.device)
    group_size = runtime.group_size
    group_count = flat_input_ids.shape[1] // ((runtime.audio_channels + 1) * group_size)
    attention_mask = torch.ones((1, group_count), dtype=torch.long, device=runtime.device)
    model_kwargs = mimo_model.prepare_inputs_for_generation(
        flat_input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    grouped_input_ids = model_kwargs["input_ids"]
    text_group_ids = grouped_input_ids[:, 0, ::group_size].long()
    prompt_group_count = prompt_token_count // group_size

    inputs_embeds = mimo_model._prepare_input_embeds(grouped_input_ids)
    outputs = mimo_model.model(
        inputs_embeds=inputs_embeds,
        attention_mask=model_kwargs["attention_mask"],
        position_ids=model_kwargs["position_ids"],
        use_cache=False,
        return_dict=True,
        cache_position=model_kwargs.get("cache_position"),
    )
    logits = mimo_model.lm_head(outputs.last_hidden_state).float()
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = text_group_ids[:, 1:].contiguous()

    label_positions = torch.arange(1, group_count, device=runtime.device).unsqueeze(0)
    loss_mask = label_positions >= prompt_group_count
    loss_mask &= shift_labels != -100
    loss_mask &= shift_labels != runtime.empty_token

    flat_loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        reduction="none",
    )
    flat_mask = loss_mask.view(-1).float()
    denom = flat_mask.sum().clamp(min=1.0)
    return (flat_loss * flat_mask).sum() / denom

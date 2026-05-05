from __future__ import annotations

from pathlib import Path
from typing import Any

from .kimi_common import DEFAULT_STEPAUDIO2_MODEL_ID
from .stepaudio2_runtime import StepAudio2Runtime, _padding_mels

STEPAUDIO2_FINETUNE_SCOPE_CHOICES = ("lm_head_only", "all_linear")


def load_stepaudio2_runtime(
    model_id: str = DEFAULT_STEPAUDIO2_MODEL_ID,
    *,
    adapter_path: Path | None = None,
    trainable_adapter: bool = False,
) -> StepAudio2Runtime:
    runtime = StepAudio2Runtime(model_id)
    if adapter_path is not None:
        from peft import PeftModel

        try:
            runtime.llm = PeftModel.from_pretrained(
                runtime.llm,
                str(adapter_path),
                is_trainable=trainable_adapter,
            )
        except TypeError:
            runtime.llm = PeftModel.from_pretrained(runtime.llm, str(adapter_path))
            if trainable_adapter:
                runtime.llm.train()
                for name, param in runtime.llm.named_parameters():
                    if "lora_" in name:
                        param.requires_grad_(True)
    runtime.llm = runtime.llm.cuda()
    runtime.llm.eval()
    if not hasattr(runtime, "close"):
        runtime.close = lambda: None  # type: ignore[attr-defined]
    return runtime


def collect_stepaudio2_lora_target_modules(model: object, scope: str) -> list[str]:
    import torch.nn as nn

    normalized = scope.strip().lower()
    if normalized not in STEPAUDIO2_FINETUNE_SCOPE_CHOICES:
        raise ValueError(f"unsupported Step-Audio-2 LoRA scope: {scope}")
    if normalized == "lm_head_only":
        return ["lm_head"]

    target_modules: list[str] = []
    for name, module in model.named_modules():
        if not name:
            continue
        if isinstance(module, nn.Linear):
            target_modules.append(name)
    return sorted(set(target_modules))


def build_stepaudio2_inputs(runtime: StepAudio2Runtime, audio_path: Path, prompt: str, answer: str | None = None):
    messages: list[dict[str, Any]] = [
        {
            "role": "human",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "audio", "audio": str(audio_path)},
            ],
        }
    ]
    if answer is None:
        messages.append({"role": "assistant", "content": None})
    else:
        messages.append({"role": "assistant", "content": str(answer)})

    chunks, mels = runtime._apply_chat_template(messages)
    prompt_ids = []
    for chunk in chunks:
        token_ids = runtime.llm_tokenizer(text=chunk, return_tensors="pt", padding=True)["input_ids"]
        prompt_ids.append(token_ids)
    input_ids = runtime.torch.cat(prompt_ids, dim=-1).cuda()
    attention_mask = runtime.torch.ones_like(input_ids)

    wavs = wav_lens = None
    if mels:
        wavs, wav_lens = _padding_mels(runtime.torch, runtime.pad_sequence, mels)
        wavs = wavs.cuda()
        wav_lens = wav_lens.cuda()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "wavs": wavs,
        "wav_lens": wav_lens,
    }


def compute_stepaudio2_text_loss(runtime: StepAudio2Runtime, batch: dict[str, Any], prompt_token_count: int):
    torch = runtime.torch
    outputs = runtime.llm(
        input_ids=batch["input_ids"],
        wavs=batch["wavs"],
        wav_lens=batch["wav_lens"],
        attention_mask=batch["attention_mask"],
    )
    logits = outputs.logits.float()
    labels = batch["input_ids"].long()
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    positions = torch.arange(1, labels.shape[1], device=labels.device).unsqueeze(0)
    loss_mask = positions >= prompt_token_count
    loss_mask &= shift_labels >= 0
    loss_mask &= shift_labels < shift_logits.shape[-1]
    loss_mask &= shift_labels != int(runtime.llm_tokenizer.pad_token_id)

    flat_loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.clamp(min=0, max=shift_logits.shape[-1] - 1).view(-1),
        reduction="none",
    )
    flat_mask = loss_mask.view(-1).float()
    denom = flat_mask.sum().clamp(min=1.0)
    return (flat_loss * flat_mask).sum() / denom

from __future__ import annotations

from pathlib import Path
from typing import Any

from .kimi_common import DEFAULT_AUDIOFLAMINGO3_MODEL_ID

AF3_FINETUNE_SCOPE_CHOICES = ("lm_head_only", "all_linear")


def load_af3_runtime(
    model_id: str = DEFAULT_AUDIOFLAMINGO3_MODEL_ID,
    *,
    adapter_path: Path | None = None,
    trainable_adapter: bool = False,
):
    import torch
    from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
    )
    if adapter_path is not None:
        from peft import PeftModel

        try:
            model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=trainable_adapter)
        except TypeError:
            model = PeftModel.from_pretrained(model, str(adapter_path))
            if trainable_adapter:
                model.train()
                for name, param in model.named_parameters():
                    if "lora_" in name:
                        param.requires_grad_(True)
    model.eval()

    class AudioFlamingo3FineTuneRuntime:
        backend = "audioflamingo3"

        def __init__(self) -> None:
            self.processor = processor
            self.model = model
            self.torch = torch

        @property
        def device(self):
            return next(self.model.parameters()).device

        def generate_text(
            self,
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
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "audio", "path": str(audio_path)},
                    ],
                }
            ]
            inputs = self.processor.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
            ).to(self.device)
            generate_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
            }
            if temperature > 0:
                generate_kwargs["temperature"] = temperature
                generate_kwargs["top_k"] = top_k
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, **generate_kwargs)
            decoded = self.processor.batch_decode(
                outputs[:, inputs.input_ids.shape[1] :],
                skip_special_tokens=True,
            )
            if not decoded:
                raise RuntimeError("AudioFlamingo3 returned no decoded text.")
            return str(decoded[0]).strip()

        def close(self) -> None:
            return None

    return AudioFlamingo3FineTuneRuntime()


def collect_af3_lora_target_modules(model: object, scope: str) -> list[str]:
    import torch.nn as nn

    normalized = scope.strip().lower()
    if normalized not in AF3_FINETUNE_SCOPE_CHOICES:
        raise ValueError(f"unsupported AudioFlamingo3 LoRA scope: {scope}")
    if normalized == "lm_head_only":
        return ["lm_head"]

    target_modules: list[str] = []
    for name, module in model.named_modules():
        if not name:
            continue
        if isinstance(module, nn.Linear):
            target_modules.append(name)
    return sorted(set(target_modules))


def build_af3_inputs(runtime: Any, audio_path: Path, prompt: str, answer: str | None = None):
    if answer is None:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio", "path": str(audio_path)},
                ],
            }
        ]
        return runtime.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        ).to(runtime.device)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "audio", "path": str(audio_path)},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": str(answer)}]},
    ]
    return runtime.processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
    ).to(runtime.device)


def compute_af3_text_loss(runtime: Any, batch: dict[str, Any]):
    model_inputs = {key: value for key, value in batch.items() if key != "prompt_token_count"}
    labels = batch["input_ids"].clone()
    labels[:, : batch["prompt_token_count"]] = -100
    audio_token_id = getattr(runtime.model.config, "audio_token_id", None)
    if audio_token_id is not None:
        labels[batch["input_ids"] == int(audio_token_id)] = -100
    outputs = runtime.model(**model_inputs, labels=labels, use_cache=False)
    return outputs.loss

from __future__ import annotations

import asyncio
import json
import os
import sys
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL_ID = "moonshotai/Kimi-Audio-7B-Instruct"
DEFAULT_GEMINI_MODEL_ID = "gemini-3-flash-preview"
DEFAULT_AUDIOFLAMINGO3_MODEL_ID = "nvidia/audio-flamingo-3-hf"
DEFAULT_STEPAUDIO2_MODEL_ID = "stepfun-ai/Step-Audio-2-mini"
DEFAULT_MIMOAUDIO_MODEL_ID = "XiaomiMiMo/MiMo-Audio-7B-Instruct"
DEFAULT_MIMOAUDIO_TOKENIZER_ID = "XiaomiMiMo/MiMo-Audio-Tokenizer"
DEFAULT_MIMOAUDIO_SEED = 1234


def default_model_id_for_backend(backend: str) -> str:
    if backend == "gemini":
        return DEFAULT_GEMINI_MODEL_ID
    if backend == "audioflamingo3":
        return DEFAULT_AUDIOFLAMINGO3_MODEL_ID
    if backend == "stepaudio2":
        return DEFAULT_STEPAUDIO2_MODEL_ID
    if backend == "mimoaudio":
        return DEFAULT_MIMOAUDIO_MODEL_ID
    return DEFAULT_MODEL_ID


def resolve_model_id(backend: str, model_id: str | None) -> str:
    if model_id:
        return model_id
    if backend == "auto":
        return DEFAULT_MODEL_ID
    return default_model_id_for_backend(backend)


def resolve_backend(backend: str, model_id: str) -> str:
    if backend not in {
        "auto",
        "kimia",
        "gemini",
        "audioflamingo3",
        "stepaudio2",
        "mimoaudio",
    }:
        raise ValueError(f"unsupported backend: {backend}")
    if backend != "auto":
        return backend

    lowered = model_id.lower()
    if "gemini" in lowered:
        return "gemini"
    if "audio-flamingo" in lowered or "audioflamingo" in lowered:
        return "audioflamingo3"
    if "step-audio" in lowered or "stepaudio" in lowered:
        return "stepaudio2"
    if "mimo-audio" in lowered or "mimoaudio" in lowered:
        return "mimoaudio"
    return "kimia"


def ensure_cuda_available(context: str = "Kimi-Audio inference") -> None:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            f"{context} requires PyTorch, but `torch` is not installed."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError(f"{context} requires CUDA, but CUDA is not available.")


def _apply_mimo_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


@dataclass
class UsageStats:
    request_count: int = 0
    retry_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add_usage(self, usage_metadata: Any) -> None:
        self.input_tokens += _usage_value(
            usage_metadata,
            ["prompt_token_count", "input_token_count"],
        )
        self.output_tokens += _usage_value(
            usage_metadata,
            ["candidates_token_count", "output_token_count"],
        )

    def estimate_cost_usd(
        self,
        input_usd_per_1m: float,
        output_usd_per_1m: float,
    ) -> float:
        in_cost = (self.input_tokens / 1_000_000.0) * input_usd_per_1m
        out_cost = (self.output_tokens / 1_000_000.0) * output_usd_per_1m
        return in_cost + out_cost


def _usage_value(usage_metadata: Any, keys: list[str]) -> int:
    for key in keys:
        val = _get_usage_field(usage_metadata, key)
        if isinstance(val, int):
            return val
    return 0


def _get_usage_field(usage_metadata: Any, key: str) -> Any:
    if usage_metadata is None:
        return None
    if isinstance(usage_metadata, dict):
        return usage_metadata.get(key)
    return getattr(usage_metadata, key, None)


def _maybe_to_dict(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, list):
        return [_maybe_to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {str(k): _maybe_to_dict(v) for k, v in obj.items()}
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="python")
        except Exception:
            pass
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return {
                str(k): _maybe_to_dict(v)
                for k, v in vars(obj).items()
                if not str(k).startswith("_")
            }
        except Exception:
            pass
    return repr(obj)


def _extract_candidate_texts(candidate: Any) -> list[str]:
    content = getattr(candidate, "content", None)
    parts = getattr(content, "parts", None)
    if not parts:
        return []

    texts: list[str] = []
    for part in parts:
        text = getattr(part, "text", None)
        if isinstance(text, str) and text.strip():
            texts.append(text)
    return texts


def summarize_gemini_response(response: Any) -> dict[str, Any]:
    candidates = getattr(response, "candidates", None) or []
    usage_metadata = getattr(response, "usage_metadata", None)
    prompt_feedback = getattr(response, "prompt_feedback", None)

    return {
        "text": getattr(response, "text", None),
        "parsed": _maybe_to_dict(getattr(response, "parsed", None)),
        "usage_metadata": _maybe_to_dict(usage_metadata),
        "prompt_feedback": _maybe_to_dict(prompt_feedback),
        "candidate_count": len(candidates),
        "candidates": [
            {
                "index": idx,
                "finish_reason": repr(getattr(candidate, "finish_reason", None)),
                "safety_ratings": _maybe_to_dict(
                    getattr(candidate, "safety_ratings", None)
                ),
                "citation_metadata": _maybe_to_dict(
                    getattr(candidate, "citation_metadata", None)
                ),
                "texts": _extract_candidate_texts(candidate),
            }
            for idx, candidate in enumerate(candidates)
        ],
    }


class AudioModelRuntime:
    def __init__(
        self,
        backend: str,
        model_id: str,
        *,
        cleanup_uploaded_files: bool = False,
        upload_timeout_sec: float = 180.0,
        max_retries: int = 5,
        initial_backoff_sec: float = 1.0,
        max_backoff_sec: float = 30.0,
        gemini_thinking_level: str | None = None,
    ) -> None:
        self.backend = backend
        self.model_id = model_id
        self.cleanup_uploaded_files = cleanup_uploaded_files
        self.upload_timeout_sec = upload_timeout_sec
        self.max_retries = max_retries
        self.initial_backoff_sec = initial_backoff_sec
        self.max_backoff_sec = max_backoff_sec
        self.gemini_thinking_level = gemini_thinking_level
        self.stats = UsageStats()
        self.mimo_seed: int | None = None

        if backend == "kimia":
            ensure_cuda_available("Kimi-Audio inference")
            from kimia_infer.api.kimia import KimiAudio

            print(f"loading kimi-audio model={model_id}")
            self.model = KimiAudio(model_path=model_id, load_detokenizer=False)
            self.client = None
            return

        if backend == "audioflamingo3":
            ensure_cuda_available("AudioFlamingo3 inference")
            try:
                from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
            except Exception as exc:
                raise RuntimeError(
                    "AudioFlamingo3 requires a Transformers build with AudioFlamingo3 support. "
                    "Install the `audioflamingo3` extra in its dedicated environment."
                ) from exc

            print(f"loading audioflamingo3 model={model_id}")
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
                model_id,
                device_map="auto",
            )
            self.model.eval()
            self.client = None
            self.async_client = None
            return

        if backend == "stepaudio2":
            ensure_cuda_available("Step-Audio-2-mini inference")
            try:
                from .stepaudio2_runtime import StepAudio2Runtime
            except Exception as exc:
                raise RuntimeError(
                    "Step-Audio-2-mini requires its dedicated environment and helper runtime. "
                    "Install the `stepaudio2` extra in .venv-stepaudio."
                ) from exc

            print(f"loading stepaudio2 model={model_id}")
            self.model = StepAudio2Runtime(model_id)
            self.client = None
            self.async_client = None
            return

        if backend == "mimoaudio":
            ensure_cuda_available("MiMo-Audio inference")
            mimo_root = Path(__file__).resolve().parents[2] / "third_party" / "MiMo-Audio"
            if not mimo_root.exists():
                raise RuntimeError(
                    "MiMo-Audio checkout not found at "
                    f"{mimo_root}. Place the upstream repo under third_party/MiMo-Audio."
                )
            if str(mimo_root) not in sys.path:
                sys.path.insert(0, str(mimo_root))

            try:
                from src.mimo_audio.mimo_audio import MimoAudio
                from src.mimo_audio.modeling_mimo_audio import MiMoStopper
            except Exception as exc:
                raise RuntimeError(
                    "MiMo-Audio backend requires the upstream checkout under third_party/MiMo-Audio "
                    "and its dependencies installed in the current environment."
                ) from exc

            tokenizer_id = os.getenv(
                "MIMOAUDIO_TOKENIZER_ID",
                DEFAULT_MIMOAUDIO_TOKENIZER_ID,
            )
            mimo_seed_env = os.getenv("MIMOAUDIO_SEED")
            if mimo_seed_env is None:
                self.mimo_seed = DEFAULT_MIMOAUDIO_SEED
            elif mimo_seed_env.strip().lower() in {"", "none", "off", "disable", "disabled"}:
                self.mimo_seed = None
            else:
                self.mimo_seed = int(mimo_seed_env)
            if self.mimo_seed is not None:
                _apply_mimo_seed(self.mimo_seed)
            print(
                f"loading mimoaudio model={model_id} tokenizer={tokenizer_id}"
                + (
                    f" seed={self.mimo_seed}"
                    if self.mimo_seed is not None
                    else " seed=disabled"
                )
            )
            self.model = MimoAudio(model_id, tokenizer_id)
            self.mimo_stopper_cls = MiMoStopper
            self.client = None
            self.async_client = None
            return

        if backend == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY is not set. Please export GEMINI_API_KEY before running."
                )
            try:
                from google import genai
            except Exception as exc:
                raise RuntimeError(
                    "google-genai is required for --backend gemini. "
                    "Install it with `uv add google-genai`."
                ) from exc

            self.client = genai.Client(api_key=api_key)
            self.async_client = self.client.aio
            self.model = None
            return

        raise ValueError(f"unsupported backend: {backend}")

    def close(self) -> None:
        if self.backend == "gemini" and getattr(self, "client", None) is not None:
            self.client.close()

    async def aclose(self) -> None:
        if self.backend == "gemini":
            async_client = getattr(self, "async_client", None)
            if async_client is not None:
                await async_client.aclose()
            client = getattr(self, "client", None)
            if client is not None:
                client.close()

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
        self.stats.request_count += 1
        if self.backend == "kimia":
            return self._generate_kimia(
                prompt=prompt,
                audio_path=audio_path,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
        if self.backend == "audioflamingo3":
            return self._generate_audioflamingo3(
                prompt=prompt,
                audio_path=audio_path,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
        if self.backend == "stepaudio2":
            return self._generate_stepaudio2(
                prompt=prompt,
                audio_path=audio_path,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
        if self.backend == "mimoaudio":
            return self._generate_mimoaudio(
                prompt=prompt,
                audio_path=audio_path,
                max_new_tokens=max_new_tokens,
            )

        return self._generate_gemini(
            prompt=prompt,
            audio_path=audio_path,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            response_schema=response_schema,
            response_mime_type=response_mime_type,
        )

    def generate_text_interleaved(
        self,
        *,
        parts: list[dict[str, Any]],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        response_schema: Any | None = None,
        response_mime_type: str | None = None,
    ) -> str:
        self.stats.request_count += 1
        if self.backend == "kimia":
            return self._generate_kimia_interleaved(
                parts=parts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
        if self.backend == "audioflamingo3":
            return self._generate_audioflamingo3_interleaved(
                parts=parts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
        if self.backend == "stepaudio2":
            return self._generate_stepaudio2_interleaved(
                parts=parts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
        if self.backend == "mimoaudio":
            return self._generate_mimoaudio_interleaved(
                parts=parts,
                max_new_tokens=max_new_tokens,
            )
        return self._generate_gemini_interleaved(
            parts=parts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            response_schema=response_schema,
            response_mime_type=response_mime_type,
        )

    async def generate_text_async(
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
        self.stats.request_count += 1
        if self.backend == "gemini":
            return await self._generate_gemini_async(
                prompt=prompt,
                audio_path=audio_path,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            response_schema=response_schema,
            response_mime_type=response_mime_type,
        )
        return await asyncio.to_thread(
            self.generate_text,
            prompt=prompt,
            audio_path=audio_path,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            response_schema=response_schema,
            response_mime_type=response_mime_type,
        )

    async def generate_text_interleaved_async(
        self,
        *,
        parts: list[dict[str, Any]],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        response_schema: Any | None = None,
        response_mime_type: str | None = None,
    ) -> str:
        self.stats.request_count += 1
        if self.backend == "gemini":
            return await self._generate_gemini_interleaved_async(
                parts=parts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                response_schema=response_schema,
                response_mime_type=response_mime_type,
            )
        return await asyncio.to_thread(
            self.generate_text_interleaved,
            parts=parts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            response_schema=response_schema,
            response_mime_type=response_mime_type,
        )

    def _generate_kimia(
        self,
        *,
        prompt: str,
        audio_path: Path,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> str:
        messages = [
            {"role": "user", "message_type": "text", "content": prompt},
            {"role": "user", "message_type": "audio", "content": str(audio_path)},
        ]
        _, raw_text = self.model.generate(
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

    def _generate_kimia_interleaved(
        self,
        *,
        parts: list[dict[str, Any]],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> str:
        messages: list[dict[str, str]] = []
        for part in parts:
            part_type = str(part.get("type", "")).lower()
            if part_type == "text":
                messages.append(
                    {
                        "role": "user",
                        "message_type": "text",
                        "content": str(part.get("text", "")),
                    }
                )
            elif part_type == "audio":
                messages.append(
                    {
                        "role": "user",
                        "message_type": "audio",
                        "content": str(part["path"]),
                    }
                )
            else:
                raise ValueError(f"unsupported interleaved part type for Kimi-Audio: {part_type}")
        _, raw_text = self.model.generate(
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


    def _generate_audioflamingo3(
        self,
        *,
        prompt: str,
        audio_path: Path,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> str:
        import torch

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
        ).to(self.model.device)

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_k"] = top_k

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **generate_kwargs)

        prompt_length = inputs.input_ids.shape[1]
        decoded = self.processor.batch_decode(
            outputs[:, prompt_length:],
            skip_special_tokens=True,
        )
        if not decoded:
            raise RuntimeError("AudioFlamingo3 returned no decoded text.")
        return str(decoded[0]).strip()

    def _generate_audioflamingo3_interleaved(
        self,
        *,
        parts: list[dict[str, Any]],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> str:
        import torch

        content: list[dict[str, str]] = []
        for part in parts:
            part_type = str(part.get("type", "")).lower()
            if part_type == "text":
                content.append({"type": "text", "text": str(part.get("text", ""))})
            elif part_type == "audio":
                content.append({"type": "audio", "path": str(part["path"])})
            else:
                raise ValueError(f"unsupported interleaved part type for AudioFlamingo3: {part_type}")

        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        ).to(self.model.device)

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_k"] = top_k

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **generate_kwargs)

        prompt_length = inputs.input_ids.shape[1]
        decoded = self.processor.batch_decode(outputs[:, prompt_length:], skip_special_tokens=True)
        if not decoded:
            raise RuntimeError("AudioFlamingo3 returned no decoded text.")
        return str(decoded[0]).strip()

    def _generate_stepaudio2(
        self,
        *,
        prompt: str,
        audio_path: Path,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> str:
        return self.model.generate_text(
            prompt=prompt,
            audio_path=audio_path,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    def _generate_stepaudio2_interleaved(
        self,
        *,
        parts: list[dict[str, Any]],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> str:
        content: list[dict[str, str]] = []
        for part in parts:
            part_type = str(part.get("type", "")).lower()
            if part_type == "text":
                content.append({"type": "text", "text": str(part.get("text", ""))})
            elif part_type == "audio":
                content.append({"type": "audio", "audio": str(part["path"])})
            else:
                raise ValueError(f"unsupported interleaved part type for Step-Audio-2: {part_type}")
        messages = [
            {"role": "human", "content": content},
            {"role": "assistant", "content": None},
        ]
        message_chunks, mels = self.model._apply_chat_template(messages)

        prompt_ids = []
        for chunk in message_chunks:
            if isinstance(chunk, str):
                token_ids = self.model.llm_tokenizer(text=chunk, return_tensors="pt", padding=True)["input_ids"]
                prompt_ids.append(token_ids)
            elif isinstance(chunk, list):
                prompt_ids.append(self.model.torch.tensor([chunk], dtype=self.model.torch.int32))
            else:
                raise ValueError(f"Unsupported prompt chunk type: {type(chunk)}")

        input_ids = self.model.torch.cat(prompt_ids, dim=-1).cuda()
        attention_mask = self.model.torch.ones_like(input_ids)

        wavs = wav_lens = None
        if mels:
            from .stepaudio2_runtime import _padding_mels

            wavs, wav_lens = _padding_mels(self.model.torch, self.model.pad_sequence, mels)
            wavs = wavs.cuda()
            wav_lens = wav_lens.cuda()

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.model.llm_tokenizer.pad_token_id,
            "eos_token_id": self.model.eos_token_id,
        }
        if temperature > 0:
            generation_kwargs.update({"temperature": temperature, "top_k": top_k, "do_sample": True})
        else:
            generation_kwargs["do_sample"] = False

        generation_config = self.model.GenerationConfig(**generation_kwargs)
        with self.model.torch.inference_mode():
            outputs = self.model.llm.generate(
                input_ids=input_ids,
                wavs=wavs,
                wav_lens=wav_lens,
                attention_mask=attention_mask,
                generation_config=generation_config,
                tokenizer=self.model.llm_tokenizer,
            )

        output_token_ids = outputs[0, input_ids.shape[-1] : -1].tolist()
        output_text_tokens = [token for token in output_token_ids if token < 151688]
        output_text = self.model.llm_tokenizer.decode(output_text_tokens).strip()
        if not output_text:
            output_text = self.model.llm_tokenizer.decode(output_token_ids, skip_special_tokens=True).strip()
        return output_text

    def _generate_mimoaudio(
        self,
        *,
        prompt: str,
        audio_path: Path,
        max_new_tokens: int,
    ) -> str:
        if self.mimo_seed is not None:
            _apply_mimo_seed(self.mimo_seed)

        stopping_criteria = [
            self.mimo_stopper_cls(
                stop_tokens=[
                    self.model.tokenizer.eos_token_id,
                    self.model.im_end_idx,
                ],
                group_size=self.model.group_size,
                audio_channels=self.model.audio_channels,
            )
        ]
        input_ids = self.model.get_audio_understanding_sft_prompt(
            str(audio_path),
            prompt,
            thinking=False,
        )
        raw_text = self.model.forward(
            input_ids,
            stopping_criteria=stopping_criteria,
            max_new_tokens=max_new_tokens,
            task_name="audio_understanding",
        )
        return str(raw_text).strip()

    def _generate_mimoaudio_interleaved(
        self,
        *,
        parts: list[dict[str, Any]],
        max_new_tokens: int,
    ) -> str:
        if self.mimo_seed is not None:
            _apply_mimo_seed(self.mimo_seed)

        from src.mimo_audio.process_speechdata import InputSegment

        lm_prompt = [
            InputSegment(
                text="<|im_start|>user\n",
                speech_zeroemb_idx=self.model.speech_zeroemb_idx,
                text_zeroemb_idx=self.model.empty_token,
            )
        ]
        for part in parts:
            part_type = str(part.get("type", "")).lower()
            if part_type == "text":
                lm_prompt.append(
                    InputSegment(
                        text=str(part.get("text", "")),
                        speech_zeroemb_idx=self.model.speech_zeroemb_idx,
                        text_zeroemb_idx=self.model.empty_token,
                    )
                )
            elif part_type == "audio":
                lm_prompt.append(
                    InputSegment(
                        audio=self.model.preprocess_input(str(part["path"])),
                        speech_zeroemb_idx=self.model.speech_zeroemb_idx,
                        text_zeroemb_idx=self.model.empty_token,
                    )
                )
            else:
                raise ValueError(f"unsupported interleaved part type for MiMo-Audio: {part_type}")
        lm_prompt.extend(
            [
                InputSegment(
                    text="<|im_end|>\n",
                    speech_zeroemb_idx=self.model.speech_zeroemb_idx,
                    text_zeroemb_idx=self.model.empty_token,
                ),
                InputSegment(
                    text="<|im_start|>assistant\n<think>\n\n</think>\n",
                    speech_zeroemb_idx=self.model.speech_zeroemb_idx,
                    text_zeroemb_idx=self.model.empty_token,
                ),
            ]
        )
        input_ids = self.model.get_input_ids(lm_prompt)
        stopping_criteria = [
            self.mimo_stopper_cls(
                stop_tokens=[self.model.tokenizer.eos_token_id, self.model.im_end_idx],
                group_size=self.model.group_size,
                audio_channels=self.model.audio_channels,
            )
        ]
        raw_text = self.model.forward(
            input_ids,
            stopping_criteria=stopping_criteria,
            max_new_tokens=max_new_tokens,
            task_name="audio_understanding",
        )
        return str(raw_text).strip()

    def _generate_gemini(
        self,
        *,
        prompt: str,
        audio_path: Path,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        response_schema: Any | None,
        response_mime_type: str | None,
    ) -> str:
        from google.genai import types

        uploaded = self._with_retry(
            lambda: self.client.files.upload(file=str(audio_path)),
            operation_name=f"upload:{audio_path.name}",
        )
        uploaded_name = getattr(uploaded, "name", None)

        try:
            uploaded = self._wait_for_uploaded_file(uploaded)

            config_kwargs: dict[str, Any] = {
                "temperature": temperature,
                "top_k": top_k,
                "max_output_tokens": max_new_tokens,
            }
            thinking_level = self.gemini_thinking_level
            if thinking_level is None:
                thinking_level = default_gemini_thinking_level(self.model_id)
            if thinking_level is not None:
                config_kwargs["thinking_config"] = {
                    "thinking_level": thinking_level,
                }
            if response_mime_type is not None:
                config_kwargs["response_mime_type"] = response_mime_type
            if response_schema is not None:
                config_kwargs["response_schema"] = response_schema
            config = types.GenerateContentConfig(**config_kwargs)

            response = self._with_retry(
                lambda: self.client.models.generate_content(
                    model=self.model_id,
                    contents=[prompt, uploaded],
                    config=config,
                ),
                operation_name=f"generate:{audio_path.name}",
            )

            usage_metadata = getattr(response, "usage_metadata", None)
            self.stats.add_usage(usage_metadata)
            text = getattr(response, "text", "")
            if text is None:
                summary = summarize_gemini_response(response)
                raise RuntimeError(
                    "Gemini returned no text response. "
                    f"Response summary: {json.dumps(summary, ensure_ascii=False)}"
                )

            text_str = str(text).strip()
            if not text_str:
                summary = summarize_gemini_response(response)
                raise RuntimeError(
                    "Gemini returned an empty text response. "
                    f"Response summary: {json.dumps(summary, ensure_ascii=False)}"
                )
            return text_str
        finally:
            if self.cleanup_uploaded_files and uploaded_name:
                try:
                    self._with_retry(
                        lambda: self.client.files.delete(name=uploaded_name),
                        operation_name=f"delete:{audio_path.name}",
                    )
                except Exception as exc:
                    print(f"warning: failed to delete uploaded file {uploaded_name}: {exc}")

    def _generate_gemini_interleaved(
        self,
        *,
        parts: list[dict[str, Any]],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        response_schema: Any | None,
        response_mime_type: str | None,
    ) -> str:
        from google.genai import types

        uploaded_files: list[Any] = []
        contents: list[Any] = []
        try:
            for part in parts:
                part_type = str(part.get("type", "")).lower()
                if part_type == "text":
                    contents.append(str(part.get("text", "")))
                elif part_type == "audio":
                    audio_path = Path(part["path"])
                    uploaded = self._with_retry(
                        lambda path=audio_path: self.client.files.upload(file=str(path)),
                        operation_name=f"upload:{audio_path.name}",
                    )
                    uploaded = self._wait_for_uploaded_file(uploaded)
                    uploaded_files.append(uploaded)
                    contents.append(uploaded)
                else:
                    raise ValueError(f"unsupported interleaved part type for Gemini: {part_type}")

            config_kwargs: dict[str, Any] = {
                "temperature": temperature,
                "top_k": top_k,
                "max_output_tokens": max_new_tokens,
            }
            thinking_level = self.gemini_thinking_level
            if thinking_level is None:
                thinking_level = default_gemini_thinking_level(self.model_id)
            if thinking_level is not None:
                config_kwargs["thinking_config"] = {"thinking_level": thinking_level}
            if response_mime_type is not None:
                config_kwargs["response_mime_type"] = response_mime_type
            if response_schema is not None:
                config_kwargs["response_schema"] = response_schema
            config = types.GenerateContentConfig(**config_kwargs)
            response = self._with_retry(
                lambda: self.client.models.generate_content(
                    model=self.model_id,
                    contents=contents,
                    config=config,
                ),
                operation_name="generate:interleaved",
            )
            usage_metadata = getattr(response, "usage_metadata", None)
            self.stats.add_usage(usage_metadata)
            text = getattr(response, "text", "")
            if text is None:
                summary = summarize_gemini_response(response)
                raise RuntimeError(
                    "Gemini returned no text response. "
                    f"Response summary: {json.dumps(summary, ensure_ascii=False)}"
                )
            text_str = str(text).strip()
            if not text_str:
                summary = summarize_gemini_response(response)
                raise RuntimeError(
                    "Gemini returned an empty text response. "
                    f"Response summary: {json.dumps(summary, ensure_ascii=False)}"
                )
            return text_str
        finally:
            if self.cleanup_uploaded_files:
                for uploaded in uploaded_files:
                    uploaded_name = getattr(uploaded, "name", None)
                    if not uploaded_name:
                        continue
                    try:
                        self._with_retry(
                            lambda name=uploaded_name: self.client.files.delete(name=name),
                            operation_name=f"delete:{uploaded_name}",
                        )
                    except Exception as exc:
                        print(f"warning: failed to delete uploaded file {uploaded_name}: {exc}")

    async def _generate_gemini_async(
        self,
        *,
        prompt: str,
        audio_path: Path,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        response_schema: Any | None,
        response_mime_type: str | None,
    ) -> str:
        from google.genai import types

        uploaded = await self._with_retry_async(
            lambda: self.async_client.files.upload(file=str(audio_path)),
            operation_name=f"upload:{audio_path.name}",
        )
        uploaded_name = getattr(uploaded, "name", None)

        try:
            uploaded = await self._wait_for_uploaded_file_async(uploaded)

            config_kwargs: dict[str, Any] = {
                "temperature": temperature,
                "top_k": top_k,
                "max_output_tokens": max_new_tokens,
            }
            thinking_level = self.gemini_thinking_level
            if thinking_level is None:
                thinking_level = default_gemini_thinking_level(self.model_id)
            if thinking_level is not None:
                config_kwargs["thinking_config"] = {
                    "thinking_level": thinking_level,
                }
            if response_mime_type is not None:
                config_kwargs["response_mime_type"] = response_mime_type
            if response_schema is not None:
                config_kwargs["response_schema"] = response_schema
            config = types.GenerateContentConfig(**config_kwargs)

            response = await self._with_retry_async(
                lambda: self.async_client.models.generate_content(
                    model=self.model_id,
                    contents=[prompt, uploaded],
                    config=config,
                ),
                operation_name=f"generate:{audio_path.name}",
            )

            usage_metadata = getattr(response, "usage_metadata", None)
            self.stats.add_usage(usage_metadata)
            text = getattr(response, "text", "")
            if text is None:
                summary = summarize_gemini_response(response)
                raise RuntimeError(
                    "Gemini returned no text response. "
                    f"Response summary: {json.dumps(summary, ensure_ascii=False)}"
                )

            text_str = str(text).strip()
            if not text_str:
                summary = summarize_gemini_response(response)
                raise RuntimeError(
                    "Gemini returned an empty text response. "
                    f"Response summary: {json.dumps(summary, ensure_ascii=False)}"
                )
            return text_str
        finally:
            if self.cleanup_uploaded_files and uploaded_name:
                try:
                    await self._with_retry_async(
                        lambda: self.async_client.files.delete(name=uploaded_name),
                        operation_name=f"delete:{audio_path.name}",
                    )
                except Exception as exc:
                    print(f"warning: failed to delete uploaded file {uploaded_name}: {exc}")

    async def _generate_gemini_interleaved_async(
        self,
        *,
        parts: list[dict[str, Any]],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        response_schema: Any | None,
        response_mime_type: str | None,
    ) -> str:
        from google.genai import types

        uploaded_files: list[Any] = []
        contents: list[Any] = []
        try:
            for part in parts:
                part_type = str(part.get("type", "")).lower()
                if part_type == "text":
                    contents.append(str(part.get("text", "")))
                elif part_type == "audio":
                    audio_path = Path(part["path"])
                    uploaded = await self._with_retry_async(
                        lambda path=audio_path: self.async_client.files.upload(file=str(path)),
                        operation_name=f"upload:{audio_path.name}",
                    )
                    uploaded = await self._wait_for_uploaded_file_async(uploaded)
                    uploaded_files.append(uploaded)
                    contents.append(uploaded)
                else:
                    raise ValueError(f"unsupported interleaved part type for Gemini: {part_type}")

            config_kwargs: dict[str, Any] = {
                "temperature": temperature,
                "top_k": top_k,
                "max_output_tokens": max_new_tokens,
            }
            thinking_level = self.gemini_thinking_level
            if thinking_level is None:
                thinking_level = default_gemini_thinking_level(self.model_id)
            if thinking_level is not None:
                config_kwargs["thinking_config"] = {"thinking_level": thinking_level}
            if response_mime_type is not None:
                config_kwargs["response_mime_type"] = response_mime_type
            if response_schema is not None:
                config_kwargs["response_schema"] = response_schema
            config = types.GenerateContentConfig(**config_kwargs)
            response = await self._with_retry_async(
                lambda: self.async_client.models.generate_content(
                    model=self.model_id,
                    contents=contents,
                    config=config,
                ),
                operation_name="generate:interleaved",
            )
            usage_metadata = getattr(response, "usage_metadata", None)
            self.stats.add_usage(usage_metadata)
            text = getattr(response, "text", "")
            if text is None:
                summary = summarize_gemini_response(response)
                raise RuntimeError(
                    "Gemini returned no text response. "
                    f"Response summary: {json.dumps(summary, ensure_ascii=False)}"
                )
            text_str = str(text).strip()
            if not text_str:
                summary = summarize_gemini_response(response)
                raise RuntimeError(
                    "Gemini returned an empty text response. "
                    f"Response summary: {json.dumps(summary, ensure_ascii=False)}"
                )
            return text_str
        finally:
            if self.cleanup_uploaded_files:
                for uploaded in uploaded_files:
                    uploaded_name = getattr(uploaded, "name", None)
                    if not uploaded_name:
                        continue
                    try:
                        await self._with_retry_async(
                            lambda name=uploaded_name: self.async_client.files.delete(name=name),
                            operation_name=f"delete:{uploaded_name}",
                        )
                    except Exception as exc:
                        print(f"warning: failed to delete uploaded file {uploaded_name}: {exc}")

    def _wait_for_uploaded_file(self, uploaded_file: Any) -> Any:
        name = getattr(uploaded_file, "name", None)
        if not name:
            return uploaded_file

        deadline = time.time() + self.upload_timeout_sec
        poll_interval = 1.0

        while True:
            current = self._with_retry(
                lambda: self.client.files.get(name=name),
                operation_name=f"files.get:{name}",
            )
            state = str(getattr(current, "state", "")).upper()

            if any(token in state for token in ("ACTIVE", "READY", "SUCCEEDED")):
                return current
            if any(token in state for token in ("FAILED", "ERROR", "CANCELLED")):
                raise RuntimeError(f"uploaded file processing failed: name={name} state={state}")
            if time.time() >= deadline:
                raise TimeoutError(
                    f"timed out waiting for uploaded file to become active: name={name} state={state}"
                )

            time.sleep(poll_interval)
            poll_interval = min(poll_interval * 1.5, 5.0)

    async def _wait_for_uploaded_file_async(self, uploaded_file: Any) -> Any:
        name = getattr(uploaded_file, "name", None)
        if not name:
            return uploaded_file

        deadline = time.time() + self.upload_timeout_sec
        poll_interval = 1.0

        while True:
            current = await self._with_retry_async(
                lambda: self.async_client.files.get(name=name),
                operation_name=f"files.get:{name}",
            )
            state = str(getattr(current, "state", "")).upper()

            if any(token in state for token in ("ACTIVE", "READY", "SUCCEEDED")):
                return current
            if any(token in state for token in ("FAILED", "ERROR", "CANCELLED")):
                raise RuntimeError(f"uploaded file processing failed: name={name} state={state}")
            if time.time() >= deadline:
                raise TimeoutError(
                    f"timed out waiting for uploaded file to become active: name={name} state={state}"
                )

            await asyncio.sleep(poll_interval)
            poll_interval = min(poll_interval * 1.5, 5.0)

    def _with_retry(self, func: Any, *, operation_name: str) -> Any:
        delay = self.initial_backoff_sec
        last_exc: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                return func()
            except Exception as exc:
                last_exc = exc
                if attempt >= self.max_retries or not _is_retryable_error(exc):
                    raise

                self.stats.retry_count += 1
                jitter = random.uniform(0.0, min(1.0, delay * 0.2))
                sleep_sec = min(delay + jitter, self.max_backoff_sec)
                print(
                    f"retrying {operation_name} "
                    f"(attempt {attempt + 1}/{self.max_retries}, sleep={sleep_sec:.2f}s): {exc}"
                )
                time.sleep(sleep_sec)
                delay = min(delay * 2.0, self.max_backoff_sec)

        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"unreachable retry state in {operation_name}")

    async def _with_retry_async(self, func: Any, *, operation_name: str) -> Any:
        delay = self.initial_backoff_sec
        last_exc: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                return await func()
            except Exception as exc:
                last_exc = exc
                if attempt >= self.max_retries or not _is_retryable_error(exc):
                    raise

                self.stats.retry_count += 1
                jitter = random.uniform(0.0, min(1.0, delay * 0.2))
                sleep_sec = min(delay + jitter, self.max_backoff_sec)
                print(
                    f"retrying {operation_name} "
                    f"(attempt {attempt + 1}/{self.max_retries}, sleep={sleep_sec:.2f}s): {exc}"
                )
                await asyncio.sleep(sleep_sec)
                delay = min(delay * 2.0, self.max_backoff_sec)

        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"unreachable retry state in {operation_name}")


def _is_retryable_error(exc: Exception) -> bool:
    message = str(exc).lower()
    retryable_phrases = [
        "429",
        "rate limit",
        "resource exhausted",
        "quota",
        "temporarily unavailable",
        "timeout",
        "503",
        "500",
        "502",
        "504",
    ]
    if any(phrase in message for phrase in retryable_phrases):
        return True

    code = getattr(exc, "code", None)
    if isinstance(code, int) and code in {429, 500, 502, 503, 504}:
        return True
    return False


def default_gemini_pricing(model_id: str) -> tuple[float, float]:
    if "gemini-3-flash-preview" in model_id:
        return (1.0, 3.0)
    return (0.0, 0.0)


def default_gemini_thinking_level(model_id: str) -> str | None:
    lowered = model_id.lower()
    if "gemini-3" in lowered:
        return "low"
    return None


def parse_json_object(raw_text: str) -> dict[str, Any] | None:
    text = raw_text.strip()
    if not text:
        return None

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def parse_gender_label(raw_text: str) -> str | None:
    obj = parse_json_object(raw_text)
    if obj:
        candidate = obj.get("gender") or obj.get("answer")
        if isinstance(candidate, str):
            normalized = _normalize_gender_token(candidate)
            if normalized is not None:
                return normalized

    t = raw_text.strip().upper()
    if t in {"M", "MALE"}:
        return "MALE"
    if t in {"F", "FEMALE"}:
        return "FEMALE"

    m = re.search(r"\b(MALE|FEMALE)\b", t)
    if m:
        return m.group(1)

    m2 = re.search(r"\b([MF])\b", t)
    if m2:
        return "MALE" if m2.group(1) == "M" else "FEMALE"
    return None


def parse_yes_no_label(raw_text: str) -> bool | None:
    obj = parse_json_object(raw_text)
    if obj:
        for key in ("answer", "contains_multiple", "multiple"):
            val = obj.get(key)
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                upper = val.strip().upper()
                if upper in {"YES", "Y", "TRUE", "MULTIPLE"}:
                    return True
                if upper in {"NO", "N", "FALSE", "SINGLE"}:
                    return False

    t = raw_text.strip().upper()
    if t in {"YES", "Y", "TRUE", "MULTIPLE"}:
        return True
    if t in {"NO", "N", "FALSE", "SINGLE"}:
        return False

    m = re.search(r"\b(YES|NO)\b", t)
    if m:
        return m.group(1) == "YES"
    return None


def _normalize_gender_token(token: str) -> str | None:
    t = token.strip().upper()
    if t in {"M", "MALE"}:
        return "MALE"
    if t in {"F", "FEMALE"}:
        return "FEMALE"
    return None


def load_speaker_gender_map(speaker_info_path: Path) -> dict[int, str]:
    if not speaker_info_path.exists():
        raise FileNotFoundError(f"speaker-info file not found: {speaker_info_path}")

    gender_by_id: dict[int, str] = {}
    lines = speaker_info_path.read_text(encoding="utf-8").splitlines()
    for line in lines[1:]:
        if not line.strip():
            continue
        cols = re.split(r"\s+", line.strip())
        if len(cols) < 3:
            continue
        try:
            spk_id = int(cols[0])
        except ValueError:
            continue
        normalized = _normalize_gender_token(cols[2])
        if normalized is not None:
            gender_by_id[spk_id] = normalized
    return gender_by_id

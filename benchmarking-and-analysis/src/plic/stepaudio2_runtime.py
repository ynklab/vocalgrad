from __future__ import annotations

from pathlib import Path
from typing import Any


def _mel_filters(torch_module: Any, librosa_module: Any, n_mels: int):
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"
    return torch_module.from_numpy(
        librosa_module.filters.mel(sr=16000, n_fft=400, n_mels=n_mels)
    )


def _load_audio(torchaudio_module: Any, file_path: str | Path, target_rate: int = 16000, max_length: int | None = None):
    waveform, sample_rate = torchaudio_module.load(str(file_path))
    if sample_rate != target_rate:
        waveform = torchaudio_module.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=target_rate,
        )(waveform)
    audio = waveform[0]
    if max_length is not None and audio.shape[0] > max_length:
        audio = audio[:max_length]
    return audio


def _log_mel_spectrogram(
    torch_module: Any,
    functional_module: Any,
    librosa_module: Any,
    audio: Any,
    *,
    n_mels: int = 128,
    padding: int = 479,
    device: Any = None,
):
    if not torch_module.is_tensor(audio):
        audio = torch_module.from_numpy(audio)
    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = functional_module.pad(audio, (0, padding))
    window = torch_module.hann_window(400).to(audio.device)
    stft = torch_module.stft(audio, 400, 160, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    filters = _mel_filters(torch_module, librosa_module, n_mels).to(audio.device)
    mel_spec = filters @ magnitudes
    log_spec = torch_module.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch_module.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


def _compute_token_num(max_feature_len: int) -> int:
    max_feature_len = max_feature_len - 2
    encoder_output_dim = (max_feature_len + 1) // 2 // 2
    padding = 1
    kernel_size = 3
    stride = 2
    adapter_output_dim = (encoder_output_dim + 2 * padding - kernel_size) // stride + 1
    return adapter_output_dim


def _padding_mels(torch_module: Any, pad_sequence: Any, data: list[Any]):
    feats_lengths = torch_module.tensor([sample.size(1) - 2 for sample in data], dtype=torch_module.int32)
    feats = [sample.t() for sample in data]
    padded_feats = pad_sequence(feats, batch_first=True, padding_value=0)
    return padded_feats.transpose(1, 2), feats_lengths


class StepAudio2Runtime:
    def __init__(self, model_path: str) -> None:
        import torch
        import librosa
        import torchaudio
        import torch.nn.functional as F
        from torch.nn.utils.rnn import pad_sequence
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

        self.torch = torch
        self.librosa = librosa
        self.torchaudio = torchaudio
        self.functional = F
        self.pad_sequence = pad_sequence
        self.GenerationConfig = GenerationConfig

        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="right",
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).cuda()
        self.llm.eval()

        self.llm_tokenizer.eos_token = "<|EOT|>"
        self.llm.config.eos_token_id = self.llm_tokenizer.convert_tokens_to_ids("<|EOT|>")
        self.eos_token_id = self.llm_tokenizer.convert_tokens_to_ids("<|EOT|>")

    def _apply_chat_template(self, messages: list[dict[str, Any]]):
        results: list[Any] = []
        mels: list[Any] = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                role = "human"

            if isinstance(content, str):
                text_with_audio = f"<|BOT|>{role}\n{content}"
                text_with_audio += "<|EOT|>" if msg.get("eot", True) else ""
                results.append(text_with_audio)
                continue

            if isinstance(content, list):
                results.append(f"<|BOT|>{role}\n")
                for item in content:
                    if item["type"] == "text":
                        results.append(f"{item['text']}")
                    elif item["type"] == "audio":
                        audio = _load_audio(self.torchaudio, item["audio"])
                        for start in range(0, audio.shape[0], 16000 * 25):
                            mel = _log_mel_spectrogram(
                                self.torch,
                                self.functional,
                                self.librosa,
                                audio[start : start + 16000 * 25],
                                n_mels=128,
                                padding=479,
                            )
                            mels.append(mel)
                            audio_tokens = "<audio_patch>" * _compute_token_num(mel.shape[1])
                            results.append(f"<audio_start>{audio_tokens}<audio_end>")
                    elif item["type"] == "token":
                        results.append(item["token"])
                    else:
                        raise ValueError(f"Unsupported content type: {item['type']}")
                if msg.get("eot", True):
                    results.append("<|EOT|>")
                continue

            if content is None:
                results.append(f"<|BOT|>{role}\n")
                continue

            raise ValueError(f"Unsupported message content type: {type(content)}")

        return results, mels

    def generate_text(
        self,
        *,
        prompt: str,
        audio_path: Path,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> str:
        messages = [
            {
                "role": "human",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio", "audio": str(audio_path)},
                ],
            },
            {"role": "assistant", "content": None},
        ]

        message_chunks, mels = self._apply_chat_template(messages)

        prompt_ids = []
        for chunk in message_chunks:
            if isinstance(chunk, str):
                token_ids = self.llm_tokenizer(text=chunk, return_tensors="pt", padding=True)["input_ids"]
                prompt_ids.append(token_ids)
            elif isinstance(chunk, list):
                prompt_ids.append(self.torch.tensor([chunk], dtype=self.torch.int32))
            else:
                raise ValueError(f"Unsupported prompt chunk type: {type(chunk)}")

        input_ids = self.torch.cat(prompt_ids, dim=-1).cuda()
        attention_mask = self.torch.ones_like(input_ids)

        wavs = None
        wav_lens = None
        if mels:
            wavs, wav_lens = _padding_mels(self.torch, self.pad_sequence, mels)
            wavs = wavs.cuda()
            wav_lens = wav_lens.cuda()

        generate_inputs = {
            "input_ids": input_ids,
            "wavs": wavs,
            "wav_lens": wav_lens,
            "attention_mask": attention_mask,
        }

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.llm_tokenizer.pad_token_id,
            "eos_token_id": self.eos_token_id,
        }
        if temperature > 0:
            generation_kwargs.update(
                {
                    "temperature": temperature,
                    "top_k": top_k,
                    "do_sample": True,
                }
            )
        else:
            generation_kwargs["do_sample"] = False

        generation_config = self.GenerationConfig(**generation_kwargs)
        with self.torch.inference_mode():
            outputs = self.llm.generate(
                **generate_inputs,
                generation_config=generation_config,
                tokenizer=self.llm_tokenizer,
            )

        output_token_ids = outputs[0, input_ids.shape[-1] : -1].tolist()
        output_text_tokens = [token for token in output_token_ids if token < 151688]
        output_text = self.llm_tokenizer.decode(output_text_tokens).strip()
        if not output_text:
            raise RuntimeError("Step-Audio-2-mini returned empty text output.")
        return output_text

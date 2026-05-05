from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .linear_probe import common_log_mel, flatten_segmented_sequence


def _is_tensor_like(value: object) -> bool:
    return hasattr(value, "shape") and hasattr(value, "float")


@dataclass(frozen=True)
class ExtractedStageFeatures:
    mel: np.ndarray
    pre_lm_audio: np.ndarray
    lm_audio: np.ndarray
    lm_text: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "mel": self.mel,
            "pre_lm_audio": self.pre_lm_audio,
            "lm_audio": self.lm_audio,
            "lm_text": self.lm_text,
        }


class BaseLinearProbeExtractor:
    def __init__(self, *, model_id: str, adapter_path: Path | None = None) -> None:
        self.model_id = model_id
        self.adapter_path = adapter_path

    def extract(self, *, audio_path: Path, prompt: str) -> ExtractedStageFeatures:
        raise NotImplementedError

    def _mel_feature(self, audio_path: Path) -> np.ndarray:
        return flatten_segmented_sequence(common_log_mel(audio_path))


class KimiLinearProbeExtractor(BaseLinearProbeExtractor):
    def __init__(self, *, model_id: str, adapter_path: Path | None = None) -> None:
        super().__init__(model_id=model_id, adapter_path=adapter_path)
        import torch

        from .kimi_finetune import load_kimi_runtime

        self.torch = torch
        self.runtime = load_kimi_runtime(model_id, adapter_path=adapter_path, trainable_adapter=False)
        self.prompt_manager = self.runtime.prompt_manager
        self.alm = self.runtime.alm
        self.extra_tokens = self.prompt_manager.extra_tokens
        self.device = torch.cuda.current_device()

    def _module_candidates(self) -> list[object]:
        candidates: list[object] = []
        seen: set[int] = set()
        stack = [self.alm]
        while stack:
            module = stack.pop(0)
            if module is None or id(module) in seen:
                continue
            seen.add(id(module))
            candidates.append(module)
            for attr in ("model", "base_model"):
                child = getattr(module, attr, None)
                if child is not None and id(child) not in seen:
                    stack.append(child)
        return candidates

    def _find_attr(self, attr_name: str):
        for module in self._module_candidates():
            if hasattr(module, attr_name):
                return getattr(module, attr_name)
        raise AttributeError(f"unable to resolve Kimi attribute `{attr_name}` through adapter/base wrapper stack")

    def _input_embeddings(self):
        getter = getattr(self.alm, "get_input_embeddings", None)
        if callable(getter):
            embeddings = getter()
            if embeddings is not None:
                return embeddings
        return self._find_attr("embed_tokens")

    def _vq_adaptor(self):
        return self._find_attr("vq_adaptor")

    def _forward_alm(self, **kwargs):
        return self.alm(**kwargs)

    def _num_text_path_states(self) -> int:
        config = getattr(self.alm, "config", None)
        if config is None or not hasattr(config, "num_hidden_layers"):
            config = self._find_attr("config")
        return int(config.num_hidden_layers) + 1

    def _build_history(self, audio_path: Path, prompt: str):
        messages = [
            {"role": "user", "message_type": "text", "content": prompt},
            {"role": "user", "message_type": "audio", "content": str(audio_path)},
        ]
        return self.prompt_manager.get_prompt(messages, output_type="text")

    def _pre_lm_audio_embeddings(self, audio_input_ids, is_continuous_mask, continuous_features):
        torch = self.torch
        with torch.inference_mode():
            input_ids = audio_input_ids.to(self.device)
            mask = is_continuous_mask.to(self.device)
            audio_emb = self._input_embeddings()(input_ids)

            if continuous_features:
                media_start_idx = (input_ids == self.extra_tokens.media_begin).nonzero()
                media_end_idx = (input_ids == self.extra_tokens.media_end).nonzero()
                feat_dim = continuous_features[0].shape[-1]
                whisper_dtype = continuous_features[0].dtype
                expanded_whisper = torch.zeros(
                    input_ids.shape[1],
                    feat_dim,
                    device=self.device,
                    dtype=whisper_dtype,
                )
                for seg_idx, ((_, start_idx), (_, end_idx)) in enumerate(zip(media_start_idx, media_end_idx)):
                    feat_len = int(end_idx - (start_idx + 1))
                    whisper_i = continuous_features[seg_idx].squeeze(0)
                    expanded_whisper[start_idx + 1 : end_idx, :] = whisper_i[:feat_len, :]
                expanded_whisper = expanded_whisper.unsqueeze(0)
                whisper_emb = self._vq_adaptor()(expanded_whisper.transpose(0, 1)).transpose(0, 1)
                combined = (audio_emb + whisper_emb) * np.sqrt(2.0)
                audio_emb = audio_emb * (~mask[:, :, None]) + combined * mask[:, :, None]

            return audio_emb[0, mask[0]].float().cpu().numpy()

    def extract(self, *, audio_path: Path, prompt: str) -> ExtractedStageFeatures:
        history = self._build_history(audio_path, prompt)
        audio_input_ids, text_input_ids, is_continuous_mask, _, _ = history.to_tensor()
        audio_input_ids = audio_input_ids.to(self.device)
        text_input_ids = text_input_ids.to(self.device)
        is_continuous_mask = is_continuous_mask.to(self.device)
        continuous_features = [f.to(self.device) for f in history.continuous_feature]

        pre_lm_audio_seq = self._pre_lm_audio_embeddings(
            audio_input_ids,
            is_continuous_mask,
            continuous_features,
        )

        with self.torch.inference_mode():
            outputs = self._forward_alm(
                input_ids=audio_input_ids,
                text_input_ids=text_input_ids,
                whisper_input_feature=continuous_features,
                is_continuous_mask=is_continuous_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )

        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        hidden_states = getattr(outputs, "hidden_states", None)
        if isinstance(last_hidden_state, (list, tuple)) and len(last_hidden_state) == 2:
            text_hidden, audio_hidden = last_hidden_state
        elif _is_tensor_like(last_hidden_state):
            text_hidden = last_hidden_state
            audio_hidden = last_hidden_state
        elif isinstance(hidden_states, (list, tuple)) and len(hidden_states) >= self._num_text_path_states():
            text_hidden = hidden_states[self._num_text_path_states() - 1]
            audio_hidden = text_hidden
        else:
            raise RuntimeError("unable to resolve Kimi hidden states for linear probing")
        audio_seq = audio_hidden[0, is_continuous_mask[0]].float().cpu().numpy()
        # Use the final assistant-start position, which is after the audio span and is the
        # state used to predict the first generated text token. The previous implementation
        # selected the last non-blank prompt token, which lives before the audio tokens and
        # becomes nearly constant across samples within a category.
        text_vec = text_hidden[0, -1].float().cpu().numpy()

        return ExtractedStageFeatures(
            mel=self._mel_feature(audio_path),
            pre_lm_audio=flatten_segmented_sequence(pre_lm_audio_seq),
            lm_audio=flatten_segmented_sequence(audio_seq),
            lm_text=text_vec.astype(np.float32, copy=False),
        )


class StepAudio2LinearProbeExtractor(BaseLinearProbeExtractor):
    def __init__(self, *, model_id: str) -> None:
        super().__init__(model_id=model_id)
        from .stepaudio2_runtime import StepAudio2Runtime, _padding_mels

        self.runtime = StepAudio2Runtime(model_id)
        self._padding_mels = _padding_mels
        self.torch = self.runtime.torch
        self.device = self.runtime.torch.cuda.current_device()
        self.audio_start_id = self.runtime.llm_tokenizer.convert_tokens_to_ids("<audio_start>")

    def _prepare_inputs(self, audio_path: Path, prompt: str):
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
        chunks, mels = self.runtime._apply_chat_template(messages)
        prompt_ids = []
        for chunk in chunks:
            token_ids = self.runtime.llm_tokenizer(text=chunk, return_tensors="pt", padding=True)["input_ids"]
            prompt_ids.append(token_ids)
        input_ids = self.torch.cat(prompt_ids, dim=-1).cuda()
        attention_mask = self.torch.ones_like(input_ids)
        wavs = wav_lens = None
        if mels:
            wavs, wav_lens = self._padding_mels(self.torch, self.runtime.pad_sequence, mels)
            wavs = wavs.cuda()
            wav_lens = wav_lens.cuda()
        return input_ids, attention_mask, wavs, wav_lens

    def extract(self, *, audio_path: Path, prompt: str) -> ExtractedStageFeatures:
        input_ids, attention_mask, wavs, wav_lens = self._prepare_inputs(audio_path, prompt)
        with self.torch.inference_mode():
            if wavs is not None and getattr(self.runtime.llm, "bf16", False):
                wavs = wavs.bfloat16()
            encoded, feat_lens = self.runtime.llm.encoder(wavs, wav_lens)
            adapted = self.runtime.llm.adapter(encoded)
            feat_lens = (feat_lens - 1) // 2 + 1

            pre_lm_audio_parts = [adapted[idx, : int(feat_lens[idx])].float().cpu().numpy() for idx in range(adapted.shape[0])]
            pre_lm_audio_seq = np.concatenate(pre_lm_audio_parts, axis=0)

            hidden_states = self.runtime.llm.model.embed_tokens(input_ids)
            audio_mask = self.torch.zeros(input_ids.shape[1], dtype=self.torch.bool, device=input_ids.device)
            insert_locations = self.torch.nonzero(input_ids == self.audio_start_id)
            for idx in range(len(insert_locations)):
                batch_idx, start_idx = insert_locations[idx]
                feat_len = int(feat_lens[idx])
                hidden_states[batch_idx, start_idx + 1 : start_idx + 1 + feat_len] = adapted[idx, :feat_len]
                audio_mask[start_idx + 1 : start_idx + 1 + feat_len] = True

            outputs = self.runtime.llm.model(
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            lm_hidden = outputs.last_hidden_state[0]

        audio_seq = lm_hidden[audio_mask].float().cpu().numpy()
        text_positions = (~audio_mask).nonzero(as_tuple=False)[:, 0]
        text_vec = lm_hidden[int(text_positions[-1].item())].float().cpu().numpy()

        return ExtractedStageFeatures(
            mel=self._mel_feature(audio_path),
            pre_lm_audio=flatten_segmented_sequence(pre_lm_audio_seq),
            lm_audio=flatten_segmented_sequence(audio_seq),
            lm_text=text_vec.astype(np.float32, copy=False),
        )


class AudioFlamingo3LinearProbeExtractor(BaseLinearProbeExtractor):
    def __init__(self, *, model_id: str) -> None:
        super().__init__(model_id=model_id)
        import torch
        from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.audio_token_id = self.model.config.audio_token_id

    def _prepare_inputs(self, audio_path: Path, prompt: str):
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
        )
        return {key: value.to(self.device) for key, value in inputs.items()}

    def extract(self, *, audio_path: Path, prompt: str) -> ExtractedStageFeatures:
        inputs = self._prepare_inputs(audio_path, prompt)
        with self.torch.inference_mode():
            audio_output = self.model.get_audio_features(
                inputs["input_features"],
                inputs["input_features_mask"],
                return_dict=True,
            )
            pre_lm_audio_seq = audio_output.pooler_output.float().cpu().numpy()
            inputs_embeds = self.model.get_input_embeddings()(inputs["input_ids"])
            audio_embeds = audio_output.pooler_output
            audio_token_mask = inputs["input_ids"] == self.audio_token_id
            inputs_embeds = inputs_embeds.masked_scatter(
                audio_token_mask.unsqueeze(-1),
                audio_embeds.to(inputs_embeds.device),
            )
            lm_outputs = self.model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )
            lm_hidden = lm_outputs.hidden_states[-1][0]

        audio_mask = (inputs["input_ids"][0] == self.audio_token_id)
        audio_seq = lm_hidden[audio_mask].float().cpu().numpy()
        text_positions = (~audio_mask).nonzero(as_tuple=False)[:, 0]
        text_vec = lm_hidden[int(text_positions[-1].item())].float().cpu().numpy()

        return ExtractedStageFeatures(
            mel=self._mel_feature(audio_path),
            pre_lm_audio=flatten_segmented_sequence(pre_lm_audio_seq),
            lm_audio=flatten_segmented_sequence(audio_seq),
            lm_text=text_vec.astype(np.float32, copy=False),
        )


class MimoLinearProbeExtractor(BaseLinearProbeExtractor):
    def __init__(self, *, model_id: str) -> None:
        super().__init__(model_id=model_id)
        import torch
        import torchaudio

        repo_root = Path(__file__).resolve().parents[2]
        mimo_root = repo_root / "third_party" / "MiMo-Audio"
        if str(mimo_root) not in sys.path:
            sys.path.insert(0, str(mimo_root))
        from src.mimo_audio.mimo_audio import MimoAudio

        self.torch = torch
        self.torchaudio = torchaudio
        self.runtime = MimoAudio(model_id, os.environ.get("MIMOAUDIO_TOKENIZER_ID", "XiaomiMiMo/MiMo-Audio-Tokenizer"))
        self.device = self.runtime.device

    def _load_resampled_wav(self, audio_path: Path):
        wav, sr = self.torchaudio.load(str(audio_path))
        if wav.ndim == 2:
            wav = wav.mean(dim=0)
        wav = self.runtime.resample_audio_if_needed(wav, sr)
        return wav.to(self.device)

    def _pre_lm_audio_sequence(self, audio_path: Path) -> np.ndarray:
        wav = self._load_resampled_wav(audio_path)
        mel = self.runtime.wav2mel(wav).transpose(0, 1)
        input_len = mel.size(0)
        segment_size = 6000
        input_len_seg = [segment_size] * (input_len // segment_size)
        if input_len % segment_size > 0:
            input_len_seg.append(input_len % segment_size)
        input_lens = self.torch.tensor(input_len_seg, device=self.device)
        with self.torch.inference_mode():
            hidden_states, _, output_lens, _ = self.runtime.mimo_audio_tokenizer.encode(
                mel,
                input_lens,
                use_quantizer=True,
            )
        parts = [hidden_states[idx, : int(output_lens[idx])].float().cpu().numpy() for idx in range(hidden_states.shape[0])]
        return np.concatenate(parts, axis=0)

    def _lm_sequences(self, audio_path: Path, prompt: str) -> tuple[np.ndarray, np.ndarray]:
        input_ids = self.runtime.get_audio_understanding_sft_prompt(str(audio_path), prompt, thinking=False)
        flat_input_ids = input_ids.T.reshape(1, -1).to(self.device)
        prompt_groups = flat_input_ids.shape[1] // ((self.runtime.audio_channels + 1) * self.runtime.group_size)
        attention_mask = self.torch.ones((1, prompt_groups), dtype=self.torch.long, device=self.device)
        model_kwargs = self.runtime.model.prepare_inputs_for_generation(
            flat_input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        grouped_input_ids = model_kwargs["input_ids"]
        inputs_embeds = self.runtime.model._prepare_input_embeds(grouped_input_ids)
        model_kwargs["inputs_embeds"] = inputs_embeds
        model_kwargs = self.runtime.model._get_initial_cache_position(flat_input_ids, model_kwargs)
        with self.torch.inference_mode():
            outputs = self.runtime.model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=model_kwargs["attention_mask"],
                position_ids=model_kwargs["position_ids"],
                past_key_values=model_kwargs.get("past_key_values"),
                use_cache=False,
                return_dict=True,
                cache_position=model_kwargs.get("cache_position"),
            )
        hidden = outputs.last_hidden_state[0]
        text_input_ids = grouped_input_ids[:, 0, :: self.runtime.group_size]
        is_audio_group = text_input_ids[0] == self.runtime.empty_token
        audio_seq = hidden[is_audio_group].float().cpu().numpy()
        text_positions = (~is_audio_group).nonzero(as_tuple=False)[:, 0]
        text_vec = hidden[int(text_positions[-1].item())].float().cpu().numpy()
        return audio_seq, text_vec

    def extract(self, *, audio_path: Path, prompt: str) -> ExtractedStageFeatures:
        pre_lm_audio_seq = self._pre_lm_audio_sequence(audio_path)
        lm_audio_seq, lm_text_vec = self._lm_sequences(audio_path, prompt)
        return ExtractedStageFeatures(
            mel=self._mel_feature(audio_path),
            pre_lm_audio=flatten_segmented_sequence(pre_lm_audio_seq),
            lm_audio=flatten_segmented_sequence(lm_audio_seq),
            lm_text=lm_text_vec.astype(np.float32, copy=False),
        )


def make_linear_probe_extractor(
    *,
    backend: str,
    model_id: str,
    adapter_path: Path | None = None,
) -> BaseLinearProbeExtractor:
    normalized = backend.lower()
    if normalized == "kimia":
        return KimiLinearProbeExtractor(model_id=model_id, adapter_path=adapter_path)
    if adapter_path is not None:
        raise ValueError(f"adapter_path is only supported for Kimi extraction, got backend={backend}")
    if normalized == "stepaudio2":
        return StepAudio2LinearProbeExtractor(model_id=model_id)
    if normalized == "audioflamingo3":
        return AudioFlamingo3LinearProbeExtractor(model_id=model_id)
    if normalized == "mimoaudio":
        return MimoLinearProbeExtractor(model_id=model_id)
    raise ValueError(f"unsupported backend for linear probe extraction: {backend}")

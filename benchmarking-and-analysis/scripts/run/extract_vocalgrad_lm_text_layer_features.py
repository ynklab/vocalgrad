from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from plic.kimi_common import resolve_backend, resolve_model_id  # noqa: E402
from plic.linear_probe import (  # noqa: E402
    collect_vocalgrad_samples,
    label_to_int,
    sample_prompt,
    slugify_model_id,
)
from plic.linear_probe_extractors import (  # noqa: E402
    AudioFlamingo3LinearProbeExtractor,
    KimiLinearProbeExtractor,
    MimoLinearProbeExtractor,
    StepAudio2LinearProbeExtractor,
)
from plic.vocalgrad import DEFAULT_VOCALGRAD_DATASET_ROOT, list_vocalgrad_categories  # noqa: E402


DEFAULT_LAYER_FEATURE_ROOT = Path("outputs/features/linear_probe_lm_text_layers")
LINEAR_PROBE_BACKEND_CHOICES = ["kimia", "audioflamingo3", "stepaudio2", "mimoaudio"]


def _is_tensor_like(value: object) -> bool:
    return hasattr(value, "shape") and hasattr(value, "float")


def _resolve_text_hidden_states(hidden_states: object) -> list[object]:
    if hidden_states is None:
        raise RuntimeError("hidden_states is None; model did not return per-layer activations")

    if isinstance(hidden_states, dict):
        for key in ("text", "text_hidden_states", "language", "lm_text"):
            candidate = hidden_states.get(key)
            if isinstance(candidate, (list, tuple)) and candidate and _is_tensor_like(candidate[0]):
                return list(candidate)

    if isinstance(hidden_states, (list, tuple)):
        if hidden_states and _is_tensor_like(hidden_states[0]):
            return list(hidden_states)
        for candidate in hidden_states:
            if isinstance(candidate, (list, tuple)) and candidate and _is_tensor_like(candidate[0]):
                return list(candidate)

    raise RuntimeError(
        "unable to resolve text hidden states from output_hidden_states payload "
        f"of type {type(hidden_states)}"
    )


def _stack_text_vectors(layer_hidden_states: list[object], token_index: int) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for layer_hidden in layer_hidden_states:
        vectors.append(layer_hidden[0, token_index].float().detach().cpu().numpy())
    return np.stack(vectors).astype(np.float32, copy=False)


def _extract_kimia_lm_text_layers(extractor: KimiLinearProbeExtractor, audio_path: Path, prompt: str) -> np.ndarray:
    history = extractor._build_history(audio_path, prompt)
    audio_input_ids, text_input_ids, is_continuous_mask, _, _ = history.to_tensor()
    audio_input_ids = audio_input_ids.to(extractor.device)
    text_input_ids = text_input_ids.to(extractor.device)
    is_continuous_mask = is_continuous_mask.to(extractor.device)
    continuous_features = [f.to(extractor.device) for f in history.continuous_feature]

    with extractor.torch.inference_mode():
        outputs = extractor._forward_alm(
            input_ids=audio_input_ids,
            text_input_ids=text_input_ids,
            whisper_input_feature=continuous_features,
            is_continuous_mask=is_continuous_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

    hidden_states = getattr(outputs, "hidden_states", None)
    if not isinstance(hidden_states, (list, tuple)):
        raise RuntimeError("Kimi hidden_states did not return a tuple of layer states")

    num_hidden_layers = getattr(extractor.alm, "config", None)
    if num_hidden_layers is None or not hasattr(num_hidden_layers, "num_hidden_layers"):
        config = extractor._find_attr("config")
        num_text_path_states = int(config.num_hidden_layers) + 1
    else:
        num_text_path_states = int(extractor.alm.config.num_hidden_layers) + 1
    if len(hidden_states) < num_text_path_states:
        raise RuntimeError(
            "Kimi hidden_states is shorter than expected for the text path: "
            f"got {len(hidden_states)} expected at least {num_text_path_states}"
        )

    text_hidden_states = list(hidden_states[:num_text_path_states])
    return _stack_text_vectors(text_hidden_states, token_index=-1)


def _extract_stepaudio2_lm_text_layers(
    extractor: StepAudio2LinearProbeExtractor,
    audio_path: Path,
    prompt: str,
) -> np.ndarray:
    input_ids, attention_mask, wavs, wav_lens = extractor._prepare_inputs(audio_path, prompt)
    with extractor.torch.inference_mode():
        if wavs is not None and getattr(extractor.runtime.llm, "bf16", False):
            wavs = wavs.bfloat16()
        encoded, feat_lens = extractor.runtime.llm.encoder(wavs, wav_lens)
        adapted = extractor.runtime.llm.adapter(encoded)
        feat_lens = (feat_lens - 1) // 2 + 1

        hidden_states = extractor.runtime.llm.model.embed_tokens(input_ids)
        audio_mask = extractor.torch.zeros(input_ids.shape[1], dtype=extractor.torch.bool, device=input_ids.device)
        insert_locations = extractor.torch.nonzero(input_ids == extractor.audio_start_id)
        for idx in range(len(insert_locations)):
            batch_idx, start_idx = insert_locations[idx]
            feat_len = int(feat_lens[idx])
            hidden_states[batch_idx, start_idx + 1 : start_idx + 1 + feat_len] = adapted[idx, :feat_len]
            audio_mask[start_idx + 1 : start_idx + 1 + feat_len] = True

        outputs = extractor.runtime.llm.model(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

    text_hidden_states = _resolve_text_hidden_states(getattr(outputs, "hidden_states", None))
    text_positions = (~audio_mask).nonzero(as_tuple=False)[:, 0]
    text_index = int(text_positions[-1].item())
    return _stack_text_vectors(text_hidden_states, token_index=text_index)


def _extract_audioflamingo3_lm_text_layers(
    extractor: AudioFlamingo3LinearProbeExtractor,
    audio_path: Path,
    prompt: str,
) -> np.ndarray:
    inputs = extractor._prepare_inputs(audio_path, prompt)
    with extractor.torch.inference_mode():
        audio_output = extractor.model.get_audio_features(
            inputs["input_features"],
            inputs["input_features_mask"],
            return_dict=True,
        )
        inputs_embeds = extractor.model.get_input_embeddings()(inputs["input_ids"])
        audio_embeds = audio_output.pooler_output
        audio_token_mask = inputs["input_ids"] == extractor.audio_token_id
        inputs_embeds = inputs_embeds.masked_scatter(
            audio_token_mask.unsqueeze(-1),
            audio_embeds.to(inputs_embeds.device),
        )
        outputs = extractor.model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
        )

    text_hidden_states = _resolve_text_hidden_states(getattr(outputs, "hidden_states", None))
    audio_mask = inputs["input_ids"][0] == extractor.audio_token_id
    text_positions = (~audio_mask).nonzero(as_tuple=False)[:, 0]
    text_index = int(text_positions[-1].item())
    return _stack_text_vectors(text_hidden_states, token_index=text_index)


def _extract_mimo_lm_text_layers(extractor: MimoLinearProbeExtractor, audio_path: Path, prompt: str) -> np.ndarray:
    input_ids = extractor.runtime.get_audio_understanding_sft_prompt(str(audio_path), prompt, thinking=False)
    flat_input_ids = input_ids.T.reshape(1, -1).to(extractor.device)
    prompt_groups = flat_input_ids.shape[1] // ((extractor.runtime.audio_channels + 1) * extractor.runtime.group_size)
    attention_mask = extractor.torch.ones((1, prompt_groups), dtype=extractor.torch.long, device=extractor.device)
    model_kwargs = extractor.runtime.model.prepare_inputs_for_generation(
        flat_input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    grouped_input_ids = model_kwargs["input_ids"]
    inputs_embeds = extractor.runtime.model._prepare_input_embeds(grouped_input_ids)
    model_kwargs["inputs_embeds"] = inputs_embeds
    model_kwargs = extractor.runtime.model._get_initial_cache_position(flat_input_ids, model_kwargs)
    with extractor.torch.inference_mode():
        outputs = extractor.runtime.model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=model_kwargs["attention_mask"],
            position_ids=model_kwargs["position_ids"],
            past_key_values=model_kwargs.get("past_key_values"),
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
            cache_position=model_kwargs.get("cache_position"),
        )

    text_hidden_states = _resolve_text_hidden_states(getattr(outputs, "hidden_states", None))
    text_input_ids = grouped_input_ids[:, 0, :: extractor.runtime.group_size]
    is_audio_group = text_input_ids[0] == extractor.runtime.empty_token
    text_positions = (~is_audio_group).nonzero(as_tuple=False)[:, 0]
    text_index = int(text_positions[-1].item())
    return _stack_text_vectors(text_hidden_states, token_index=text_index)


def _build_layer_extractor(backend: str, model_id: str, adapter_path: Path | None = None):
    normalized = backend.lower()
    if normalized == "kimia":
        instance = KimiLinearProbeExtractor(model_id=model_id, adapter_path=adapter_path)
        return lambda audio_path, prompt: _extract_kimia_lm_text_layers(instance, audio_path, prompt)
    if adapter_path is not None:
        raise ValueError(f"adapter_path is only supported for Kimi extraction, got backend={backend}")
    if normalized == "stepaudio2":
        instance = StepAudio2LinearProbeExtractor(model_id=model_id)
        return lambda audio_path, prompt: _extract_stepaudio2_lm_text_layers(instance, audio_path, prompt)
    if normalized == "audioflamingo3":
        instance = AudioFlamingo3LinearProbeExtractor(model_id=model_id)
        return lambda audio_path, prompt: _extract_audioflamingo3_lm_text_layers(instance, audio_path, prompt)
    if normalized == "mimoaudio":
        instance = MimoLinearProbeExtractor(model_id=model_id)
        return lambda audio_path, prompt: _extract_mimo_lm_text_layers(instance, audio_path, prompt)
    raise ValueError(f"unsupported backend for lm_text layer extraction: {backend}")


def _summarize_existing_feature_file(path: Path, category: str) -> dict[str, object]:
    loaded = np.load(path, allow_pickle=False)
    arrays = {key: loaded[key] for key in loaded.files}
    labels = arrays.get("label")
    layer_features = arrays.get("lm_text_layer_features")
    if labels is None or layer_features is None:
        raise KeyError(f"existing feature file is missing required arrays: {path}")
    return {
        "category": category,
        "n_samples": int(labels.shape[0]),
        "path": str(path),
        "n_layers": int(layer_features.shape[1]),
        "hidden_dim": int(layer_features.shape[2]),
        "skipped_existing": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract lm_text token representations from all transformer layers for VocalGrad linear probing."
    )
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / DEFAULT_VOCALGRAD_DATASET_ROOT)
    parser.add_argument(
        "--split-name",
        default=None,
        help="Optional split label used under outputs/features/linear_probe_lm_text_layers/.",
    )
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--max-samples-per-category", type=int, default=None)
    parser.add_argument("--backend", required=True, choices=LINEAR_PROBE_BACKEND_CHOICES)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--adapter-path", type=Path, default=None, help="Optional LoRA adapter path for Kimi extraction.")
    parser.add_argument(
        "--model-stem-override",
        default=None,
        help="Optional output model stem override, useful for treating a fine-tuned adapter as a separate model.",
    )
    parser.add_argument("--out-root", type=Path, default=REPO_ROOT / DEFAULT_LAYER_FEATURE_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing per-category .npz files instead of failing when they already exist.",
    )
    return parser


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()

    model_id = resolve_model_id(args.backend, args.model_id)
    backend = resolve_backend(args.backend, model_id)
    model_stem = args.model_stem_override or slugify_model_id(model_id)
    split_name = args.split_name or args.dataset_root.name
    categories = args.categories or list_vocalgrad_categories(args.dataset_root)

    layer_extractor = _build_layer_extractor(
        backend=backend,
        model_id=model_id,
        adapter_path=args.adapter_path,
    )
    out_dir = args.out_root / split_name / model_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[dict[str, object]] = []
    for category in categories:
        samples = collect_vocalgrad_samples(
            dataset_root=args.dataset_root,
            categories=[category],
            max_samples_per_category=args.max_samples_per_category,
        )
        out_path = out_dir / f"{category}.npz"
        if out_path.exists() and not args.overwrite:
            manifest_entries.append(_summarize_existing_feature_file(out_path, category))
            continue
        if not samples:
            raise ValueError(f"no VocalGrad samples found for category={category} under {args.dataset_root}")

        ids: list[str] = []
        labels: list[int] = []
        prompt_texts: list[str] = []
        per_sample_layer_features: list[np.ndarray] = []

        for sample in tqdm(samples, desc=f"extract layer-wise lm_text {category}"):
            prompt = sample_prompt(sample)
            layer_features = layer_extractor(sample.audio_path, prompt)
            ids.append(sample.benchmark_clip_id)
            labels.append(label_to_int(sample.gold_label))
            prompt_texts.append(prompt)
            per_sample_layer_features.append(np.asarray(layer_features, dtype=np.float32))

        stacked = np.stack(per_sample_layer_features).astype(np.float32, copy=False)
        if stacked.ndim != 3:
            raise ValueError(f"expected stacked layer features shape [N, L, D], got {stacked.shape}")
        layer_indices = np.arange(stacked.shape[1], dtype=np.int64)

        np.savez_compressed(
            out_path,
            benchmark_clip_id=np.asarray(ids, dtype="U64"),
            category=np.asarray([category] * len(ids), dtype="U64"),
            prompt=np.asarray(prompt_texts, dtype="U512"),
            label=np.asarray(labels, dtype=np.int64),
            layer_index=layer_indices,
            lm_text_layer_features=stacked,
        )
        manifest_entries.append(
            {
                "category": category,
                "n_samples": len(ids),
                "path": str(out_path),
                "n_layers": int(stacked.shape[1]),
                "hidden_dim": int(stacked.shape[2]),
            }
        )

    manifest = {
        "dataset_root": str(args.dataset_root),
        "split_name": split_name,
        "backend": backend,
        "model_id": model_id,
        "model_stem": model_stem,
        "adapter_path": str(args.adapter_path) if args.adapter_path is not None else None,
        "feature_type": "lm_text_layer_features",
        "categories": categories,
        "files": manifest_entries,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote layer-wise lm_text features to {out_dir}")


if __name__ == "__main__":
    main()

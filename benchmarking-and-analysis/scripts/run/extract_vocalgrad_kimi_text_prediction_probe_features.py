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

from plic.kimi_common import resolve_model_id  # noqa: E402
from plic.linear_probe import (
    collect_vocalgrad_samples,
    label_to_int,
    sample_prompt,
    slugify_model_id,
)  # noqa: E402
from plic.linear_probe_extractors import KimiLinearProbeExtractor  # noqa: E402
from plic.vocalgrad import DEFAULT_VOCALGRAD_DATASET_ROOT, list_vocalgrad_categories  # noqa: E402


DEFAULT_OUT_ROOT = Path(
    "outputs/features/rebuttal/text_prediction_representations_sparse4"
)
DEFAULT_SHARED_LAYER_INDICES = (1, 10, 25)
TEXT_POSITIONS_AUDIO_TO_QUERY = (
    "output",
    "query_last",
    "query_penultimate",
    "attribute_last",
)


def _messages(audio_path: Path, prompt: str, input_order: str) -> list[dict[str, str]]:
    text = {"role": "user", "message_type": "text", "content": prompt}
    audio = {"role": "user", "message_type": "audio", "content": str(audio_path)}
    return [text, audio] if input_order == "query_to_audio" else [audio, text]


def _history(
    extractor: KimiLinearProbeExtractor, audio_path: Path, prompt: str, input_order: str
):
    return extractor.prompt_manager.get_prompt(
        _messages(audio_path, prompt, input_order), output_type="text"
    )


def _token_ids(tokenizer: object, text: str) -> list[int]:
    """Return unwrapped, special-token-free ids for the runtime tokenizer."""
    if hasattr(tokenizer, "encode"):
        try:
            encoded = tokenizer.encode(text, bos=False, eos=False)
        except TypeError:
            encoded = tokenizer.encode(text, add_special_tokens=False)
    else:
        encoded = tokenizer(text, add_special_tokens=False)
    if isinstance(encoded, dict):
        ids = encoded["input_ids"]
    elif isinstance(encoded, (list, tuple)):
        ids = encoded
    else:
        ids = encoded.input_ids
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    while ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(value) for value in ids]


def _find_subsequence(
    haystack: list[int], needle: list[int], *, name: str
) -> tuple[int, int]:
    if not needle:
        raise ValueError(f"tokenization produced no tokens for {name}")
    matches = [
        idx
        for idx in range(len(haystack) - len(needle) + 1)
        if haystack[idx : idx + len(needle)] == needle
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {name} token span, found {len(matches)}"
        )
    return matches[0], matches[0] + len(needle)


def _attribute_text(category: str) -> str:
    return {"volume": "volume", "voice_pitch": "voice pitch"}.get(
        category, category.replace("_", " ")
    )


def _attribute_token_end(
    tokenizer: object, prompt_ids: list[int], category: str
) -> int:
    attribute = _attribute_text(category)
    for text in (attribute, f" {attribute}", f"the {attribute}", f" the {attribute}"):
        try:
            _, end = _find_subsequence(
                prompt_ids, _token_ids(tokenizer, text), name=f"attribute {text!r}"
            )
        except ValueError:
            continue
        return end
    raise ValueError(f"could not locate attribute token span for category={category!r}")


def _text_positions(
    extractor: KimiLinearProbeExtractor,
    text_input_ids: object,
    prompt: str,
    category: str,
    input_order: str,
) -> tuple[list[str], list[int]]:
    # Kimi's text ids are aligned to the joint sequence; zero marks non-text slots.
    ids = text_input_ids[0].detach().cpu().tolist()
    dense_positions = [idx for idx, token_id in enumerate(ids) if int(token_id) != 0]
    dense_ids = [int(ids[idx]) for idx in dense_positions]
    output_position = (
        len(ids) - 1
    )  # assistant-start, the state predicting answer token 1
    if input_order == "query_to_audio":
        return ["output"], [output_position]

    tokenizer = getattr(extractor.prompt_manager, "text_tokenizer", None)
    if tokenizer is None:
        tokenizer = getattr(extractor.prompt_manager, "tokenizer", None)
    if tokenizer is None:
        raise AttributeError(
            "Kimi prompt manager exposes neither text_tokenizer nor tokenizer"
        )
    prompt_ids = _token_ids(tokenizer, prompt)
    query_start, query_end = _find_subsequence(dense_ids, prompt_ids, name="query")
    if (
        query_end >= len(dense_positions)
        or dense_positions[query_end - 1] >= output_position
    ):
        raise ValueError("query span does not precede Kimi assistant-start position")
    relative_end = _attribute_token_end(tokenizer, prompt_ids, category)
    return list(TEXT_POSITIONS_AUDIO_TO_QUERY), [
        output_position,
        dense_positions[query_end - 1],
        dense_positions[query_end - 2],
        dense_positions[query_start + relative_end - 1],
    ]


def _segment_end_positions(audio_positions: list[int], num_segments: int) -> list[int]:
    if not audio_positions:
        raise ValueError("Kimi history has no continuous audio-token positions")
    boundaries = np.linspace(0, len(audio_positions), num_segments + 1, dtype=np.int64)
    return [
        audio_positions[max(int(boundaries[idx + 1]) - 1, int(boundaries[idx]))]
        for idx in range(num_segments)
    ]


def _stack(states: list[object], positions: list[int]) -> np.ndarray:
    return np.stack(
        [
            np.stack(
                [
                    state[0, position].float().detach().cpu().numpy()
                    for position in positions
                ]
            )
            for state in states
        ]
    ).astype(np.float32, copy=False)


def _select_states(
    states: list[object], indices: list[int], *, stream: str
) -> list[object]:
    if not indices:
        raise ValueError(f"no {stream} layer indices were requested")
    invalid = [index for index in indices if index < 0 or index >= len(states)]
    if invalid:
        raise ValueError(
            f"invalid {stream} layer indices {invalid}; available range is 0..{len(states) - 1}"
        )
    return [states[index] for index in indices]


def _extract_one(
    extractor: KimiLinearProbeExtractor,
    audio_path: Path,
    prompt: str,
    category: str,
    input_order: str,
    num_segments: int,
    shared_layer_indices: list[int],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    history = _history(extractor, audio_path, prompt, input_order)
    audio_input_ids, text_input_ids, is_continuous_mask, _, _ = history.to_tensor()
    audio_input_ids = audio_input_ids.to(extractor.device)
    text_input_ids = text_input_ids.to(extractor.device)
    is_continuous_mask = is_continuous_mask.to(extractor.device)
    continuous_features = [
        feature.to(extractor.device) for feature in history.continuous_feature
    ]
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
    if not isinstance(hidden_states, (tuple, list)):
        raise RuntimeError("Kimi did not return layer hidden states")
    text_state_count = extractor._num_text_path_states()
    text_states = list(hidden_states[:text_state_count])
    audio_positions = (
        is_continuous_mask[0].nonzero(as_tuple=False)[:, 0].detach().cpu().tolist()
    )
    segment_positions = _segment_end_positions(
        [int(value) for value in audio_positions], num_segments
    )
    text_names, text_positions = _text_positions(
        extractor, text_input_ids, prompt, category, input_order
    )
    selected_shared = _select_states(text_states, shared_layer_indices, stream="shared")
    return (
        _stack(selected_shared, segment_positions),
        _stack(selected_shared, text_positions),
        text_names,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract Kimi decoder representations for rebuttal probing."
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=REPO_ROOT / DEFAULT_VOCALGRAD_DATASET_ROOT
    )
    parser.add_argument("--split-name", required=True)
    parser.add_argument(
        "--input-order", choices=("query_to_audio", "audio_to_query"), required=True
    )
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--max-samples-per-category", type=int, default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--out-root", type=Path, default=REPO_ROOT / DEFAULT_OUT_ROOT)
    parser.add_argument("--num-audio-segments", type=int, default=4)
    parser.add_argument(
        "--shared-layer-indices",
        nargs="+",
        type=int,
        default=list(DEFAULT_SHARED_LAYER_INDICES),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()
    model_id = resolve_model_id("kimia", args.model_id)
    model_stem = slugify_model_id(model_id)
    categories = args.categories or list_vocalgrad_categories(args.dataset_root)
    out_dir = args.out_root / args.input_order / args.split_name / model_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    extractor = KimiLinearProbeExtractor(model_id=model_id)
    manifest_files: list[dict[str, object]] = []
    for category in categories:
        out_path = out_dir / f"{category}.npz"
        if out_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"feature cache already exists: {out_path}; pass --overwrite"
            )
        samples = collect_vocalgrad_samples(
            args.dataset_root,
            categories=[category],
            max_samples_per_category=args.max_samples_per_category,
        )
        if not samples:
            raise ValueError(f"no samples found for {category}")
        shared_audio_rows: list[np.ndarray] = []
        text_rows: list[np.ndarray] = []
        ids: list[str] = []
        labels: list[int] = []
        prompts: list[str] = []
        expected_text_names: list[str] | None = None
        for sample in tqdm(samples, desc=f"{args.input_order}:{category}"):
            prompt = sample_prompt(sample)
            shared_audio, text, text_names = _extract_one(
                extractor,
                sample.audio_path,
                prompt,
                category,
                args.input_order,
                args.num_audio_segments,
                args.shared_layer_indices,
            )
            if expected_text_names is None:
                expected_text_names = text_names
            elif expected_text_names != text_names:
                raise RuntimeError(
                    "text position names changed within one feature cache"
                )
            shared_audio_rows.append(shared_audio)
            text_rows.append(text)
            ids.append(sample.benchmark_clip_id)
            labels.append(label_to_int(sample.gold_label))
            prompts.append(prompt)
        shared_audio_features = np.stack(shared_audio_rows)
        text_features = np.stack(text_rows)
        np.savez_compressed(
            out_path,
            benchmark_clip_id=np.asarray(ids, dtype="U64"),
            label=np.asarray(labels, dtype=np.int64),
            prompt=np.asarray(prompts, dtype="U512"),
            shared_audio_segment_layer_features=shared_audio_features,
            shared_audio_layer_index=np.asarray(
                args.shared_layer_indices, dtype=np.int64
            ),
            audio_segment_index=np.arange(args.num_audio_segments, dtype=np.int64),
            text_layer_features=text_features,
            text_layer_index=np.asarray(args.shared_layer_indices, dtype=np.int64),
            text_position_name=np.asarray(expected_text_names, dtype="U32"),
        )
        manifest_files.append(
            {"category": category, "path": str(out_path), "n_samples": len(samples)}
        )
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "input_order": args.input_order,
                "split": args.split_name,
                "num_audio_segments": args.num_audio_segments,
                "shared_layer_indices": args.shared_layer_indices,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote Kimi rebuttal features to {out_dir}")


if __name__ == "__main__":
    main()

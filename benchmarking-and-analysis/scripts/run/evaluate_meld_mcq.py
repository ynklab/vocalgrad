from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, TYPE_CHECKING


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

if TYPE_CHECKING:
    from plic.kimi_common import AudioModelRuntime
from plic.model_names import slugify_model_id  # noqa: E402
from plic.meld import TASK_LABELS, build_meld_prompt, load_meld_test  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate local audio LMs on MELD multiple-choice classification."
    )
    p.add_argument("--backend", required=True, choices=("kimia",))
    p.add_argument("--model-id", default=None)
    p.add_argument(
        "--adapter-path",
        type=Path,
        default=None,
        help="Kimi-Audio LoRA adapter directory.",
    )
    p.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "datasets" / "meld")
    p.add_argument("--task", required=True, choices=tuple(TASK_LABELS))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    return p


def _tokenizer(runtime: AudioModelRuntime) -> tuple[object, dict[str, Any]]:
    return runtime.model.prompt_manager.text_tokenizer, {"bos": False, "eos": False}


def _candidate_ids(
    runtime: AudioModelRuntime, candidates: dict[str, str], *, kind: str
) -> dict[str, int]:
    # Score the single token for an ASCII space followed by the option letter.
    tokenizer, kwargs = _tokenizer(runtime)
    result: dict[str, int] = {}
    for key, text in candidates.items():
        ids = tokenizer.encode(f" {text}", **kwargs)
        if len(ids) != 1:
            raise RuntimeError(
                f"{runtime.backend} tokenizes {kind} {text!r} as {ids}; single-token MCQ scoring unavailable"
            )
        result[key] = int(ids[0])
    return result


def _score_candidate_set(
    final: Any, candidate_ids: dict[str, int], *, prefix: str
) -> dict[str, Any]:
    import torch

    selected = final[[candidate_ids[letter] for letter in candidate_ids]].float()
    probs = torch.softmax(selected, dim=0)
    letters = list(candidate_ids)
    if not bool(torch.isfinite(selected).all()):
        raise ValueError("Non-finite option-letter logits")
    # torch.argmax selects the first maximum, following candidate order A--G.
    index = int(torch.argmax(selected).item())
    return {
        f"{prefix}_logits": {
            letter: float(selected[idx].item()) for idx, letter in enumerate(letters)
        },
        f"{prefix}_probabilities": {
            letter: float(probs[idx].item()) for idx, letter in enumerate(letters)
        },
        f"{prefix}_argmax": letters[index],
    }


def _score_logits(
    logits: Any, letter_ids: dict[str, int]
) -> dict[str, Any]:
    final = logits[0, -1] if logits.dim() == 3 else logits[0]
    return _score_candidate_set(final, letter_ids, prefix="letter_choice")


def _logits_kimia(
    runtime: AudioModelRuntime,
    prompt: str,
    audio: Path,
    letter_ids: dict[str, int],
) -> dict[str, Any]:
    import torch

    history = runtime.model.prompt_manager.get_prompt(
        [
            {"role": "user", "message_type": "text", "content": prompt},
            {"role": "user", "message_type": "audio", "content": str(audio)},
        ],
        output_type="text",
    )
    audio_ids, text_ids, mask, _, _ = history.to_tensor()
    device = torch.cuda.current_device()
    with torch.inference_mode():
        _, logits, _ = runtime.model.alm.forward(
            input_ids=audio_ids.to(device),
            text_input_ids=text_ids.to(device),
            whisper_input_feature=[f.to(device) for f in history.continuous_feature],
            is_continuous_mask=mask.to(device),
            position_ids=torch.arange(audio_ids.shape[1], device=device).unsqueeze(0),
            past_key_values=None,
            return_dict=False,
        )
    return _score_logits(logits, letter_ids)


def _logit_prediction(
    runtime: AudioModelRuntime,
    prompt: str,
    audio: Path,
    letter_ids: dict[str, int],
) -> dict[str, Any]:
    return _logits_kimia(runtime, prompt, audio, letter_ids)


def _apply_kimi_adapter(runtime: AudioModelRuntime, adapter_path: Path) -> None:
    import torch
    from peft import PeftModel

    if runtime.backend != "kimia":
        raise ValueError("--adapter-path is supported only for --backend kimia")
    if not (adapter_path / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"Kimi adapter_config.json not found under {adapter_path}"
        )
    runtime.model.alm = PeftModel.from_pretrained(runtime.model.alm, str(adapter_path))
    runtime.model.alm = runtime.model.alm.to(torch.cuda.current_device())
    runtime.model.alm.eval()


def main() -> None:
    from dotenv import load_dotenv
    from tqdm.auto import tqdm
    from plic.kimi_common import AudioModelRuntime, resolve_backend, resolve_model_id

    load_dotenv()
    args = parser().parse_args()
    if args.out.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {args.out}; pass --overwrite")
    model_id = resolve_model_id(args.backend, args.model_id)
    backend = resolve_backend(args.backend, model_id)
    prompt, choices = build_meld_prompt(args.task)
    samples = load_meld_test(args.dataset_root)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    runtime = AudioModelRuntime(backend=backend, model_id=model_id)
    try:
        if args.adapter_path is not None:
            _apply_kimi_adapter(runtime, args.adapter_path)
        letter_ids = _candidate_ids(runtime, {letter: letter for letter in choices}, kind="option letter")
        with args.out.open("w", encoding="utf-8") as f:
            for sample in tqdm(
                samples, desc=f"MELD {args.task} {slugify_model_id(model_id)}"
            ):
                gold = getattr(sample, args.task)
                logits = _logit_prediction(
                    runtime, prompt, sample.audio_path, letter_ids
                )
                letter_argmax = logits["letter_choice_argmax"]
                row = {
                    "sample_id": sample.sample_id,
                    "audio_path": str(sample.audio_path),
                    "task": args.task,
                    "model_id": model_id,
                    "adapter_path": str(args.adapter_path)
                    if args.adapter_path
                    else None,
                    "gold_label": gold,
                    "prompt": prompt,
                    "choices": choices,
                    "candidate_letter_token_ids": letter_ids,
                    "logit_letter_prediction": choices[letter_argmax],
                    "logit_letter_correct": choices[letter_argmax] == gold,
                }
                row.update(logits)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        runtime.close()
    print(f"wrote MELD {args.task} results to {args.out}")


if __name__ == "__main__":
    main()

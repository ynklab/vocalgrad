from __future__ import annotations

import argparse
import math
import random
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm.auto import tqdm

from plic.af3_finetune import (
    AF3_FINETUNE_SCOPE_CHOICES,
    build_af3_inputs,
    collect_af3_lora_target_modules,
    compute_af3_text_loss,
    load_af3_runtime,
)
from plic.kimi_common import (
    DEFAULT_AUDIOFLAMINGO3_MODEL_ID,
    DEFAULT_STEPAUDIO2_MODEL_ID,
    ensure_cuda_available,
)
from plic.kimi_finetune import (
    DEFAULT_KIMI_FINETUNE_CHECKPOINT_ROOT,
    DEFAULT_KIMI_FINETUNE_DATA_ROOT,
    load_finetune_examples,
    model_stem_for,
    read_json,
    write_json,
)
from plic.stepaudio2_finetune import (
    STEPAUDIO2_FINETUNE_SCOPE_CHOICES,
    build_stepaudio2_inputs,
    collect_stepaudio2_lora_target_modules,
    compute_stepaudio2_text_loss,
    load_stepaudio2_runtime,
)

BACKEND_CHOICES = ("audioflamingo3", "stepaudio2")


def _default_model_id(backend: str) -> str:
    if backend == "audioflamingo3":
        return DEFAULT_AUDIOFLAMINGO3_MODEL_ID
    if backend == "stepaudio2":
        return DEFAULT_STEPAUDIO2_MODEL_ID
    raise ValueError(f"unsupported backend: {backend}")


def _scope_choices(backend: str) -> tuple[str, ...]:
    if backend == "audioflamingo3":
        return AF3_FINETUNE_SCOPE_CHOICES
    if backend == "stepaudio2":
        return STEPAUDIO2_FINETUNE_SCOPE_CHOICES
    raise ValueError(f"unsupported backend: {backend}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune an audio-language backend with LoRA on one VocalGrad category.")
    parser.add_argument("--backend", required=True, choices=BACKEND_CHOICES)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_KIMI_FINETUNE_DATA_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_KIMI_FINETUNE_CHECKPOINT_ROOT)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--run-name", default="default")
    parser.add_argument("--train-category", required=True)
    parser.add_argument("--scope", default="lm_head_only")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-accum-steps", type=int, default=16)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--save-every-steps", type=int, default=200)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _count_labels(examples) -> dict[str, int]:
    return dict(Counter(example.gold_label for example in examples))


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _experiment_dir(args: argparse.Namespace) -> Path:
    return args.checkpoint_root / model_stem_for(args.model_id) / args.run_name / args.scope / args.train_category


def _resume_dir(experiment_dir: Path) -> Path:
    return experiment_dir / "latest"


def _final_adapter_dir(experiment_dir: Path) -> Path:
    return experiment_dir / "adapter"


def _done_marker_path(experiment_dir: Path) -> Path:
    return experiment_dir / "done.train.json"


def _resume_state_path(experiment_dir: Path) -> Path:
    return _resume_dir(experiment_dir) / "train_state.json"


def _optimizer_state_path(experiment_dir: Path) -> Path:
    return _resume_dir(experiment_dir) / "optimizer.pt"


def _resume_adapter_dir(experiment_dir: Path) -> Path:
    return _resume_dir(experiment_dir) / "adapter"


def _prepare_model_and_runtime(args: argparse.Namespace, experiment_dir: Path):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model

    resume_adapter_dir = _resume_adapter_dir(experiment_dir)
    if args.backend == "audioflamingo3":
        if resume_adapter_dir.exists() and _resume_state_path(experiment_dir).exists():
            runtime = load_af3_runtime(args.model_id, adapter_path=resume_adapter_dir, trainable_adapter=True)
            return runtime, runtime.model, True
        runtime = load_af3_runtime(args.model_id)
        target_modules = collect_af3_lora_target_modules(runtime.model, args.scope)
    elif args.backend == "stepaudio2":
        if resume_adapter_dir.exists() and _resume_state_path(experiment_dir).exists():
            runtime = load_stepaudio2_runtime(args.model_id, adapter_path=resume_adapter_dir, trainable_adapter=True)
            return runtime, runtime.llm, True
        runtime = load_stepaudio2_runtime(args.model_id)
        target_modules = collect_stepaudio2_lora_target_modules(runtime.llm, args.scope)
    else:
        raise ValueError(f"unsupported backend: {args.backend}")

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    if args.backend == "audioflamingo3":
        runtime.model = get_peft_model(runtime.model, config)
        model = runtime.model
    else:
        runtime.llm = get_peft_model(runtime.llm, config)
        model = runtime.llm

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if args.backend != "stepaudio2" and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "config"):
        model.config.use_cache = False
    model.train()
    return runtime, model, False


def _encode_example(runtime, args: argparse.Namespace, example):
    if args.backend == "audioflamingo3":
        prompt_inputs = build_af3_inputs(runtime, example.audio_path, example.prompt)
        full_inputs = build_af3_inputs(runtime, example.audio_path, example.prompt, answer=example.gold_label)
        batch = dict(full_inputs)
        batch["prompt_token_count"] = int(prompt_inputs["input_ids"].shape[1])
        return batch
    if args.backend == "stepaudio2":
        prompt_inputs = build_stepaudio2_inputs(runtime, example.audio_path, example.prompt)
        full_inputs = build_stepaudio2_inputs(runtime, example.audio_path, example.prompt, answer=example.gold_label)
        full_inputs["prompt_token_count"] = int(prompt_inputs["input_ids"].shape[1])
        return full_inputs
    raise ValueError(f"unsupported backend: {args.backend}")


def _compute_loss(runtime, args: argparse.Namespace, batch: dict[str, Any]):
    if args.backend == "audioflamingo3":
        return compute_af3_text_loss(runtime, batch)
    if args.backend == "stepaudio2":
        return compute_stepaudio2_text_loss(runtime, batch, int(batch["prompt_token_count"]))
    raise ValueError(f"unsupported backend: {args.backend}")


def _evaluate_validation(runtime, args: argparse.Namespace, examples, max_samples: int | None = None) -> float | None:
    import torch

    if not examples:
        return None
    model = runtime.model if args.backend == "audioflamingo3" else runtime.llm
    losses: list[float] = []
    model.eval()
    with torch.inference_mode():
        for idx, example in enumerate(examples):
            if max_samples is not None and idx >= max_samples:
                break
            loss = _compute_loss(runtime, args, _encode_example(runtime, args, example))
            losses.append(float(loss.detach().cpu().item()))
    model.train()
    return float(sum(losses) / len(losses)) if losses else None


def _progress_postfix(*, loss_value: float | None, optimizer_step: int, best_val_loss: float | None) -> dict[str, str]:
    return {
        "loss": f"{loss_value:.4f}" if isinstance(loss_value, float) and math.isfinite(loss_value) else "n/a",
        "opt_step": str(optimizer_step),
        "best_val": f"{best_val_loss:.4f}" if isinstance(best_val_loss, float) and math.isfinite(best_val_loss) else "n/a",
    }


def _save_resume_checkpoint(*, experiment_dir: Path, model, optimizer, state: dict[str, Any]) -> None:
    import torch

    latest_dir = _resume_dir(experiment_dir)
    latest_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = _resume_adapter_dir(experiment_dir)
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
    model.save_pretrained(adapter_dir)
    torch.save(optimizer.state_dict(), _optimizer_state_path(experiment_dir))
    write_json(_resume_state_path(experiment_dir), state)


def _save_final_adapter(experiment_dir: Path, model) -> None:
    adapter_dir = _final_adapter_dir(experiment_dir)
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
    model.save_pretrained(adapter_dir)


def main() -> None:
    import torch

    load_dotenv()
    args = build_parser().parse_args()
    args.model_id = args.model_id or _default_model_id(args.backend)
    if args.scope not in _scope_choices(args.backend):
        raise SystemExit(f"--scope must be one of {_scope_choices(args.backend)} for backend={args.backend}")
    ensure_cuda_available(f"{args.backend} fine-tuning")
    if args.per_device_batch_size != 1:
        raise SystemExit("per-device-batch-size currently must be 1 for audio LoRA fine-tuning")

    experiment_dir = _experiment_dir(args)
    done_path = _done_marker_path(experiment_dir)
    if done_path.exists() and not args.overwrite:
        print(f"[train] already completed: {done_path}")
        return
    if args.overwrite and experiment_dir.exists():
        shutil.rmtree(experiment_dir)

    train_path = args.data_root / model_stem_for(args.model_id) / args.train_category / "train.jsonl"
    val_path = args.data_root / model_stem_for(args.model_id) / args.train_category / "val.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"training data not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"validation data not found: {val_path}")

    train_examples = load_finetune_examples(train_path)
    val_examples = load_finetune_examples(val_path)
    if args.max_train_samples is not None:
        train_examples = train_examples[: args.max_train_samples]
    if args.max_val_samples is not None:
        val_examples = val_examples[: args.max_val_samples]
    if not train_examples:
        raise SystemExit("no training examples available")

    _set_seed(args.seed)
    runtime, model, resumed = _prepare_model_and_runtime(args, experiment_dir)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    resume_state = {"epoch_index": 0, "sample_offset": 0, "optimizer_step": 0, "best_val_loss": None, "history": []}
    if resumed and _resume_state_path(experiment_dir).exists():
        resume_state = read_json(_resume_state_path(experiment_dir))
        if _optimizer_state_path(experiment_dir).exists():
            optimizer.load_state_dict(torch.load(_optimizer_state_path(experiment_dir), map_location="cpu"))
        print(
            f"[train] resuming backend={args.backend} category={args.train_category} scope={args.scope} "
            f"epoch={resume_state['epoch_index']} sample_offset={resume_state['sample_offset']} "
            f"optimizer_step={resume_state['optimizer_step']}"
        )
    else:
        print(f"[train] starting fresh backend={args.backend} category={args.train_category} scope={args.scope}")

    write_json(
        experiment_dir / "run_manifest.json",
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "backend": args.backend,
            "model_id": args.model_id,
            "model_stem": model_stem_for(args.model_id),
            "run_name": args.run_name,
            "train_category": args.train_category,
            "scope": args.scope,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "grad_accum_steps": args.grad_accum_steps,
            "per_device_batch_size": args.per_device_batch_size,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "seed": args.seed,
            "train_size": len(train_examples),
            "val_size": len(val_examples),
            "train_label_counts": _count_labels(train_examples),
            "val_label_counts": _count_labels(val_examples),
            "resumed": resumed,
        },
    )
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    optimizer.zero_grad(set_to_none=True)
    best_val_loss = resume_state.get("best_val_loss")
    history = list(resume_state.get("history") or [])
    global_optimizer_step = int(resume_state.get("optimizer_step", 0))

    for epoch_index in range(int(resume_state.get("epoch_index", 0)), args.epochs):
        order = list(range(len(train_examples)))
        random.Random(args.seed + epoch_index).shuffle(order)
        sample_offset = int(resume_state.get("sample_offset", 0)) if epoch_index == int(resume_state.get("epoch_index", 0)) else 0
        running_losses: list[float] = []
        accumulation = 0
        processed = sample_offset
        epoch_progress = tqdm(
            total=len(order),
            initial=sample_offset,
            desc=f"train:{args.backend}:{args.train_category}:{args.scope}:epoch{epoch_index + 1}/{args.epochs}",
            unit="sample",
        )
        try:
            for position in range(sample_offset, len(order)):
                batch = _encode_example(runtime, args, train_examples[order[position]])
                loss = _compute_loss(runtime, args, batch)
                loss_value = float(loss.detach().cpu().item())
                running_losses.append(loss_value)
                (loss / args.grad_accum_steps).backward()
                accumulation += 1
                processed = position + 1
                epoch_progress.update(1)

                should_step = accumulation >= args.grad_accum_steps or processed >= len(order)
                if should_step:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [param for param in model.parameters() if param.requires_grad],
                            args.max_grad_norm,
                        )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    accumulation = 0
                    global_optimizer_step += 1
                    resume_state = {
                        "epoch_index": epoch_index,
                        "sample_offset": processed,
                        "optimizer_step": global_optimizer_step,
                        "best_val_loss": best_val_loss,
                        "history": history,
                    }
                    if global_optimizer_step % max(1, args.save_every_steps) == 0:
                        _save_resume_checkpoint(experiment_dir=experiment_dir, model=model, optimizer=optimizer, state=resume_state)
                        print(
                            f"[train] checkpoint backend={args.backend} category={args.train_category} scope={args.scope} "
                            f"epoch={epoch_index + 1}/{args.epochs} samples={processed}/{len(order)} "
                            f"optimizer_step={global_optimizer_step}"
                        )
                epoch_progress.set_postfix(
                    _progress_postfix(loss_value=loss_value, optimizer_step=global_optimizer_step, best_val_loss=best_val_loss)
                )
        finally:
            epoch_progress.close()

        train_loss = float(sum(running_losses) / len(running_losses)) if running_losses else math.nan
        val_loss = _evaluate_validation(runtime, args, val_examples, max_samples=args.max_val_samples)
        if val_loss is not None and (best_val_loss is None or val_loss < best_val_loss):
            best_val_loss = val_loss
        epoch_record = {
            "epoch": epoch_index + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "optimizer_step": global_optimizer_step,
        }
        history.append(epoch_record)
        resume_state = {
            "epoch_index": epoch_index + 1,
            "sample_offset": 0,
            "optimizer_step": global_optimizer_step,
            "best_val_loss": best_val_loss,
            "history": history,
        }
        _save_resume_checkpoint(experiment_dir=experiment_dir, model=model, optimizer=optimizer, state=resume_state)
        write_json(experiment_dir / "train_history.json", {"history": history})
        print(
            f"[train] epoch_done backend={args.backend} category={args.train_category} scope={args.scope} "
            f"epoch={epoch_index + 1}/{args.epochs} train_loss={train_loss:.6f} "
            f"val_loss={val_loss if val_loss is not None else 'n/a'}"
        )

    _save_final_adapter(experiment_dir, model)
    write_json(
        done_path,
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "backend": args.backend,
            "model_id": args.model_id,
            "model_stem": model_stem_for(args.model_id),
            "run_name": args.run_name,
            "train_category": args.train_category,
            "scope": args.scope,
            "epochs": args.epochs,
            "optimizer_step": global_optimizer_step,
            "best_val_loss": best_val_loss,
            "final_adapter_dir": str(_final_adapter_dir(experiment_dir)),
            "history": history,
        },
    )
    print(f"[train] completed backend={args.backend} category={args.train_category} scope={args.scope} adapter={_final_adapter_dir(experiment_dir)}")
    runtime.close()


if __name__ == "__main__":
    main()

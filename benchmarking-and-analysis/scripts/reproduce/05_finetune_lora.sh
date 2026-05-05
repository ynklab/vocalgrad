#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for assignment in "$@"; do
  eval "$assignment"
done

MODEL_STEM="${MODEL_STEM:-kimi-audio}"
TRAIN_CATEGORY="${TRAIN_CATEGORY:?set TRAIN_CATEGORY, for example TRAIN_CATEGORY=volume}"
SCOPE="${SCOPE:-lm_head_only}"
RUN_NAME="${RUN_NAME:-epoch1}"
TRAIN_DATASET_ROOT="${TRAIN_DATASET_ROOT:-datasets/vocalgrad/train}"
TEST_DATASET_ROOT="${TEST_DATASET_ROOT:-datasets/vocalgrad/test}"
DATA_ROOT="${DATA_ROOT:-outputs/data/vocalgrad_finetune}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-outputs/checkpoints/vocalgrad_finetune}"
RAW_EVAL_ROOT="${RAW_EVAL_ROOT:-outputs/raw/vocalgrad_finetune_eval}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-outputs/analysis/vocalgrad_finetune_eval}"
CATEGORIES="${CATEGORIES:-$PAPER_CATEGORIES_DEFAULT}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
EPOCHS="${EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
SEED="${SEED:-1234}"
OVERWRITE="${OVERWRITE:-0}"
SKIP_BUILD_DATA="${SKIP_BUILD_DATA:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
RUN_COMPARE="${RUN_COMPARE:-1}"
ANALYSIS_PYTHON_BIN="${ANALYSIS_PYTHON_BIN:-.venv/bin/python}"

backend="$(resolve_backend_for_model_stem "$MODEL_STEM")"
model_id="$(resolve_model_id_for_model_stem "$MODEL_STEM")"
backend_python="${PYTHON_BIN:-$(resolve_python_for_backend "$backend")}"
read -r -a category_array <<< "$CATEGORIES"

if [[ "$SKIP_BUILD_DATA" != "1" ]]; then
  build_cmd=(
    "$backend_python" scripts/run/build_vocalgrad_finetune_dataset.py
    --dataset-root "$TRAIN_DATASET_ROOT"
    --out-root "$DATA_ROOT"
    --model-id "$model_id"
    --categories "$TRAIN_CATEGORY"
    --seed "$SEED"
  )
  if [[ "$OVERWRITE" == "1" ]]; then
    build_cmd+=(--overwrite)
  fi
  printf '[finetune:build-data] command:'
  printf ' %q' "${build_cmd[@]}"
  printf '\n'
  "${build_cmd[@]}"
fi

if [[ "$SKIP_TRAIN" != "1" ]]; then
  case "$MODEL_STEM" in
    kimi-audio)
      train_script="scripts/run/finetune_kimi_vocalgrad_category.py"
      train_cmd=("$backend_python" "$train_script")
      ;;
    mimo-audio)
      train_script="scripts/run/finetune_mimo_vocalgrad_category.py"
      train_cmd=("$backend_python" "$train_script")
      ;;
    audioflamingo3|step-audio-2-mini)
      train_script="scripts/run/finetune_audio_lora_vocalgrad_category.py"
      train_cmd=("$backend_python" "$train_script" --backend "$backend")
      ;;
    *)
      echo "unsupported MODEL_STEM: $MODEL_STEM" >&2
      exit 1
      ;;
  esac

  train_cmd+=(
    --data-root "$DATA_ROOT"
    --checkpoint-root "$CHECKPOINT_ROOT"
    --model-id "$model_id"
    --run-name "$RUN_NAME"
    --train-category "$TRAIN_CATEGORY"
    --scope "$SCOPE"
    --epochs "$EPOCHS"
    --learning-rate "$LEARNING_RATE"
    --weight-decay "$WEIGHT_DECAY"
    --grad-accum-steps "$GRAD_ACCUM_STEPS"
    --per-device-batch-size "$PER_DEVICE_BATCH_SIZE"
    --seed "$SEED"
    --lora-r "$LORA_R"
    --lora-alpha "$LORA_ALPHA"
    --lora-dropout "$LORA_DROPOUT"
  )
  if [[ -n "$MAX_TRAIN_SAMPLES" ]]; then
    train_cmd+=(--max-train-samples "$MAX_TRAIN_SAMPLES")
  fi
  if [[ -n "$MAX_VAL_SAMPLES" ]]; then
    train_cmd+=(--max-val-samples "$MAX_VAL_SAMPLES")
  fi
  if [[ "$OVERWRITE" == "1" ]]; then
    train_cmd+=(--overwrite)
  fi

  printf '[finetune:train] command:'
  printf ' %q' "${train_cmd[@]}"
  printf '\n'
  "${train_cmd[@]}"
fi

if [[ "$SKIP_EVAL" != "1" ]]; then
  case "$MODEL_STEM" in
    kimi-audio)
      eval_script="scripts/run/evaluate_kimi_finetuned_vocalgrad.py"
      eval_cmd=("$backend_python" "$eval_script")
      ;;
    mimo-audio)
      eval_script="scripts/run/evaluate_mimo_finetuned_vocalgrad.py"
      eval_cmd=("$backend_python" "$eval_script")
      ;;
    audioflamingo3|step-audio-2-mini)
      eval_script="scripts/run/evaluate_audio_lora_finetuned_vocalgrad.py"
      eval_cmd=("$backend_python" "$eval_script" --backend "$backend")
      ;;
  esac

  eval_cmd+=(
    --dataset-root "$TEST_DATASET_ROOT"
    --checkpoint-root "$CHECKPOINT_ROOT"
    --raw-root "$RAW_EVAL_ROOT"
    --analysis-root "$ANALYSIS_ROOT"
    --model-id "$model_id"
    --run-name "$RUN_NAME"
    --train-category "$TRAIN_CATEGORY"
    --scope "$SCOPE"
    --categories "${category_array[@]}"
  )
  if [[ -n "$MAX_EVAL_SAMPLES" ]]; then
    eval_cmd+=(--max-samples "$MAX_EVAL_SAMPLES")
  fi
  if [[ "$OVERWRITE" == "1" ]]; then
    eval_cmd+=(--overwrite)
  fi

  printf '[finetune:evaluate] command:'
  printf ' %q' "${eval_cmd[@]}"
  printf '\n'
  "${eval_cmd[@]}"
fi

if [[ "$RUN_COMPARE" == "1" ]]; then
  "$ANALYSIS_PYTHON_BIN" scripts/analysis/compare_vocalgrad_finetune_generalization.py \
    --root "$ANALYSIS_ROOT" \
    --run-name "$RUN_NAME" \
    --model-id "$model_id" \
    --out-dir "$ANALYSIS_ROOT/_comparisons/$RUN_NAME"

  "$ANALYSIS_PYTHON_BIN" scripts/analysis/plot_vocalgrad_finetune_generalization_heatmaps.py \
    --root "$ANALYSIS_ROOT" \
    --run-name "$RUN_NAME" \
    --comparison-dir "$ANALYSIS_ROOT/_comparisons/$RUN_NAME"
fi

#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for assignment in "$@"; do
  eval "$assignment"
done

TRAIN_CATEGORY="${TRAIN_CATEGORY:?set TRAIN_CATEGORY, for example TRAIN_CATEGORY=volume}"
SCOPE="${SCOPE:-all_linear}"
RUN_NAME="${RUN_NAME:-epoch1}"
MODEL_STEM="${MODEL_STEM:-kimi-audio}"
BACKEND="${BACKEND:-kimia}"
MODEL_ID="${MODEL_ID:-$(resolve_model_id_for_model_stem "$MODEL_STEM")}"
PYTHON_BIN="${PYTHON_BIN:-$(resolve_python_for_backend "$BACKEND")}"
ANALYSIS_PYTHON_BIN="${ANALYSIS_PYTHON_BIN:-.venv/bin/python}"
TRAIN_DATASET_ROOT="${TRAIN_DATASET_ROOT:-datasets/vocalgrad/train}"
TEST_DATASET_ROOT="${TEST_DATASET_ROOT:-datasets/vocalgrad/test}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-outputs/checkpoints/vocalgrad_finetune}"
ADAPTER_PATH="${ADAPTER_PATH:-$CHECKPOINT_ROOT/$MODEL_STEM/$RUN_NAME/$SCOPE/$TRAIN_CATEGORY/adapter}"
OUT_MODEL_STEM="${OUT_MODEL_STEM:-kimi-audio-ft-$TRAIN_CATEGORY}"
CATEGORIES="${CATEGORIES:-$PAPER_CATEGORIES_DEFAULT}"
MAX_SAMPLES_PER_CATEGORY="${MAX_SAMPLES_PER_CATEGORY:-}"
OVERWRITE="${OVERWRITE:-0}"

LINEAR_FEATURE_ROOT="${LINEAR_FEATURE_ROOT:-outputs/features/linear_probe_finetuned_kimi}"
LINEAR_ANALYSIS_ROOT="${LINEAR_ANALYSIS_ROOT:-outputs/analysis/linear_probe_finetuned_kimi}"
LAYER_FEATURE_ROOT="${LAYER_FEATURE_ROOT:-outputs/features/linear_probe_lm_text_layers_finetuned_kimi}"
LAYER_ANALYSIS_ROOT="${LAYER_ANALYSIS_ROOT:-outputs/analysis/linear_probe_lm_text_layers_finetuned_kimi}"
SUMMARY_ROOT="${SUMMARY_ROOT:-outputs/analysis/kimi_finetuned_probe_completed}"

read -r -a category_array <<< "$CATEGORIES"

if [[ ! -d "$ADAPTER_PATH" ]]; then
  echo "adapter not found: $ADAPTER_PATH" >&2
  echo "Run scripts/reproduce/05_finetune_lora.sh first, or set ADAPTER_PATH." >&2
  exit 1
fi

for split in train test; do
  if [[ "$split" == "train" ]]; then
    dataset_root="$TRAIN_DATASET_ROOT"
  else
    dataset_root="$TEST_DATASET_ROOT"
  fi

  linear_extract_cmd=(
    "$PYTHON_BIN" scripts/run/extract_vocalgrad_linear_probe_features.py
    --backend "$BACKEND"
    --model-id "$MODEL_ID"
    --adapter-path "$ADAPTER_PATH"
    --model-stem-override "$OUT_MODEL_STEM"
    --dataset-root "$dataset_root"
    --split-name "$split"
    --out-root "$LINEAR_FEATURE_ROOT"
    --categories "${category_array[@]}"
  )
  layer_extract_cmd=(
    "$PYTHON_BIN" scripts/run/extract_vocalgrad_lm_text_layer_features.py
    --backend "$BACKEND"
    --model-id "$MODEL_ID"
    --adapter-path "$ADAPTER_PATH"
    --model-stem-override "$OUT_MODEL_STEM"
    --dataset-root "$dataset_root"
    --split-name "$split"
    --out-root "$LAYER_FEATURE_ROOT"
    --categories "${category_array[@]}"
  )
  if [[ -n "$MAX_SAMPLES_PER_CATEGORY" ]]; then
    linear_extract_cmd+=(--max-samples-per-category "$MAX_SAMPLES_PER_CATEGORY")
    layer_extract_cmd+=(--max-samples-per-category "$MAX_SAMPLES_PER_CATEGORY")
  fi
  if [[ "$OVERWRITE" == "1" ]]; then
    linear_extract_cmd+=(--overwrite)
    layer_extract_cmd+=(--overwrite)
  else
    linear_extract_cmd+=(--skip-existing)
    layer_extract_cmd+=(--skip-existing)
  fi

  printf '[finetuned-kimi-probe:linear-extract] command:'
  printf ' %q' "${linear_extract_cmd[@]}"
  printf '\n'
  "${linear_extract_cmd[@]}"

  printf '[finetuned-kimi-probe:layer-extract] command:'
  printf ' %q' "${layer_extract_cmd[@]}"
  printf '\n'
  "${layer_extract_cmd[@]}"
done

"$ANALYSIS_PYTHON_BIN" scripts/run/train_vocalgrad_linear_probe.py \
  --train-root "$LINEAR_FEATURE_ROOT/train" \
  --test-root "$LINEAR_FEATURE_ROOT/test" \
  --model-stem "$OUT_MODEL_STEM" \
  --analysis-root "$LINEAR_ANALYSIS_ROOT" \
  --run-name "$RUN_NAME" \
  --categories "${category_array[@]}"

"$ANALYSIS_PYTHON_BIN" scripts/run/train_vocalgrad_lm_text_layer_probe.py \
  --train-root "$LAYER_FEATURE_ROOT/train" \
  --test-root "$LAYER_FEATURE_ROOT/test" \
  --model-stem "$OUT_MODEL_STEM" \
  --analysis-root "$LAYER_ANALYSIS_ROOT" \
  --run-name "$RUN_NAME" \
  --categories "${category_array[@]}"

"$ANALYSIS_PYTHON_BIN" scripts/analysis/compare_linear_probe_results.py \
  --root "$LINEAR_ANALYSIS_ROOT" \
  --run-name "$RUN_NAME" \
  --out-dir "$LINEAR_ANALYSIS_ROOT/_comparisons/$RUN_NAME"

"$ANALYSIS_PYTHON_BIN" scripts/analysis/plot_linear_probe_heatmaps.py \
  --root "$LINEAR_ANALYSIS_ROOT" \
  --run-name "$RUN_NAME" \
  --out-dir "$SUMMARY_ROOT/$RUN_NAME/linprobe_heatmaps"

"$ANALYSIS_PYTHON_BIN" scripts/analysis/make_kimi_finetuned_probe_completed_summary.py \
  --run-name "$RUN_NAME" \
  --linear-root "$LINEAR_ANALYSIS_ROOT" \
  --lm-text-root "$LAYER_ANALYSIS_ROOT" \
  --out-root "$SUMMARY_ROOT"

"$ANALYSIS_PYTHON_BIN" scripts/analysis/plot_selected_kimi_finetuned_lm_text_panels.py \
  --root "$LAYER_ANALYSIS_ROOT" \
  --run-name "$RUN_NAME" \
  --out-dir "$SUMMARY_ROOT/$RUN_NAME/lm_text_layer_lines"

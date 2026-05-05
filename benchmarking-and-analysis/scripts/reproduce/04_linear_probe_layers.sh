#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for assignment in "$@"; do
  eval "$assignment"
done

TRAIN_DATASET_ROOT="${TRAIN_DATASET_ROOT:-datasets/vocalgrad/train}"
TEST_DATASET_ROOT="${TEST_DATASET_ROOT:-datasets/vocalgrad/test}"
FEATURE_ROOT="${FEATURE_ROOT:-outputs/features/linear_probe_lm_text_layers}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-outputs/analysis/linear_probe_lm_text_layers}"
RUN_NAME="${RUN_NAME:-default}"
MODEL_STEMS="${MODEL_STEMS:-kimi-audio mimo-audio step-audio-2-mini audioflamingo3}"
CATEGORIES="${CATEGORIES:-$PAPER_CATEGORIES_DEFAULT}"
MAX_SAMPLES_PER_CATEGORY="${MAX_SAMPLES_PER_CATEGORY:-}"
OVERWRITE="${OVERWRITE:-0}"
ANALYSIS_PYTHON_BIN="${ANALYSIS_PYTHON_BIN:-.venv/bin/python}"

read -r -a model_stem_array <<< "$MODEL_STEMS"
read -r -a category_array <<< "$CATEGORIES"

for model_stem in "${model_stem_array[@]}"; do
  backend="$(resolve_backend_for_model_stem "$model_stem")"
  model_id="$(resolve_model_id_for_model_stem "$model_stem")"
  extract_python="$(resolve_python_for_backend "$backend")"

  for split in train test; do
    if [[ "$split" == "train" ]]; then
      dataset_root="$TRAIN_DATASET_ROOT"
    else
      dataset_root="$TEST_DATASET_ROOT"
    fi

    extract_cmd=(
      "$extract_python" scripts/run/extract_vocalgrad_lm_text_layer_features.py
      --backend "$backend"
      --model-id "$model_id"
      --dataset-root "$dataset_root"
      --split-name "$split"
      --out-root "$FEATURE_ROOT"
      --categories "${category_array[@]}"
    )
    if [[ -n "$MAX_SAMPLES_PER_CATEGORY" ]]; then
      extract_cmd+=(--max-samples-per-category "$MAX_SAMPLES_PER_CATEGORY")
    fi
    if [[ "$OVERWRITE" == "1" ]]; then
      extract_cmd+=(--overwrite)
    fi

    printf '[linear-probe-layers:extract] command:'
    printf ' %q' "${extract_cmd[@]}"
    printf '\n'
    "${extract_cmd[@]}"
  done

  train_cmd=(
    "$ANALYSIS_PYTHON_BIN" scripts/run/train_vocalgrad_lm_text_layer_probe.py
    --train-root "$FEATURE_ROOT/train"
    --test-root "$FEATURE_ROOT/test"
    --model-stem "$model_stem"
    --analysis-root "$ANALYSIS_ROOT"
    --run-name "$RUN_NAME"
    --categories "${category_array[@]}"
  )
  printf '[linear-probe-layers:train] command:'
  printf ' %q' "${train_cmd[@]}"
  printf '\n'
  "${train_cmd[@]}"
done

"$ANALYSIS_PYTHON_BIN" scripts/analysis/compare_lm_text_layer_probe_results.py \
  --root "$ANALYSIS_ROOT" \
  --run-name "$RUN_NAME" \
  --out-dir "$ANALYSIS_ROOT/_comparisons/$RUN_NAME"

"$ANALYSIS_PYTHON_BIN" scripts/analysis/plot_lm_text_layer_probe_lines.py \
  --root "$ANALYSIS_ROOT" \
  --run-name "$RUN_NAME" \
  --out-dir "$ANALYSIS_ROOT/_comparisons/$RUN_NAME/layer_lines"

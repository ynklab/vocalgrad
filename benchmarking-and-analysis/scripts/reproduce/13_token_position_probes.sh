#!/usr/bin/env bash

# Probe Kimi's decoder representations at audio-segment and text-token positions.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../reproduce/common.sh"

for assignment in "$@"; do eval "$assignment"; done

TRAIN_DATASET_ROOT="${TRAIN_DATASET_ROOT:-datasets/vocalgrad/train}"
TEST_DATASET_ROOT="${TEST_DATASET_ROOT:-datasets/vocalgrad/test}"
FEATURE_ROOT="${FEATURE_ROOT:-outputs/features/rebuttal/text_prediction_representations_sparse4}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-outputs/analysis/rebuttal/text_prediction_representations_sparse4}"
RUN_NAME="${RUN_NAME:-default}"
CATEGORIES="${CATEGORIES:-$PAPER_CATEGORIES_DEFAULT}"
MAX_SAMPLES_PER_CATEGORY="${MAX_SAMPLES_PER_CATEGORY:-}"
OVERWRITE="${OVERWRITE:-0}"
INPUT_ORDERS="${INPUT_ORDERS:-query_to_audio audio_to_query}"
SKIP_EXTRACTION="${SKIP_EXTRACTION:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_PLOT="${SKIP_PLOT:-0}"
ANALYZE_ONLY="${ANALYZE_ONLY:-0}"
KIMI_PYTHON_BIN="${KIMI_PYTHON_BIN:-.venv-kimi/bin/python}"
ANALYSIS_PYTHON_BIN="${ANALYSIS_PYTHON_BIN:-.venv/bin/python}"

read -r -a category_array <<< "$CATEGORIES"
read -r -a input_order_array <<< "$INPUT_ORDERS"

if [[ "$ANALYZE_ONLY" == 1 ]]; then
  for input_order in query_to_audio audio_to_query; do
    "$ANALYSIS_PYTHON_BIN" scripts/run/train_vocalgrad_kimi_text_prediction_probe.py \
      --train-root "$FEATURE_ROOT/$input_order/train" --test-root "$FEATURE_ROOT/$input_order/test" \
      --model-stem kimi-audio --analysis-root "$ANALYSIS_ROOT" --input-order "$input_order" \
      --run-name "$RUN_NAME" --rebuild-summary-only
  done
  "$ANALYSIS_PYTHON_BIN" scripts/analysis/make_kimi_token_position_tables.py \
    --root "$ANALYSIS_ROOT" --run-name "$RUN_NAME" \
    --out-dir "$ANALYSIS_ROOT/_comparisons/$RUN_NAME"
  exit 0
fi

for input_order in "${input_order_array[@]}"; do
  if [[ "$SKIP_EXTRACTION" != 1 ]]; then
    for split in train test; do
      dataset_root="$TRAIN_DATASET_ROOT"
      [[ "$split" == test ]] && dataset_root="$TEST_DATASET_ROOT"
      cmd=("$KIMI_PYTHON_BIN" scripts/run/extract_vocalgrad_kimi_text_prediction_probe_features.py
        --dataset-root "$dataset_root" --split-name "$split" --input-order "$input_order"
        --out-root "$FEATURE_ROOT" --categories "${category_array[@]}")
      [[ -n "$MAX_SAMPLES_PER_CATEGORY" ]] && cmd+=(--max-samples-per-category "$MAX_SAMPLES_PER_CATEGORY")
      [[ "$OVERWRITE" == 1 ]] && cmd+=(--overwrite)
      printf '[rebuttal:extract] '; printf '%q ' "${cmd[@]}"; printf '\n'
      "${cmd[@]}"
    done
  fi
  if [[ "$SKIP_TRAIN" != 1 ]]; then
    train_cmd=("$ANALYSIS_PYTHON_BIN" scripts/run/train_vocalgrad_kimi_text_prediction_probe.py
      --train-root "$FEATURE_ROOT/$input_order/train" --test-root "$FEATURE_ROOT/$input_order/test"
      --model-stem kimi-audio --analysis-root "$ANALYSIS_ROOT" --input-order "$input_order"
      --run-name "$RUN_NAME" --categories "${category_array[@]}")
    printf '[rebuttal:train] '; printf '%q ' "${train_cmd[@]}"; printf '\n'
    "${train_cmd[@]}"
  fi
done

if [[ "$SKIP_PLOT" != 1 ]]; then
  "$ANALYSIS_PYTHON_BIN" scripts/analysis/make_kimi_token_position_tables.py \
    --root "$ANALYSIS_ROOT" --run-name "$RUN_NAME" \
    --out-dir "$ANALYSIS_ROOT/_comparisons/$RUN_NAME"
fi

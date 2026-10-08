#!/usr/bin/env bash

# Ternary question ablation: augmented clips are increase/decrease; source clips are constant.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../reproduce/common.sh"

for assignment in "$@"; do
  eval "$assignment"
done

BACKEND="kimia"
VARIANT="${VARIANT:-default}"
DATASET_ROOT="${DATASET_ROOT:-datasets/vocalgrad/test}"
SOURCE_ROOT="${SOURCE_ROOT:-datasets/vocalgrad/test/source_clips}"
RAW_ROOT="${RAW_ROOT:-outputs/raw/rebuttal/ternary_question/$VARIANT}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-outputs/analysis/rebuttal/ternary_question/$VARIANT}"
CATEGORIES="${CATEGORIES:-$PAPER_CATEGORIES_DEFAULT}"
PYTHON_BIN="${PYTHON_BIN:-$(resolve_python_for_backend "$BACKEND")}"
MAX_AUGMENTED_SAMPLES="${MAX_AUGMENTED_SAMPLES:-}"
MAX_SOURCE_CLIPS="${MAX_SOURCE_CLIPS:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
RETRY_MAX_NEW_TOKENS="${RETRY_MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-}"
TOP_K="${TOP_K:-5}"
CONCURRENCY="${CONCURRENCY:-8}"
OUT_STEM="${OUT_STEM:-}"

read -r -a category_array <<< "$CATEGORIES"
cmd=("$PYTHON_BIN" -m plic.cli_vocalgrad_ternary_all --backend "$BACKEND" --dataset-root "$DATASET_ROOT" --source-root "$SOURCE_ROOT" --raw-root "$RAW_ROOT" --analysis-root "$ANALYSIS_ROOT" --categories "${category_array[@]}" --max-new-tokens "$MAX_NEW_TOKENS" --retry-max-new-tokens "$RETRY_MAX_NEW_TOKENS" --top-k "$TOP_K" --concurrency "$CONCURRENCY")
[[ -n "$MAX_AUGMENTED_SAMPLES" ]] && cmd+=(--max-augmented-samples "$MAX_AUGMENTED_SAMPLES")
[[ -n "$MAX_SOURCE_CLIPS" ]] && cmd+=(--max-source-clips "$MAX_SOURCE_CLIPS")
[[ -n "$TEMPERATURE" ]] && cmd+=(--temperature "$TEMPERATURE")
[[ -n "$OUT_STEM" ]] && cmd+=(--out-stem "$OUT_STEM")

printf '[rebuttal:ternary-question] command:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

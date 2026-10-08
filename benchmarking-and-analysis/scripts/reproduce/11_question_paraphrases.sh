#!/usr/bin/env bash

# Evaluate the three reviewer-facing question paraphrases for each supported category.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../reproduce/common.sh"
for assignment in "$@"; do eval "$assignment"; done

BACKEND="kimia"
DATASET_ROOT="${DATASET_ROOT:-datasets/vocalgrad/test}"
RAW_ROOT="${RAW_ROOT:-outputs/raw/rebuttal/paraphrase_questions/default}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-outputs/analysis/rebuttal/paraphrase_questions/default}"
CATEGORIES="${CATEGORIES:-volume voice_pitch speaking_speed}"
PYTHON_BIN="${PYTHON_BIN:-$(resolve_python_for_backend "$BACKEND")}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
RETRY_MAX_NEW_TOKENS="${RETRY_MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-}"
TOP_K="${TOP_K:-5}"
CONCURRENCY="${CONCURRENCY:-8}"
OUT_STEM="${OUT_STEM:-}"
read -r -a category_array <<< "$CATEGORIES"
cmd=("$PYTHON_BIN" -m plic.cli_vocalgrad_paraphrase_all --backend "$BACKEND" --dataset-root "$DATASET_ROOT" --raw-root "$RAW_ROOT" --analysis-root "$ANALYSIS_ROOT" --categories "${category_array[@]}" --max-new-tokens "$MAX_NEW_TOKENS" --retry-max-new-tokens "$RETRY_MAX_NEW_TOKENS" --top-k "$TOP_K" --concurrency "$CONCURRENCY")
[[ -n "$MAX_SAMPLES" ]] && cmd+=(--max-samples "$MAX_SAMPLES")
[[ -n "$TEMPERATURE" ]] && cmd+=(--temperature "$TEMPERATURE")
[[ -n "$OUT_STEM" ]] && cmd+=(--out-stem "$OUT_STEM")
printf '[rebuttal:paraphrase-questions] command:'; printf ' %q' "${cmd[@]}"; printf '\n'
"${cmd[@]}"

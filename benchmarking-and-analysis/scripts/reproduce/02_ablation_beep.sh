#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for assignment in "$@"; do
  eval "$assignment"
done

BACKEND="${BACKEND:-kimia}"
DATASET_ROOT="${DATASET_ROOT:-datasets/vocalgrad/ablation_beep}"
RAW_ROOT="${RAW_ROOT:-outputs/raw/vocalgrad_ablation_beep/default}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-outputs/analysis/vocalgrad_ablation_beep/default}"
CATEGORIES="${CATEGORIES:-$BEEP_CATEGORIES_DEFAULT}"
PYTHON_BIN="${PYTHON_BIN:-$(resolve_python_for_backend "$BACKEND")}"
ANALYSIS_PYTHON_BIN="${ANALYSIS_PYTHON_BIN:-.venv/bin/python}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
RETRY_MAX_NEW_TOKENS="${RETRY_MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-}"
TOP_K="${TOP_K:-5}"
CONCURRENCY="${CONCURRENCY:-8}"
OUT_STEM="${OUT_STEM:-}"
RUN_TABLES="${RUN_TABLES:-1}"

read -r -a category_array <<< "$CATEGORIES"

cmd=(
  "$PYTHON_BIN" -m plic.cli_vocalgrad_beep_all
  --backend "$BACKEND"
  --dataset-root "$DATASET_ROOT"
  --raw-root "$RAW_ROOT"
  --analysis-root "$ANALYSIS_ROOT"
  --categories "${category_array[@]}"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --retry-max-new-tokens "$RETRY_MAX_NEW_TOKENS"
  --top-k "$TOP_K"
  --concurrency "$CONCURRENCY"
)

if [[ -n "$MAX_SAMPLES" ]]; then
  cmd+=(--max-samples "$MAX_SAMPLES")
fi
if [[ -n "$TEMPERATURE" ]]; then
  cmd+=(--temperature "$TEMPERATURE")
fi
if [[ -n "$OUT_STEM" ]]; then
  cmd+=(--out-stem "$OUT_STEM")
fi

printf '[ablation-beep] command:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

if [[ "$RUN_TABLES" == "1" ]]; then
  "$ANALYSIS_PYTHON_BIN" scripts/analysis/make_ablation_beep_accuracy_table.py \
    --analysis-root "$ANALYSIS_ROOT" \
    --raw-root "$RAW_ROOT" \
    --out-tex "$ANALYSIS_ROOT/ablation_beep_accuracy_comparison.tex" \
    --out-csv "$ANALYSIS_ROOT/ablation_beep_accuracy_comparison.csv"
fi

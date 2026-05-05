#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for assignment in "$@"; do
  eval "$assignment"
done

BACKEND="${BACKEND:-kimia}"
VARIANT="${VARIANT:-default}"
DATASET_ROOT="${DATASET_ROOT:-datasets/vocalgrad/test}"
RAW_ROOT="${RAW_ROOT:-outputs/raw/vocalgrad/$VARIANT}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-outputs/analysis/vocalgrad/$VARIANT}"
CATEGORIES="${CATEGORIES:-$PAPER_CATEGORIES_DEFAULT}"
PYTHON_BIN="${PYTHON_BIN:-$(resolve_python_for_backend "$BACKEND")}"
ANALYSIS_PYTHON_BIN="${ANALYSIS_PYTHON_BIN:-.venv/bin/python}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
RETRY_MAX_NEW_TOKENS="${RETRY_MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-}"
TOP_K="${TOP_K:-5}"
CONCURRENCY="${CONCURRENCY:-8}"
OUT_STEM="${OUT_STEM:-}"
RUN_TABLES="${RUN_TABLES:-}"
if [[ -z "$RUN_TABLES" ]]; then
  if [[ "$VARIANT" == "default" ]]; then
    RUN_TABLES=1
  else
    RUN_TABLES=0
  fi
fi

read -r -a category_array <<< "$CATEGORIES"
read -r -a prompt_flags <<< "$(variant_flags "$VARIANT")"

cmd=(
  "$PYTHON_BIN" -m plic.cli_vocalgrad_all
  --backend "$BACKEND"
  --dataset-root "$DATASET_ROOT"
  --raw-root "$RAW_ROOT"
  --analysis-root "$ANALYSIS_ROOT"
  --categories "${category_array[@]}"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --retry-max-new-tokens "$RETRY_MAX_NEW_TOKENS"
  --top-k "$TOP_K"
  --concurrency "$CONCURRENCY"
  "${prompt_flags[@]}"
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

printf '[benchmark] command:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

if [[ "$RUN_TABLES" == "1" ]]; then
  "$ANALYSIS_PYTHON_BIN" scripts/analysis/make_human_model_accuracy_table.py \
    --model-analysis-root "$ANALYSIS_ROOT" \
    --model-raw-root "$RAW_ROOT" \
    --out-tex "$ANALYSIS_ROOT/human_model_accuracy_comparison.tex" \
    --out-csv "$ANALYSIS_ROOT/human_model_accuracy_comparison.csv"

  "$ANALYSIS_PYTHON_BIN" scripts/analysis/make_vocalgrad_accuracy_bootstrap_ci.py \
    --raw-root "$RAW_ROOT" \
    --out-csv "$ANALYSIS_ROOT/model_accuracy_bootstrap_ci.csv" \
    --out-tex "$ANALYSIS_ROOT/model_accuracy_bootstrap_ci.tex"

  "$ANALYSIS_PYTHON_BIN" scripts/analysis/make_vocalgrad_parsing_failure_tables.py \
    --raw-root "$RAW_ROOT" \
    --out-root "$ANALYSIS_ROOT"
fi

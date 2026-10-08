#!/usr/bin/env bash

# Benchmark one source-dataset variant with the standard VocalGrad task.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../reproduce/common.sh"
for assignment in "$@"; do eval "$assignment"; done

DATASET_VARIANT="${DATASET_VARIANT:?set DATASET_VARIANT to commonvoice_spontaneous or loquaciousset}"
case "$DATASET_VARIANT" in commonvoice_spontaneous|loquaciousset) ;; *) echo "unsupported DATASET_VARIANT=$DATASET_VARIANT" >&2; exit 2;; esac
BACKEND="kimia"
DATASET_ROOT="${DATASET_ROOT:-datasets/$DATASET_VARIANT/test}"
RAW_ROOT="${RAW_ROOT:-outputs/raw/rebuttal/dataset_variants/$DATASET_VARIANT/benchmark}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-outputs/analysis/rebuttal/dataset_variants/$DATASET_VARIANT/benchmark}"
CATEGORIES="${CATEGORIES:-$PAPER_CATEGORIES_DEFAULT}"
RUN_TABLES="${RUN_TABLES:-0}"

cmd=(env "BACKEND=$BACKEND" "DATASET_ROOT=$DATASET_ROOT" "RAW_ROOT=$RAW_ROOT" "ANALYSIS_ROOT=$ANALYSIS_ROOT"
  "CATEGORIES=$CATEGORIES" "RUN_TABLES=$RUN_TABLES")
[[ -n "${MAX_SAMPLES:-}" ]] && cmd+=("MAX_SAMPLES=$MAX_SAMPLES")
[[ -n "${MAX_NEW_TOKENS:-}" ]] && cmd+=("MAX_NEW_TOKENS=$MAX_NEW_TOKENS")
[[ -n "${TEMPERATURE:-}" ]] && cmd+=("TEMPERATURE=$TEMPERATURE")
cmd+=(bash scripts/reproduce/01_benchmark.sh)
printf '[rebuttal:variant-benchmark] '; printf '%q ' "${cmd[@]}"; printf '\n'
"${cmd[@]}"

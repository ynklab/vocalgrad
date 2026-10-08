#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
KIMI_PYTHON_BIN="${KIMI_PYTHON_BIN:-.venv-kimi/bin/python}"
ANALYSIS_PYTHON_BIN="${ANALYSIS_PYTHON_BIN:-.venv/bin/python}"
OUT_ROOT="${OUT_ROOT:-outputs/analysis/rebuttal/category_direction_alignment}"
SKIP_EXTRACTION="${SKIP_EXTRACTION:-0}"
if [[ "$SKIP_EXTRACTION" != 1 ]]; then
  for category in volume voice_pitch; do
    "$KIMI_PYTHON_BIN" scripts/run/extract_vocalgrad_lm_text_layer_features.py \
      --backend kimia --dataset-root datasets/vocalgrad/test --split-name test \
      --adapter-path "outputs/checkpoints/vocalgrad_finetune/kimi-audio/epoch1/all_linear/$category/adapter" \
      --model-stem "kimi-audio-ft-$category" --out-root outputs/features/linear_probe_lm_text_layers_finetuned_kimi \
      --categories $PAPER_CATEGORIES_DEFAULT
  done
  "$KIMI_PYTHON_BIN" scripts/run/extract_vocalgrad_lm_text_layer_features.py \
    --backend kimia --dataset-root datasets/vocalgrad/test --split-name test \
    --out-root outputs/features/linear_probe_lm_text_layers --categories $PAPER_CATEGORIES_DEFAULT
fi
"$ANALYSIS_PYTHON_BIN" scripts/analysis/analyze_kimi_category_direction_alignment.py --repo "$repo_root" --out "$OUT_ROOT"
"$ANALYSIS_PYTHON_BIN" scripts/analysis/summarize_kimi_category_direction_alignment.py --root "$OUT_ROOT"
"$ANALYSIS_PYTHON_BIN" scripts/analysis/plot_kimi_category_direction_alignment.py --root "$OUT_ROOT"

#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
KIMI_PYTHON_BIN="${KIMI_PYTHON_BIN:-.venv-kimi/bin/python}"
ANALYSIS_PYTHON_BIN="${ANALYSIS_PYTHON_BIN:-.venv/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-datasets/meld}"
RAW_ROOT="${RAW_ROOT:-outputs/raw/rebuttal/meld_mcq}"
OUT_ROOT="${OUT_ROOT:-outputs/analysis/rebuttal/meld_mcq/finetune_prediction_changes}"
ANALYZE_ONLY="${ANALYZE_ONLY:-0}"
if [[ "$ANALYZE_ONLY" != 1 ]]; then
 for condition in base volume voice_pitch; do
  stem=kimi-audio
  args=()
  if [[ "$condition" != base ]]; then
   stem="kimi-audio-ft-$condition"
   args+=(--adapter-path "outputs/checkpoints/vocalgrad_finetune/kimi-audio/epoch1/all_linear/$condition/adapter")
  fi
  [[ -n "${MAX_SAMPLES:-}" ]] && args+=(--max-samples "$MAX_SAMPLES")
  "$KIMI_PYTHON_BIN" scripts/run/evaluate_meld_mcq.py --backend kimia --task emotion \
    --dataset-root "$DATASET_ROOT" --out "$RAW_ROOT/emotion/$stem.jsonl" "${args[@]}"
 done
fi
"$ANALYSIS_PYTHON_BIN" scripts/analysis/analyze_meld_emotion_transfer.py --raw-root "$RAW_ROOT" --out-root "$OUT_ROOT"
"$ANALYSIS_PYTHON_BIN" scripts/analysis/make_meld_emotion_logit_letter_prf_table.py --input "$OUT_ROOT/per_class_f1.csv" --output "$OUT_ROOT/emotion_logit_letter_precision_recall_f1.md"

#!/usr/bin/env bash

set -euo pipefail
# Aggregate predictions with 16_aggregate_results.sh before generating tables.
# Point BENCHMARK_ANALYSIS_ROOT, PROMPT_ROOT, CROSS_ATTRIBUTE_ROOT and
# FEWSHOT_ROOT below at that tree.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for assignment in "$@"; do
  eval "$assignment"
done

ANALYSIS_PYTHON_BIN="${ANALYSIS_PYTHON_BIN:-.venv/bin/python}"
BENCHMARK_RAW_ROOT="${BENCHMARK_RAW_ROOT:-outputs/raw/vocalgrad/default}"
BENCHMARK_ANALYSIS_ROOT="${BENCHMARK_ANALYSIS_ROOT:-outputs/analysis/vocalgrad/default}"
SOURCE_BIAS_RAW_ROOT="${SOURCE_BIAS_RAW_ROOT:-outputs/raw/source_clips_direction_generation}"
SOURCE_BIAS_ANALYSIS_ROOT="${SOURCE_BIAS_ANALYSIS_ROOT:-outputs/analysis/source_clips_direction_generation}"
PROMPT_ROOT="${PROMPT_ROOT:-outputs/analysis/vocalgrad}"
CROSS_ATTRIBUTE_ROOT="${CROSS_ATTRIBUTE_ROOT:-outputs/analysis/vocalgrad_cross_attribute}"
FEWSHOT_ROOT="${FEWSHOT_ROOT:-outputs/analysis/vocalgrad_fewshot}"
FEWSHOT_RUN_NAME="${FEWSHOT_RUN_NAME:-fewshot_2x2_seed1234_audio-ref}"

"$ANALYSIS_PYTHON_BIN" scripts/analysis/make_human_model_accuracy_table.py \
  --model-analysis-root "$BENCHMARK_ANALYSIS_ROOT" \
  --model-raw-root "$BENCHMARK_RAW_ROOT" \
  --out-tex "$BENCHMARK_ANALYSIS_ROOT/human_model_accuracy_comparison.tex" \
  --out-csv "$BENCHMARK_ANALYSIS_ROOT/human_model_accuracy_comparison.csv"

"$ANALYSIS_PYTHON_BIN" scripts/analysis/make_vocalgrad_accuracy_bootstrap_ci.py \
  --raw-root "$BENCHMARK_RAW_ROOT" \
  --out-csv "$BENCHMARK_ANALYSIS_ROOT/model_accuracy_bootstrap_ci.csv" \
  --out-tex "$BENCHMARK_ANALYSIS_ROOT/model_accuracy_bootstrap_ci.tex"

"$ANALYSIS_PYTHON_BIN" scripts/analysis/make_vocalgrad_parsing_failure_tables.py \
  --raw-root "$BENCHMARK_RAW_ROOT" \
  --out-root "$BENCHMARK_ANALYSIS_ROOT"

"$ANALYSIS_PYTHON_BIN" scripts/analysis/plot_human_model_accuracy_ci_grid.py \
  --model-raw-root "$BENCHMARK_RAW_ROOT" \
  --out "$BENCHMARK_ANALYSIS_ROOT/human_model_accuracy_ci_grid.pdf" \
  --out-csv "$BENCHMARK_ANALYSIS_ROOT/human_model_accuracy_ci_grid.csv"

"$ANALYSIS_PYTHON_BIN" scripts/analysis/plot_human_model_tier_accuracy_lines.py \
  --model-analysis-root "$BENCHMARK_ANALYSIS_ROOT" \
  --out-dir "$BENCHMARK_ANALYSIS_ROOT/human_model_tier_accuracy_lines"

"$ANALYSIS_PYTHON_BIN" scripts/analysis/plot_selected_human_model_tier_panels.py \
  --input-csv "$BENCHMARK_ANALYSIS_ROOT/human_model_tier_accuracy_lines/human_model_tier_accuracy.csv" \
  --out-pdf "$BENCHMARK_ANALYSIS_ROOT/human_model_tier_accuracy_lines/background_noise_voice_pitch_pair.pdf" \
  --out-tex "$BENCHMARK_ANALYSIS_ROOT/human_model_tier_accuracy_lines/background_noise_voice_pitch_pair.tex"

"$ANALYSIS_PYTHON_BIN" scripts/analysis/make_original_source_increase_response_table.py \
  --original-analysis-root "$BENCHMARK_ANALYSIS_ROOT" \
  --original-raw-root "$BENCHMARK_RAW_ROOT" \
  --source-analysis-root "$SOURCE_BIAS_ANALYSIS_ROOT" \
  --source-raw-root "$SOURCE_BIAS_RAW_ROOT" \
  --out-tex "$SOURCE_BIAS_ANALYSIS_ROOT/original_vs_source_increase_response_summary.tex" \
  --out-csv "$SOURCE_BIAS_ANALYSIS_ROOT/original_vs_source_increase_response_summary.csv"

"$ANALYSIS_PYTHON_BIN" scripts/analysis/compare_vocalgrad_prompt_variants.py --root "$PROMPT_ROOT"

"$ANALYSIS_PYTHON_BIN" scripts/analysis/compare_vocalgrad_cross_attribute.py \
  --root "$CROSS_ATTRIBUTE_ROOT/default" \
  --md-out "$CROSS_ATTRIBUTE_ROOT/default/comparison.md" \
  --csv-out "$CROSS_ATTRIBUTE_ROOT/default/comparison.csv" \
  --matrix-csv-out "$CROSS_ATTRIBUTE_ROOT/default/matrix_mean_accuracy.csv"

"$ANALYSIS_PYTHON_BIN" scripts/analysis/plot_vocalgrad_cross_attribute_heatmaps.py \
  --root "$CROSS_ATTRIBUTE_ROOT/default" \
  --out-dir "$CROSS_ATTRIBUTE_ROOT/default/heatmaps"

"$ANALYSIS_PYTHON_BIN" scripts/analysis/make_human_model_accuracy_table.py \
  --model-analysis-root "$FEWSHOT_ROOT/$FEWSHOT_RUN_NAME" \
  --model-raw-root "outputs/raw/vocalgrad_fewshot/$FEWSHOT_RUN_NAME" \
  --out-tex "$FEWSHOT_ROOT/$FEWSHOT_RUN_NAME/human_model_accuracy_comparison.tex" \
  --out-csv "$FEWSHOT_ROOT/$FEWSHOT_RUN_NAME/human_model_accuracy_comparison.csv"

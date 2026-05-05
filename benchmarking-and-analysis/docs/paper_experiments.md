# Paper Experiment Map

This document maps the VocalGrad paper experiments to the
reproduction entrypoints in this repository.

The paper reports results for nine VocalGrad categories:

```text
speaking_speed voice_pitch volume audio_distortion audio_roughness
background_noise echo voice_clarity voice_vibration
```

Reproduction scripts use these nine paper categories by default.

## Main Benchmark

Paper target:
- Section 4.1-4.2
- Table 2: human and model accuracies on VocalGrad
- Appendix C.1-C.4: tier analysis, parsing failures, confidence intervals, and
  output-distribution bias

Primary commands:

```bash
bash scripts/reproduce/01_benchmark.sh BACKEND=kimia
bash scripts/reproduce/01_benchmark.sh BACKEND=mimoaudio
bash scripts/reproduce/01_benchmark.sh BACKEND=stepaudio2
bash scripts/reproduce/01_benchmark.sh BACKEND=audioflamingo3
bash scripts/reproduce/01_benchmark.sh BACKEND=gemini
```

Important inputs:
- `datasets/vocalgrad/test/<category>/audio/`
- `datasets/vocalgrad/test/<category>/metadata/<category>_manifest.csv`
- `outputs/annotation_data/<category>/shared_annotation_50/` for the human row

Important outputs:
- Raw predictions: `outputs/raw/vocalgrad/default/<category>/<model>.jsonl`
- Per-category summaries:
  `outputs/analysis/vocalgrad/default/<category>/<model>.json`
- Table 2:
  `outputs/analysis/vocalgrad/default/human_model_accuracy_comparison.tex`
- Bootstrap confidence intervals:
  `outputs/analysis/vocalgrad/default/model_accuracy_bootstrap_ci.tex`
- Parsing summary tables:
  `outputs/analysis/vocalgrad/default/exact_match_accuracy.tex`
  and `outputs/analysis/vocalgrad/default/parsing_output_match.tex`

## Ablation: Removing Linguistic Content

Paper target:
- Section 4.3
- Table 3

Primary commands:

```bash
bash scripts/reproduce/02_ablation_beep.sh BACKEND=kimia
bash scripts/reproduce/02_ablation_beep.sh BACKEND=mimoaudio
bash scripts/reproduce/02_ablation_beep.sh BACKEND=stepaudio2
bash scripts/reproduce/02_ablation_beep.sh BACKEND=audioflamingo3
bash scripts/reproduce/02_ablation_beep.sh BACKEND=gemini
```

Important inputs:
- `datasets/vocalgrad/ablation_beep/`

Important outputs:
- Raw predictions:
  `outputs/raw/vocalgrad_ablation_beep/default/<category>/<model>.jsonl`
- Table 3:
  `outputs/analysis/vocalgrad_ablation_beep/default/ablation_beep_accuracy_comparison.tex`

## Linear Probing: Audio Representations

Paper target:
- Section 5.1-5.3
- Figure 2

Primary command:

```bash
bash scripts/reproduce/03_linear_probe_audio.sh
```

Important inputs:
- `datasets/vocalgrad/train/`
- `datasets/vocalgrad/test/`

Important outputs:
- Cached features:
  `outputs/features/linear_probe/{train,test}/<model_stem>/<category>.npz`
- Per-category probe results:
  `outputs/analysis/linear_probe/<model_stem>/default/<category>.json`
- Cross-model comparison:
  `outputs/analysis/linear_probe/_comparisons/default/comparison.csv`
- Figure 2 style heatmaps:
  `outputs/analysis/linear_probe/_comparisons/default/heatmaps/`

The audio-representation stages are:
- `mel`: shared log-mel reference features
- `pre_lm_audio`: representations before the language decoder
- `lm_audio`: late language-decoder audio-token representations
- `lm_text`: final text-token representation from the standard extractor

## Linear Probing: Text Prediction Layers

Paper target:
- Section 5.1-5.3
- Figure 3

Primary command:

```bash
bash scripts/reproduce/04_linear_probe_layers.sh
```

Important outputs:
- Cached features:
  `outputs/features/linear_probe_lm_text_layers/{train,test}/<model_stem>/<category>.npz`
- Per-category layer-wise probe results:
  `outputs/analysis/linear_probe_lm_text_layers/<model_stem>/default/<category>.json`
- Layer comparison:
  `outputs/analysis/linear_probe_lm_text_layers/_comparisons/default/comparison.csv`
- Figure 3 style line plots:
  `outputs/analysis/linear_probe_lm_text_layers/_comparisons/default/layer_lines/`

## LoRA Fine-Tuning

Paper target:
- Section 5.4
- Table 4
- Figure 4
- Appendix C.7 / Table 12

Primary command, one model-category-scope at a time:

```bash
bash scripts/reproduce/05_finetune_lora.sh MODEL_STEM=kimi-audio TRAIN_CATEGORY=volume SCOPE=lm_head_only
bash scripts/reproduce/05_finetune_lora.sh MODEL_STEM=kimi-audio TRAIN_CATEGORY=volume SCOPE=all_linear
```

The paper reproduction wrapper sets one training epoch, learning rate `2e-4`,
batch size `1`, gradient accumulation `16`, LoRA rank `8`, LoRA alpha `16`,
and LoRA dropout `0.05`.

For the fine-tuned Kimi layer-probing panels in Figure 4b:

```bash
bash scripts/reproduce/07_finetuned_kimi_probe.sh TRAIN_CATEGORY=volume
bash scripts/reproduce/07_finetuned_kimi_probe.sh TRAIN_CATEGORY=voice_pitch
```

Important inputs:
- `datasets/vocalgrad/train/`
- `datasets/vocalgrad/test/`

Important outputs:
- Fine-tuning data:
  `outputs/data/vocalgrad_finetune/<category>/`
- Adapters:
  `outputs/checkpoints/vocalgrad_finetune/<model_stem>/<run_name>/<scope>/<category>/adapter/`
- Cross-category raw predictions:
  `outputs/raw/vocalgrad_finetune_eval/<model_stem>/<run_name>/<scope>/<category>/`
- Cross-category analysis:
  `outputs/analysis/vocalgrad_finetune_eval/<model_stem>/<run_name>/<scope>/<category>/`
- Aggregated comparison:
  `outputs/analysis/vocalgrad_finetune_eval/_comparisons/<run_name>/`
- Fine-tuned Kimi probing plots:
  `outputs/analysis/kimi_finetuned_probe_completed/<run_name>/`

## Appendix Analyses

Paper target:
- Appendix C.1-C.6

Primary command:

```bash
bash scripts/reproduce/06_appendix_analyses.sh
```

This script regenerates analysis tables and plots from existing raw prediction
files. It does not run model inference.

The prompt-sensitivity rows require running the main benchmark with each prompt
variant before aggregation:

```bash
bash scripts/reproduce/01_benchmark.sh BACKEND=kimia VARIANT=swap-order RUN_TABLES=0
bash scripts/reproduce/01_benchmark.sh BACKEND=kimia VARIANT=audio-ref RUN_TABLES=0
```

Repeat those commands for the other reported backends by changing `BACKEND`.

The cross-attribute appendix requires raw cross-attribute generations:

```bash
.venv-kimi/bin/python -m plic.cli_vocalgrad_cross_attribute_all --backend kimia
```

Repeat for other backends with the corresponding backend environment and
`--backend` value.

The few-shot appendix uses two examples per label, seed 1234, and the explicit
audio-reference prompt variant:

```bash
.venv-kimi/bin/python -m plic.cli_vocalgrad_fewshot_all \
  --backend kimia \
  --raw-root outputs/raw/vocalgrad_fewshot/fewshot_2x2_seed1234_audio-ref \
  --analysis-root outputs/analysis/vocalgrad_fewshot/fewshot_2x2_seed1234_audio-ref \
  --shots-per-label 2 \
  --seed 1234 \
  --explicit-audio-clip-reference
```

Repeat for other backends with the corresponding backend environment and
`--backend` value.

The source-clip response-bias rows in Appendix C.4 / Table 8 require raw source
clip generations. Generate them one backend at a time before running the
appendix aggregation:

```bash
.venv-kimi/bin/python scripts/run/evaluate_source_clips_generation_bias.py --backend kimia
.venv-mimo/bin/python scripts/run/evaluate_source_clips_generation_bias.py --backend mimoaudio
.venv-stepaudio/bin/python scripts/run/evaluate_source_clips_generation_bias.py --backend stepaudio2
.venv-af3/bin/python scripts/run/evaluate_source_clips_generation_bias.py --backend audioflamingo3
.venv/bin/python scripts/run/evaluate_source_clips_generation_bias.py --backend gemini
```

# VocalGrad Reproduction Code

This repository contains the experiment code used for the VocalGrad paper:
evaluating whether large audio language models perceive gradual acoustic
changes in speech.

The current reproduction target is the VocalGrad paper experiments. Utilities
that were not required for the paper reproduction path have been removed.

## What To Reproduce

Start with the experiment map:

- `docs/paper_experiments.md`: paper table/figure to command mapping
- `docs/data_setup.md`: expected dataset and annotation layout
- `docs/environment_setup.md`: backend-specific Python environments

The paper reports nine VocalGrad categories:

```text
speaking_speed voice_pitch volume audio_distortion audio_roughness
background_noise echo voice_clarity voice_vibration
```

## Repository Layout

- `src/plic/`: VocalGrad runtime, backend integrations, and shared experiment code
- `scripts/run/`: lower-level execution scripts for inference, probing, and fine-tuning
- `scripts/analysis/`: table and figure generation scripts
- `scripts/reproduce/`: scheduler-free shell entrypoints for paper reproduction
- `docs/`: reproduction notes and operational details
- `datasets/`: local dataset mount or copy, not tracked by git
- `outputs/`: generated predictions, features, analysis files, and checkpoints

Use `scripts/reproduce/` as the portable command surface. Scheduler-specific
job files are not part of this reproducibility package.
For paper-number reproduction, prefer these wrapper scripts over calling
lower-level files in `scripts/run/` directly.

## Environment

Run these commands from `benchmarking-and-analysis/`. Create the base analysis
environment, including PyTorch for linear-probe training, first:

```bash
uv venv --python 3.10 .venv
UV_PROJECT_ENVIRONMENT=.venv uv sync --locked --extra analysis
```

Then create backend-specific environments as needed. See
[the environment setup](docs/environment_setup.md) for the experimental platform
and Kimi-Audio, MiMo-Audio, Step-Audio-2 Mini, AudioFlamingo3, and Gemini details.

For MiMo-Audio, the runtime uses the upstream `audio_understanding` generation
path without passing local sampling parameters; the upstream defaults include
temperature 0.3 and top-p 0.95. The local CLI still records its generic
temperature field in run manifests, but that field is not passed into the
MiMo-Audio generation call.

## Data

Place VocalGrad under:

```text
datasets/vocalgrad/test/
datasets/vocalgrad/train/
datasets/vocalgrad/ablation_beep/
```

Human annotation files for Table 2 are expected under:

```text
outputs/annotation_data/<category>/shared_annotation_50/
```

See `docs/data_setup.md` for the exact manifest and directory contract.

## Main Commands

Run the main benchmark one backend at a time:

```bash
bash scripts/reproduce/01_benchmark.sh BACKEND=kimia
bash scripts/reproduce/01_benchmark.sh BACKEND=mimoaudio
bash scripts/reproduce/01_benchmark.sh BACKEND=stepaudio2
bash scripts/reproduce/01_benchmark.sh BACKEND=audioflamingo3
bash scripts/reproduce/01_benchmark.sh BACKEND=gemini
```

Run the beep ablation:

```bash
bash scripts/reproduce/02_ablation_beep.sh BACKEND=kimia
```

Run linear probing:

```bash
bash scripts/reproduce/03_linear_probe_audio.sh
bash scripts/reproduce/04_linear_probe_layers.sh
```

Run LoRA fine-tuning for one model/category/scope:

```bash
bash scripts/reproduce/05_finetune_lora.sh MODEL_STEM=kimi-audio TRAIN_CATEGORY=volume SCOPE=lm_head_only
bash scripts/reproduce/05_finetune_lora.sh MODEL_STEM=kimi-audio TRAIN_CATEGORY=volume SCOPE=all_linear
```

Run the fine-tuned Kimi probing analysis used for Figure 4b:

```bash
bash scripts/reproduce/07_finetuned_kimi_probe.sh TRAIN_CATEGORY=volume
bash scripts/reproduce/07_finetuned_kimi_probe.sh TRAIN_CATEGORY=voice_pitch
```

Regenerate appendix analyses from existing raw predictions:

```bash
bash scripts/reproduce/06_appendix_analyses.sh
```

Appendix prompt-sensitivity, cross-attribute, few-shot, and source-clip
response-bias analyses require their corresponding raw inference files first.
The required generation commands are listed in `docs/paper_experiments.md`.

Use `MAX_SAMPLES`, `MAX_SAMPLES_PER_CATEGORY`, or `MAX_EVAL_SAMPLES` for small
local checks before running full experiments.

## Camera-ready experiments

The additional paper experiments follow the existing `scripts/reproduce/`, `scripts/run/`, and `scripts/analysis/` layout. See `docs/camera_ready_experiments.md` for commands and table/figure mappings.

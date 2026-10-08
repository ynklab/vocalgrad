# VocalGrad

This directory contains the data-construction and human-annotation utilities for
VocalGrad, a benchmark for evaluating whether audio language models can detect
the direction of gradual acoustic change in speech.

The published dataset is available at [ynklab/vocalgrad on Hugging Face](https://huggingface.co/datasets/ynklab/vocalgrad).
You can also regenerate the dataset locally from VCTK.

This directory focuses on reproducing the dataset and annotation assets. Model
benchmarking, representation probing, and LoRA fine-tuning scripts are in the
sibling `benchmarking-and-analysis/` directory.

## Benchmark Summary

VocalGrad uses VCTK source speech and applies controlled time-varying
transformations. The task is binary:

```text
Does the {attribute} increase or decrease over time?

Answer with only one word: "increase" or "decrease".
```

Active categories:

- `volume`
- `speaking_speed`
- `voice_pitch`
- `background_noise`
- `audio_distortion`
- `audio_roughness`
- `voice_clarity`
- `voice_vibration`
- `echo`

## Dataset Sizes

Speech benchmark:

- test source clips: `50 speakers x 2 clips = 100`
- train source clips: `500` clips sampled from `57` non-test speakers
- conditions per source clip: `3 tiers x 4 curves x 2 directions = 24`
- test clips per active category: `100 x 24 = 2400`
- train clips per active category: `500 x 24 = 12000`
- active categories: `9`
- total test clips: `21600`
- total train clips: `108000`

Synthetic beep ablation:

- source clips: `10`
- categories: `4` (`voice_pitch`, `volume`, `background_noise`, `voice_vibration`)
- clips per category: `10 x 24 = 240`
- total ablation clips: `960`

## Repository Layout

```text
.
├── annotation_tool/                 # Web tool and manifests for human annotation
├── configs/                         # Category-specific augmentation hyperparameters
├── data/
│   ├── metadata/selection/          # Reproducible source clip selections
│   ├── original/VCTK-Corpus/        # Manually placed VCTK source data, not tracked
│   ├── processed/                   # Generated benchmark audio, not tracked
│   └── vocalgrad_public/            # Hugging Face packaging output, not tracked
├── docs/
│   ├── pipeline_runbook.md          # End-to-end dataset generation commands
│   └── reproducibility.md           # Paper-section to repository mapping
└── scripts/                         # Selection, generation, packaging, and validation tools
```

## Setup

Requirements:

- Python 3.11+
- `uv`
- VCTK 0.92 placed at `data/original/VCTK-Corpus`

Expected VCTK layout:

```text
data/original/VCTK-Corpus/
├── speaker-info.txt
├── txt/pXXX/*.txt
└── wav48/pXXX/*.wav
```

Install dependencies:

```bash
uv sync --group dev
```

## Reproduce Dataset Construction

Select source clips:

```bash
uv run python scripts/select_vocalgrad_splits.py \
  --exclude-clip-id p306_140 \
  --exclude-clip-id p360_140

uv run python scripts/select_train_source_clips.py
```

The train selection samples 500 clips as evenly as possible across all non-test
VCTK speakers. In the current selection this yields 57 train speakers, with 8 or
9 clips per speaker.

Detect and merge onset/offset:

```bash
uv run python scripts/analyze_selected_clip_onset.py

uv run python scripts/analyze_selected_clip_onset.py \
  --source-csv data/metadata/selection/train_source_clips_500.csv \
  --output-dir data/metadata/analysis/train_onset_offset_min10_retuned
```

Generate all active speech benchmark clips:

```bash
uv run python scripts/generate_all_benchmarks.py --split all --overwrite
```

Generate the synthetic beep ablation:

```bash
uv run python scripts/generate_beep_source_clips.py
uv run python scripts/generate_ablation_beep_benchmarks.py --overwrite
```

Validate metadata and generated manifests:

```bash
uv run python scripts/validate_vocalgrad_dataset.py --require-processed
```

For detailed commands, expected outputs, and validation checks, see
[docs/pipeline_runbook.md](docs/pipeline_runbook.md).

## Public Dataset Packaging

Prepare the test-only public release:

```bash
uv run python scripts/prepare_vocalgrad_public.py
```

The generated public dataset can also be downloaded from [ynklab/vocalgrad on Hugging Face](https://huggingface.co/datasets/ynklab/vocalgrad).

For placement of the downloaded test split, generated train/beep data, and
unmodified test sources into the evaluation directory, see
[the evaluation data setup](../benchmarking-and-analysis/docs/data_setup.md).

## Human Annotation

The annotation interface and shared 50-item manifests are in `annotation_tool/`.
The annotation subset is balanced over source clips, tiers, trajectories, and
directions. The tool records binary labels, response time, and replay count.

See [annotation_tool/README.md](annotation_tool/README.md) for local setup and
usage details.

## Reproducibility Map

The repository-level mapping from paper sections to commands, inputs, and
expected artifacts is maintained in
[docs/reproducibility.md](docs/reproducibility.md).

## Development

Run lint checks:

```bash
uv run ruff check .
```

Format code:

```bash
uv run ruff format .
```

## Alternative source datasets

The frozen LoquaciousSet and Common Voice test selections and their generation instructions are documented in `docs/alternate_sources.md`.

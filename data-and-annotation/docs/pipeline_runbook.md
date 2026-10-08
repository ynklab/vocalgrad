# VocalGrad Pipeline Runbook

This runbook describes how to reproduce the dataset assets maintained in this
repository. It covers source selection, onset/offset detection, active speech
benchmark generation, synthetic beep ablation generation, public packaging, and
human annotation manifests.

Model benchmarking, linear probing, and LoRA fine-tuning are handled in the
sibling `benchmarking-and-analysis/` directory.

## Assumptions

- Working directory: this directory, `data-and-annotation/`
- Dependencies are installed with `uv sync --group dev`
- VCTK 0.92 is available at `data/original/VCTK-Corpus`
- Generated audio under `data/processed/` is not tracked by git

The generated public dataset can also be downloaded from [ynklab/vocalgrad on Hugging Face](https://huggingface.co/datasets/ynklab/vocalgrad).

## Quick Start

```bash
uv sync --group dev

uv run python scripts/select_vocalgrad_splits.py \
  --exclude-clip-id p306_140 \
  --exclude-clip-id p360_140

uv run python scripts/select_train_source_clips.py

uv run python scripts/analyze_selected_clip_onset.py

uv run python scripts/analyze_selected_clip_onset.py \
  --source-csv data/metadata/selection/train_source_clips_500.csv \
  --output-dir data/metadata/analysis/train_onset_offset_min10_retuned

uv run python scripts/generate_all_benchmarks.py --split all --overwrite

uv run python scripts/generate_beep_source_clips.py
uv run python scripts/generate_ablation_beep_benchmarks.py --overwrite

uv run python scripts/validate_vocalgrad_dataset.py --require-processed

uv run python scripts/prepare_vocalgrad_public.py
```

## Step 1: Test Source Selection

Command:

```bash
uv run python scripts/select_vocalgrad_splits.py \
  --exclude-clip-id p306_140 \
  --exclude-clip-id p360_140
```

Behavior:

- Selects `50` VCTK speakers for the test split.
- Samples `2` clips per selected speaker.
- Requires `duration >= 3.0 sec`.
- Copies transcript text into metadata.

Outputs:

- `data/metadata/selection/test_speakers_50.csv`
- `data/metadata/selection/test_source_clips_100.csv`
- `data/metadata/selection/selection_report.json`

Expected:

- `100` test source clips
- no source clip shorter than `3.0 sec`

## Step 2: Train Source Selection

Command:

```bash
uv run python scripts/select_train_source_clips.py
```

Behavior:

- Uses VCTK speakers not used in the test split.
- Samples `500` train clips as evenly as possible across non-test speakers.
- Requires `duration >= 3.0 sec`.
- Excludes transcript text that overlaps with selected test source clips.

Outputs:

- `data/metadata/selection/train_speakers_57.csv`
- `data/metadata/selection/train_source_clips_500.csv`
- `data/metadata/selection/train_selection_report.json`

Expected:

- `500` train source clips
- `57` train speakers in the current selection
- `8` or `9` clips per train speaker
- no transcript overlap with the test source clips

## Step 3: Onset/Offset Detection

Test source clips:

```bash
uv run python scripts/analyze_selected_clip_onset.py
```

Train source clips:

```bash
uv run python scripts/analyze_selected_clip_onset.py \
  --source-csv data/metadata/selection/train_source_clips_500.csv \
  --output-dir data/metadata/analysis/train_onset_offset_min10_retuned
```

Default detection parameters:

- `frame_ms = 10`
- `hop_ms = 10`
- `noise_ref_ms = 200`
- `noise_multiplier = 1.8`
- `peak_fraction = 0.003`
- `abs_threshold = 5e-5`
- `min_consecutive_frames = 10`
- `pre_emphasis_alpha = 0.97`

Behavior:

- Computes short-time RMS.
- Estimates a noise floor from the initial region.
- Marks speech frames using an adaptive threshold.
- Retains only active regions with enough consecutive frames.
- Merges onset/offset columns back into the source CSV.

Expected:

- test: `100 / 100` detected
- train: `500 / 500` detected
- no missing `onset_sec` or `offset_sec`

## Step 4: Active Speech Benchmark Generation

Command:

```bash
uv run python scripts/generate_all_benchmarks.py --split all --overwrite
```

Use `--split test` or `--split train` to generate one split only.

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

Condition grid:

- `tier`: `low`, `mid`, `high`
- `curve`: `linear`, `quad_first_flat`, `quad_last_flat`, `jump`
- `direction`: `up`, `down`

Expected outputs:

- test: `100 source clips x 24 conditions x 9 categories = 21600 clips`
- train: `500 source clips x 24 conditions x 9 categories = 108000 clips`
- per test category: `2400` clips
- per train category: `12000` clips

The category hyperparameters are stored in `configs/*_benchmark.json`.

## Step 5: Synthetic Beep Ablation

Generate the 10 synthetic source clips:

```bash
uv run python scripts/generate_beep_source_clips.py
```

Generate the four ablation categories used in the paper:

```bash
uv run python scripts/generate_ablation_beep_benchmarks.py --overwrite
```

Categories:

- `voice_pitch`
- `volume`
- `background_noise`
- `voice_vibration`

Expected outputs:

- per category: `10 source clips x 24 conditions = 240 clips`
- total: `960` clips

## Step 6: Human Annotation Manifests

Command:

```bash
uv run python scripts/build_annotation_manifests.py
```

Behavior:

- Builds shared 50-item annotation manifests from processed test manifests.
- Reuses the same source-clip and condition assignment structure across
  categories.
- Balances `increase` / `decrease`.
- Balances tiers and trajectories as evenly as possible over 50 items.

Outputs:

- `annotation_tool/data/manifests/<category>/shared_annotation_50.csv`
- `annotation_tool/data/manifests/shared_source_clips_50.csv`
- `annotation_tool/data/manifests/shared_source_clips_50.txt`

Notes:

- Raw annotation results and local credentials are not intended for public
  release.

## Step 7: Public Packaging

Prepare the test-only Hugging Face layout:

```bash
uv run python scripts/prepare_vocalgrad_public.py
```

Outputs:

- `data/vocalgrad_public/`

## Validation

Run this after source selection to validate tracked metadata:

```bash
uv run python scripts/validate_vocalgrad_dataset.py
```

Run this after generating benchmark audio to require all processed manifests:

```bash
uv run python scripts/validate_vocalgrad_dataset.py --require-processed
```

Run this after public packaging to require the Hugging Face metadata:

```bash
uv run python scripts/validate_vocalgrad_dataset.py --require-processed --require-public
```

The validation script checks:

- source metadata row counts
- train/test speaker disjointness
- train/test transcript non-overlap
- onset/offset completeness
- train clip-count distribution over speakers
- processed manifest row counts when `--require-processed` is set
- public `metadata.csv` row count when `--require-public` is set

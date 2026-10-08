# VocalGrad Reproducibility Map

This document maps the paper components that are reproducible from this
repository to the corresponding inputs, commands, and expected artifacts.

Model benchmarking, hidden-state probing, and LoRA fine-tuning are handled in
the sibling `benchmarking-and-analysis/` directory.

## Scope

In scope:

- VCTK source clip selection
- train/test source metadata
- speech onset/offset detection
- 9-category VocalGrad speech benchmark generation
- 4-category synthetic beep ablation generation
- public Hugging Face dataset packaging
- human annotation manifests and annotation web tool

Out of scope for this directory:

- running Gemini 3 Flash, Kimi-Audio, MiMo-Audio, Step-Audio-2 Mini, or
  AudioFlamingo3
- aggregating model prediction CSVs
- linear probing over model representations
- LoRA fine-tuning

## Paper-To-Repository Map

| Paper component | Repository support | Main commands | Expected artifacts |
| --- | --- | --- | --- |
| Section 3.2 Source Clip Selection | Select VCTK test and train source clips | `scripts/select_vocalgrad_splits.py`, `scripts/select_train_source_clips.py` | `test_source_clips_100.csv`, `train_source_clips_500.csv` |
| Section 3.2 Speech Onset and Offset Detection | Energy-based speech span detection | `scripts/analyze_selected_clip_onset.py` | onset/offset columns in source CSVs |
| Section 3.2 Data Augmentation | Shared condition grid over tiers, curves, directions | `scripts/generate_all_benchmarks.py --split all --overwrite` | `data/processed/test/*`, `data/processed/train/*` |
| Section 3.3 Category-Specific Augmentations | 9 active augmentation generators and configs | `scripts/generate_*_benchmark.py`, `configs/*_benchmark.json` | per-category manifests and generated wavs |
| Section 3.4 Prompting Format | Prompt text in public metadata | [Published dataset](https://huggingface.co/datasets/ynklab/vocalgrad) | Downloaded `metadata.csv` |
| Section 4.1 Human Evaluation Setup | Shared 50-item manifests and web interface | `scripts/build_annotation_manifests.py`, `annotation_tool/` | `annotation_tool/data/manifests/*/shared_annotation_50.csv` |
| Section 4.3 Ablation: Removing Linguistic Content | Synthetic beep sources and four ablation categories | `scripts/generate_beep_source_clips.py`, `scripts/generate_ablation_beep_benchmarks.py` | `data/processed/ablation_beep/*` |
| Public Dataset Release | Download and arrange the test split | [Data setup](../../benchmarking-and-analysis/docs/data_setup.md) | `datasets/vocalgrad/test/` in the evaluation directory |
| Repository Validation | Check source metadata, manifests, and public metadata | `scripts/validate_vocalgrad_dataset.py` | pass/fail validation output |

## Dataset Construction Contract

The active benchmark uses these fixed factors:

- categories: `9`
- tiers: `low`, `mid`, `high`
- curves: `linear`, `quad_first_flat`, `quad_last_flat`, `jump`
- directions: `up`, `down`
- label mapping: `up -> increase`, `down -> decrease`

Expected source selections:

- test: `50 speakers x 2 clips = 100 source clips`
- train: `500 source clips` sampled as evenly as possible from the `57`
  non-test speakers in the current selection
- train/test speaker sets do not overlap
- train transcript text does not overlap with selected test transcript text

Expected generated speech benchmark sizes:

- test: `100 x 24 x 9 = 21600`
- train: `500 x 24 x 9 = 108000`
- per test category: `2400`
- per train category: `12000`

Expected synthetic beep ablation sizes:

- source clips: `10`
- categories: `4`
- per category: `240`
- total: `960`

## Data Access

Generated benchmark audio is not tracked in git. It can be either regenerated
locally from VCTK or downloaded from [ynklab/vocalgrad on Hugging Face](https://huggingface.co/datasets/ynklab/vocalgrad).

The local generation commands are documented in
[pipeline_runbook.md](pipeline_runbook.md).

## Human Annotation Assets

The annotation tool supports local collection of binary labels with replay count
and response time. Public repository sharing should include:

- annotation tool source code
- shared annotation manifests
- local setup instructions

Public repository sharing should exclude:

- local plaintext credentials
- hashed local user files
- raw per-user annotation JSONL files unless explicitly anonymized and approved

See [annotation_tool/README.md](../annotation_tool/README.md).

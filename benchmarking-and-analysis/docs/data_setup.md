# Data Setup

This repository expects VocalGrad data to be available as local files. The code
does not commit datasets, raw predictions, features, annotations, or checkpoints.

## Download and Arrange the Public Test Split

Run from `benchmarking-and-analysis/`. Download the published test audio and
metadata, then convert the release layout into the evaluation layout:

```bash
uvx --from huggingface-hub hf download ynklab/vocalgrad --repo-type dataset \
  --include metadata.csv 'audio/**' --local-dir datasets/vocalgrad_hf
python3 scripts/prepare_vocalgrad_data.py hf-test \
  --download-root datasets/vocalgrad_hf \
  --output-root datasets/vocalgrad/test --mode symlink
```

The standard-library-only converter reads each `file_name` from `metadata.csv`,
including numbered audio shards. It uses `category`, `direction`, and
`benchmark_clip_id` to place the audio and writes one manifest per category.
All metadata columns are preserved. It does not decode, resample, or otherwise
modify the audio. Use `--mode copy` to make the output independent of the
HF download directory; symlinks require that directory to remain in place.

Before writing, the converter checks for 2,400 clips in each of nine categories,
duplicate category/clip IDs, valid direction/label pairs, missing source files,
and existing destination files. Existing files are never overwritten. Use a
fresh destination if repeating the conversion. These checks inspect metadata
and file existence, not audio contents or hashes.

## Generate and Arrange Train and Beep Data

The public HF release contains the test split. For train and beep data, follow
[the construction runbook](../../data-and-annotation/docs/pipeline_runbook.md)
to prepare VCTK, source selections, and onset/offset metadata. Preserve the
committed source selections when reproducing the paper. From
`data-and-annotation/`, generate the remaining audio:

```bash
uv sync --group dev
uv run python scripts/generate_all_benchmarks.py --split train
uv run python scripts/generate_beep_source_clips.py
uv run python scripts/generate_ablation_beep_benchmarks.py
```

Then run from `benchmarking-and-analysis/` to link the generated splits:

```bash
mkdir -p datasets/vocalgrad
for split in train ablation_beep; do
  generated="$PWD/../data-and-annotation/data/processed/$split"
  destination="datasets/vocalgrad/$split"
  if [ ! -d "$generated" ] || [ -e "$destination" ] || [ -L "$destination" ]; then
    echo "Missing source or existing destination: $split" >&2
    break
  fi
  ln -s "$generated" "$destination"
done
```

Alternatively, copy each generated split directory to its corresponding
previously nonexistent destination. Both generated splits already use the
required category/audio/metadata layout. Expected sizes are 108,000 train clips
(12,000 per category) and 960 beep clips (240 per category across four categories).
If generating test audio locally instead of downloading HF, generate with
`--split test` and place `data/processed/test` at `datasets/vocalgrad/test` in
the same way. Choose one source for the test split.

## Unmodified Sources for Alternative Problem Formats

The binary and ternary experiment wrappers also require the 100 unmodified
VCTK test source clips. These are separate from the augmented HF test audio.
With VCTK arranged as `wav48/<speaker>/<clip_id>.wav`, run from
`benchmarking-and-analysis/`:

```bash
python3 scripts/prepare_vocalgrad_data.py sources \
  --selection-csv ../data-and-annotation/data/metadata/selection/test_source_clips_100.csv \
  --vctk-root ../data-and-annotation/data/original/VCTK-Corpus \
  --output-root datasets/vocalgrad/test/source_clips --mode symlink
```

This creates `source_clips/audio/<speaker>/<clip_id>.wav` and
`source_clips/manifest.csv`. It preserves `test_item_id`, source IDs, text, and
onset/offset metadata, and adds the relative `selected_audio_path` required by
the evaluation loaders. The audio remains full-length: no onset/offset trimming
or augmentation is applied. The command checks the 100-row selection, unique
item IDs and destinations, and source file existence before writing.
`--mode copy` is also available. VCTK must already be in the WAV layout above;
this command does not convert an upstream FLAC distribution or select a microphone.

## Optional Validation

For locally generated data, the construction-side validator is available from
`data-and-annotation/`:

```bash
uv run python scripts/validate_vocalgrad_dataset.py --require-processed
```

It expects the construction-side inputs and generated splits; it is not a
validator for an HF-only download. The two placement commands above perform
their own metadata and file-existence checks as part of conversion. Full audio
validation and model inference are separate from placement.

## Required Layout

Place the paper evaluation split here:

```text
datasets/vocalgrad/test/
  <category>/
    audio/
      up/*.wav
      down/*.wav
    metadata/
      <category>_manifest.csv
```

Place the training split used for linear probing and LoRA fine-tuning here:

```text
datasets/vocalgrad/train/
  <category>/
    audio/
      up/*.wav
      down/*.wav
    metadata/
      <category>_manifest.csv
```

Place the sine-wave ablation split here:

```text
datasets/vocalgrad/ablation_beep/
  voice_pitch/
  volume/
  background_noise/
  voice_vibration/
```

Human annotation files for Table 2 are expected under:

```text
outputs/annotation_data/<category>/shared_annotation_50/annotator_*.jsonl
```

## Paper Categories

The paper reports these nine categories:

```text
speaking_speed voice_pitch volume audio_distortion audio_roughness
background_noise echo voice_clarity voice_vibration
```

The reproduction scripts default to these nine categories. Override
`CATEGORIES` only when intentionally running a subset of the paper categories.

## Manifest Contract

Each manifest row must contain at least:

- `category`
- `direction`, with `up` for increase and `down` for decrease
- `benchmark_clip_id`

The loader resolves audio paths as:

```text
<dataset_root>/<category>/audio/<direction>/<benchmark_clip_id>.wav
```

Additional metadata columns are preserved and used by analysis scripts when
available, for example intensity tier and transformation trajectory.

## External Assets

VocalGrad is derived from VCTK. Keep original dataset licensing and citation
requirements with the released dataset. Model weights are downloaded from their
respective providers and are not stored in this repository.

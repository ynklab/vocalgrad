# Data Setup

This repository expects VocalGrad data to be available as local files. The code
does not commit datasets, raw predictions, features, annotations, or checkpoints.

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

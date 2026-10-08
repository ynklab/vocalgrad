# VocalGrad

This repository contains the reproducibility material for "VocalGrad: Evaluating Acoustic Perception in Audio Language
Models", accepted at the **NeurIPS 2026 Evaluations & Datasets Track**.

Dataset: [ynklab/vocalgrad on Hugging Face](https://huggingface.co/datasets/ynklab/vocalgrad).

VocalGrad evaluates whether large audio language models can detect the direction
of gradual acoustic change in speech. The paper uses nine categories:

```text
speaking_speed voice_pitch volume audio_distortion audio_roughness
background_noise echo voice_clarity voice_vibration
```

## Repository Layout

- `data-and-annotation/`: VCTK source selection, VocalGrad audio generation,
  synthetic beep ablation generation, public dataset packaging, and the human
  annotation web tool.
- `benchmarking-and-analysis/`: model inference, linear probing, LoRA
  fine-tuning, and table/figure generation for the paper experiments.

The generated audio, model predictions, hidden-state features, checkpoints, and
raw human annotation files are not committed. They are either regenerated from
the scripts or placed in the documented local paths.

## Reproduction Order

1. Build or obtain the VocalGrad dataset with
   [the data preparation instructions](data-and-annotation/README.md).
2. Place the generated or downloaded data under the layout described in
   [the dataset layout](benchmarking-and-analysis/docs/data_setup.md).
3. Prepare the analysis and required backend environments using
   [the environment setup](benchmarking-and-analysis/docs/environment_setup.md).
4. Run the paper experiment entrypoints in
   `benchmarking-and-analysis/scripts/reproduce/`.

For appendix-only prompt sensitivity, cross-attribute, few-shot, and source-clip
response-bias runs, see
`benchmarking-and-analysis/docs/paper_experiments.md`.

Useful entrypoints:

```bash
cd data-and-annotation
uv sync --group dev
uv run python scripts/select_vocalgrad_splits.py --exclude-clip-id p306_140 --exclude-clip-id p360_140
uv run python scripts/select_train_source_clips.py
uv run python scripts/analyze_selected_clip_onset.py
uv run python scripts/generate_all_benchmarks.py --split all --overwrite
uv run python scripts/generate_beep_source_clips.py
uv run python scripts/generate_ablation_beep_benchmarks.py --overwrite

cd ../benchmarking-and-analysis
UV_PROJECT_ENVIRONMENT=.venv uv sync --locked --extra analysis
UV_PROJECT_ENVIRONMENT=.venv-kimi uv sync --locked --extra kimia
bash scripts/reproduce/01_benchmark.sh BACKEND=kimia
bash scripts/reproduce/03_linear_probe_audio.sh
bash scripts/reproduce/04_linear_probe_layers.sh
bash scripts/reproduce/05_finetune_lora.sh MODEL_STEM=kimi-audio TRAIN_CATEGORY=volume SCOPE=all_linear
```

See the subdirectory READMEs for the full setup and expected outputs.

## Coding Assistance

We used OpenAI Codex (https://openai.com/codex/) for coding assistance and for
reorganizing the code before submission.

## Camera-ready additions

See `benchmarking-and-analysis/docs/camera_ready_experiments.md` for the published additional experiments and `data-and-annotation/docs/alternate_sources.md` for the frozen alternate-source selections.

## Remaining camera-ready revisions

For the evaluation rules and table/figure generation commands,
see [the evaluation and reproduction procedure](benchmarking-and-analysis/docs/camera_ready_experiments.md#evaluation-and-analysis).

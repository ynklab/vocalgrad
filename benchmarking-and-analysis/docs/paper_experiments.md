# Paper Experiment Map

This document maps the VocalGrad paper experiments to the
reproduction entrypoints in this repository.

The paper reports results for nine VocalGrad categories:

```text
speaking_speed voice_pitch volume audio_distortion audio_roughness
background_noise echo voice_clarity voice_vibration
```

Reproduction scripts use these nine paper categories by default.

Run all commands below from `benchmarking-and-analysis/`. Prepare the
[backend environments](environment_setup.md) and [input datasets](data_setup.md)
first. For direct Python module calls, install the package in that environment
or set `PYTHONPATH=src`.

## Experiment Index

| Experiment / paper target | Instructions | Primary entrypoint |
| --- | --- | --- |
| Data generation; Sections 3.1–3.3, Appendix B.2–B.3 | [Data preparation](../../data-and-annotation/README.md), [trajectories](../../data-and-annotation/docs/augmentation_curves.md) | `../data-and-annotation/scripts/generate_all_benchmarks.py` |
| Human annotation; Appendix B.4 | [Annotation tool](../../data-and-annotation/annotation_tool/README.md) | `../data-and-annotation/scripts/build_annotation_manifests.py` |
| Main benchmark; Table 2 | [Main Benchmark](#main-benchmark) | `scripts/reproduce/01_benchmark.sh` |
| Beep ablation; Table 3 | [Removing Linguistic Content](#ablation-removing-linguistic-content) | `scripts/reproduce/02_ablation_beep.sh` |
| Audio-stage probes; Figure 2 | [Audio Representations](#linear-probing-audio-representations) | `scripts/reproduce/03_linear_probe_audio.sh` |
| Text-prediction layer probes; Figure 3 | [Text Prediction Layers](#linear-probing-text-prediction-layers) | `scripts/reproduce/04_linear_probe_layers.sh` |
| Fine-tuning; Tables 4, 20, Figure 4 | [LoRA Fine-Tuning](#lora-fine-tuning) | `scripts/reproduce/05_finetune_lora.sh`, `07_finetuned_kimi_probe.sh` |
| Trajectory breakdown; Table 5 | [Source Domains and Trajectory Breakdown](#source-domains-and-trajectory-breakdown) | `scripts/analysis/make_trajectory_accuracy_table.py` |
| Tier, parsing, confidence intervals; Appendix C.2–C.4 | [Appendix Analyses](#appendix-analyses) | `scripts/reproduce/06_appendix_analyses.sh` |
| Response bias, prompt sensitivity, cross-attribute, few-shot; Appendix C.5–C.7 | [Appendix Analyses](#appendix-analyses) | Generation commands below; `scripts/reproduce/06_appendix_analyses.sh` |
| Source domains; Table 15 | [Source Domains and Trajectory Breakdown](#source-domains-and-trajectory-breakdown) | `scripts/reproduce/08_source_domain_benchmark.sh` |
| Alternative formats; Tables 16–17 | [Query Rewrites and Alternative Formats](#query-rewrites-and-alternative-formats) | `scripts/reproduce/09_problem_format_binary.sh`, `10_problem_format_ternary.sh` |
| Paraphrases; Tables 12–13 | [Query Rewrites and Alternative Formats](#query-rewrites-and-alternative-formats) | `scripts/reproduce/11_question_paraphrases.sh` |
| Token positions and input order; Tables 18–19 | [Token Positions](#token-positions) | `scripts/reproduce/12_audio_then_query.sh`, `13_token_position_probes.sh` |
| Category alignment; Table 21, Figure 9 | [Direction Alignment](#direction-alignment) | `scripts/reproduce/14_category_direction_alignment.sh` |
| MELD transfer; Table 22 | [MELD Emotion Transfer](#meld-emotion-transfer) | `scripts/reproduce/15_meld_transfer.sh` |
| Scoring and aggregation | [Evaluation and Analysis](#evaluation-and-analysis) | `scripts/reproduce/16_aggregate_results.sh` |

## Main Benchmark

Paper target:
- Section 4.1-4.2
- Table 2: human and model accuracies on VocalGrad
- Appendix C.1-C.5: trajectory/tier analysis, parsing failures, confidence
  intervals, and output-distribution bias

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

The ten source signals are pure sine waves at 700–2,050 Hz in 150-Hz
increments, with peak amplitude 0.65. Each lasts 3.6 seconds: 0.75 seconds
of leading silence, 2.1 seconds of active signal, and 0.75 seconds of trailing
silence. The four categories use the speech benchmark's augmentation code and
intensity settings, with two directions, three tiers, and four trajectories
per source (240 clips per category).

All models use the same prompt as the main benchmark:
`Does the {attribute} increase or decrease over time?`
followed by `Answer with only one word: "increase" or "decrease".`
No system prompt is used.

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
- Figure 2 stage line plots:
  `outputs/analysis/linear_probe/_comparisons/default/stage_line_charts/`
- Additional heatmaps:
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
- Appendix C.11 / Table 20

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
  `outputs/analysis/vocalgrad_finetune_eval/_comparisons/<model_stem>/<run_name>/`
- Fine-tuned Kimi probing plots:
  `outputs/analysis/kimi_finetuned_probe_completed/<run_name>/`

## Appendix Analyses

Paper target:
- Appendix C.1-C.7

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

The source-clip response-bias rows in Appendix C.5 / Table 9 require raw source
clip generations. Generate them one backend at a time before running the
appendix aggregation:

```bash
.venv-kimi/bin/python scripts/run/evaluate_source_clips_generation_bias.py --backend kimia
.venv-mimo/bin/python scripts/run/evaluate_source_clips_generation_bias.py --backend mimoaudio
.venv-stepaudio/bin/python scripts/run/evaluate_source_clips_generation_bias.py --backend stepaudio2
.venv-af3/bin/python scripts/run/evaluate_source_clips_generation_bias.py --backend audioflamingo3
.venv/bin/python scripts/run/evaluate_source_clips_generation_bias.py --backend gemini
```

## Source Domains and Trajectory Breakdown

Follow [the alternate-source guide](../../data-and-annotation/docs/alternate_sources.md) to reconstruct the exact
100 source clips and recorded speech bounds for each alternate dataset. Place
the generated nine-category directories under `datasets/loquaciousset/test/`
and `datasets/commonvoice_spontaneous/test/`.

```bash
DATASET_VARIANT=loquaciousset bash scripts/reproduce/08_source_domain_benchmark.sh
DATASET_VARIANT=commonvoice_spontaneous bash scripts/reproduce/08_source_domain_benchmark.sh
.venv/bin/python scripts/analysis/make_source_domain_accuracy_table.py
.venv/bin/python scripts/analysis/make_trajectory_accuracy_table.py
```

Table 5 computes each human annotator's accuracy per category/trajectory,
then averages annotators and the nine categories equally. Models average
category-level trajectory accuracies equally, using `by_curve.accuracy` over all target clips with valid gold labels. Missing, unparsed, ambiguous and error responses are incorrect. This is
distinct from Table 16 macro-F1, which retains unparsed outputs as false negatives. The expected human results are
98.46, 93.73, 92.02 and 93.83 percent (jump, linear, first-flat, last-flat).

## Query Rewrites and Alternative Formats

```bash
bash scripts/reproduce/11_question_paraphrases.sh
.venv/bin/python scripts/analysis/make_paraphrase_prompt_metrics_table.py
bash scripts/reproduce/09_problem_format_binary.sh
bash scripts/reproduce/10_problem_format_ternary.sh
.venv/bin/python scripts/analysis/make_rebuttal_question_format_weighted_accuracy_table.py
.venv/bin/python scripts/analysis/make_rebuttal_question_prediction_distribution_table.py
```

Table 16 reports **macro-F1**. Per category, gold counts are
2400 yes / 100 no, or 1200 increase / 1200 decrease / 100 constant. Unparsed
answers remain false negatives for their gold class and are not an extra
class in the macro average. Source labels denote absence of an imposed
trajectory, not physically constant speech.

## Token Positions

```bash
bash scripts/reproduce/12_audio_then_query.sh
bash scripts/reproduce/13_token_position_probes.sh
```

Only the published shared-decoder stream is extracted. Audio endpoints a1–a4
and output-prediction position o are evaluated for Query→Audio. Audio→Query
additionally includes attribute-last, query-penultimate and query-last.
The published tables average nine category probes equally, at layers 1,10,25.

## Direction Alignment

Prepare the one-epoch all-linear adapters with `05_finetune_lora.sh` (set
`RUN_NAME=epoch1`). The alignment wrapper extracts base and both adapted test
features. For existing complete caches, use `SKIP_EXTRACTION=1`.

```bash
bash scripts/reproduce/14_category_direction_alignment.sh
# If all required test feature caches already exist:
SKIP_EXTRACTION=1 bash scripts/reproduce/14_category_direction_alignment.sh
```

The script handles base, Volume-tuned and Pitch-tuned Kimi only. All nine
categories use the final input token before answer generation. Pair up/down
clips with identical source, tier and curve; average the 12 differences per
source, then the 100 sources equally. Table 21 uses all 36 unordered category
pairs and indices 5,10,15,20,25,28. Figure 9 is signed cosine at index 28.
The embedding state (index 0) has undefined cosine for zero change directions.

Keep each feature manifest and exact checkpoint revision. For caches
with incomplete per-file provenance, input-ID/prompt matching cannot certify
the extraction-time adapter revision. Extract features with recorded adapter
paths to preserve their provenance.

## MELD Emotion Transfer

Use the audio-only test archive from `ajyy/MELD_audio` on Hugging Face:
`archive/test.tar.gz`. Extract it so `datasets/meld/test/` contains FLAC files.
Obtain the official MELD `test_sent_emo.csv` and put it in `datasets/meld/`.
The archive revision used for the reported experiment was not recorded;
preserve the revision and checksum when downloading. The label CSV used
for the reported experiment has SHA-256 `8d37103938f7067600839fe29d5a114a6cd1bcdafb75bec101e06464c5006888`.

```bash
bash scripts/reproduce/15_meld_transfer.sh
# Aggregate existing three-model raw predictions:
ANALYZE_ONLY=1 bash scripts/reproduce/15_meld_transfer.sh
```

The prompt lists seven options (A--G) and asks "Answer with only the option
letter." The mapping is A: anger, B: disgust, C: fear, D: joy, E: neutral,
F: sadness, G: surprise. Kimi receives user text followed by user audio,
with no system prompt.

Each candidate is the single token encoding an ASCII space followed by its
option letter: `" A"`, ..., `" G"`, with `bos=False, eos=False`. Tokenization
must produce exactly one token per candidate. At the final input position,
before answer generation, select the candidate with the highest logit and map
its letter to the emotion. Ties select the first candidate in A--G order.
The scoring rule is `argmax_L z_last[token_id(" " + L)]`.

The three conditions share 2,610 test IDs and gold labels. Report precision,
recall and F1 per class, and the equally weighted average of each metric over
the seven classes (zero-division=0). Expected macro-F1 percentages are
41.3, 37.7 and 34.2 for Base, Volume-tuned and Pitch-tuned, respectively.
Of the 1,228 clips predicted as Neutral by Base, Pitch-tuned predicts Joy for
317 (25.8%). The analysis reads `letter_choice_logits`, checks any stored
option-letter predictions against their argmax, and writes the per-class CSV,
Neutral-to-Joy counts and `emotion_logit_letter_precision_recall_f1.md`.

Raw predictions, feature caches, audio and checkpoints are not committed.
Small smoke runs can use existing wrappers' sample limits, but they do not
reproduce the full paper estimates. No cluster-specific PBS scripts are needed
for these portable entrypoints.

## Evaluation and Analysis

The direction parser uses the case-insensitive whole-word pattern
`\b(increase|decrease|increases|decreases)\b`. An answer is valid if it contains
exactly one direction; repeated words of that direction are allowed. Parsing
does not infer negation or meaning. Paraphrase experiments use their synonym
vocabulary and also require exactly one direction.

Accuracy uses all target clips with valid gold labels. Missing answers,
unmatched answers, answers containing both directions and runtime errors count
as incorrect. The same definition applies to category, curve, tier and direction
groups, and to direction recalls and balanced accuracy.

`accuracy` and `accuracy_all` report all-clip accuracy. `accuracy_parsed` is a
diagnostic metric. `total_evaluable` and `n_parsed` count parsed answers.
Table 9 response distributions and Table 13 increase ratios use parsed answers
as their denominator. The confusion matrix lists parsed predictions;
`unparsed_by_gold` records the remaining false negatives. Alternative-format
macro-F1, probes, alignment and MELD have their own protocols.

Run from `benchmarking-and-analysis/` after obtaining model predictions:

```bash
python3 -m plic.cli_summarize_vocalgrad \
  --input outputs/raw/vocalgrad/default/volume/kimi-audio.jsonl \
  --out outputs/analysis/vocalgrad/default/volume/kimi-audio.json
python3 scripts/analysis/make_trajectory_accuracy_table.py \
  --analysis-root outputs/analysis/vocalgrad/default \
  --models-only --out outputs/analysis/table5_models.csv
python3 scripts/analysis/make_paraphrase_prompt_metrics_table.py \
  --root outputs/analysis/rebuttal/paraphrase_questions/default \
  --output outputs/analysis/table13.md
python3 scripts/analysis/make_source_domain_accuracy_table.py \
  --vctk-analysis-root outputs/analysis/vocalgrad/default \
  --variant-root outputs/analysis/rebuttal/dataset_variants \
  --out outputs/analysis/table15.csv
.venv/bin/python scripts/analysis/plot_vocalgrad_cross_attribute_heatmaps.py \
  --root outputs/analysis/vocalgrad_cross_attribute/default \
  --out-dir outputs/analysis/figure8 --model kimi-audio --model mimo-audio \
  --categories audio_distortion audio_roughness background_noise echo speaking_speed voice_clarity voice_pitch voice_vibration volume
python3 -m unittest discover -s tests -p 'test_direction_evaluation.py'
```

The model inference wrappers write per-file summaries automatically. The first
command above summarizes one prediction file independently; install the package
or set `PYTHONPATH=src` when calling its modules directly.
Omit `--models-only` and supply `--annotation-root` to include the human row in
Table 5. Figure rendering requires the analysis plotting dependencies. With
`--categories`, it does not require local audio manifests.

For a batch of prediction files, supply a CSV manifest listing
`path,n_expected,parser`, with paths relative to the raw root. Parser identifiers
are `main-unique-direction-v1` and `synonym-unique-direction-v1`. Optional
`sha256`, `id_sha256` and metric columns enable reference validation.

```bash
bash scripts/reproduce/16_aggregate_results.sh \
  --raw-root outputs/raw \
  --out-root outputs/analysis-batch \
  --manifest /path/to/run_manifest.csv
```

The batch command checks expected counts, valid gold labels, unique clip IDs,
parser metadata and any supplied checksums or metrics. It writes summaries with
the same relative layout and an `aggregation_manifest.json` upon completion.
Use a new output directory. `tests/fixtures/direction_metrics.csv`
provides reference metrics for the recorded experiment outputs; file checksums
apply only to those exact inputs.

For `06_appendix_analyses.sh`, set `BENCHMARK_ANALYSIS_ROOT`, `PROMPT_ROOT`,
`CROSS_ATTRIBUTE_ROOT` and `FEWSHOT_ROOT` to the corresponding summary directories.
Fine-tuning comparison CSVs are generated before the four-model table;
`--comparison-root` selects their directory.

### Generation and scoring conditions

Direction generation retries once with the same prompt only when no whole-word
`increase(s)` or `decrease(s)` occurs. A response containing both directions does
not trigger a retry, but is counted as incorrect by final scoring. This retry rule applies to all models. Final
accuracy always requires exactly one direction and includes all target clips.

For the unmodified-source response-bias experiment, open models use 16 initial
and 32 retry tokens; Gemini uses 128 and 256. The source runner selects these
budgets by backend unless explicitly overridden. Table 9 uses parsed responses
as the denominator, including when generating from summary JSON.

The fine-tuning wrapper selects model/scope-specific run names matching the
four-model table entrypoint. Comparison files are isolated under
`outputs/analysis/vocalgrad_finetune_eval/_comparisons/<model-stem>/<run-name>/`.
Kimi defaults to all paper test categories; other models default to the training
category. Set `CATEGORIES` explicitly for additional evaluations. Custom run
names or comparison directories require corresponding table input overrides.

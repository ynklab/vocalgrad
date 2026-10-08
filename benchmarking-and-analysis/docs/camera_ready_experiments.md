# Camera-ready experiment map

These entrypoints reproduce the additions in the 36-page camera-ready draft.
Run commands from `benchmarking-and-analysis/`, using the existing backend and
analysis environments described in `environment_setup.md`. New inference
entrypoints use Kimi-Audio only. There is no Kimi system prompt.

| Paper | Entry point / analysis | Required inputs |
| --- | --- | --- |
| B.2, trajectory formulae | `../data-and-annotation/docs/augmentation_curves.md` and existing category generators | Existing augmentation configurations |
| C.1, Table 5 | `scripts/analysis/make_trajectory_accuracy_table.py` | Main prediction summaries and human annotation JSONLs |
| C.6, Tables 12–13 | `scripts/reproduce/11_question_paraphrases.sh`, `make_paraphrase_prompt_metrics_table.py` | VCTK test clips |
| C.8, Table 15 | `scripts/reproduce/08_source_domain_benchmark.sh`, `make_source_domain_accuracy_table.py` | Frozen alternate-source test clips |
| C.9, Tables 16–17 | `scripts/reproduce/09_problem_format_binary.sh`, `10_problem_format_ternary.sh`; `make_rebuttal_question_format_weighted_accuracy_table.py`, `make_rebuttal_question_prediction_distribution_table.py` | VCTK augmented test clips and original 100 source clips |
| C.10, Tables 18–19 | `scripts/reproduce/12_audio_then_query.sh`, `13_token_position_probes.sh` | VCTK train/test clips; shared decoder layers 1,10,25 |
| C.12, Table 21 / Figure 9 | `scripts/reproduce/14_category_direction_alignment.sh` | Base and one-epoch all-linear Volume/Pitch adapters and test clips |
| C.13, Table 22 | `scripts/reproduce/15_meld_transfer.sh` | MELD test audio/labels and the same Volume/Pitch adapters |
| Figures 3 / 4 right | Existing `plot_lm_text_layer_probe_lines.py`, `plot_selected_kimi_finetuned_lm_text_panels.py` | Existing layer-probe result JSONs |

## Original sources and trajectory breakdown

Follow `../data-and-annotation/docs/alternate_sources.md` to reconstruct the exact
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
category-level trajectory accuracies equally, using the existing summarizer's `by_curve.accuracy` on evaluable clips (unparsed outputs excluded). This is
distinct from Table 16 macro-F1, which retains unparsed outputs as false negatives. The expected human results are
98.46, 93.73, 92.02 and 93.83 percent (jump, linear, first-flat, last-flat).

## Query rewrites and alternative formats

```bash
bash scripts/reproduce/11_question_paraphrases.sh
.venv/bin/python scripts/analysis/make_paraphrase_prompt_metrics_table.py
bash scripts/reproduce/09_problem_format_binary.sh
bash scripts/reproduce/10_problem_format_ternary.sh
.venv/bin/python scripts/analysis/make_rebuttal_question_format_weighted_accuracy_table.py
.venv/bin/python scripts/analysis/make_rebuttal_question_prediction_distribution_table.py
```

The historical `weighted_accuracy` script name is retained, but the Table 16
function calculates **macro-F1**, not accuracy. Per category, gold counts are
2400 yes / 100 no, or 1200 increase / 1200 decrease / 100 constant. Unparsed
answers remain false negatives for their gold class and are not an extra
class in the macro average. Source labels denote absence of an imposed
trajectory, not physically constant speech.

## Token positions

```bash
bash scripts/reproduce/12_audio_then_query.sh
bash scripts/reproduce/13_token_position_probes.sh
```

Only the published shared-decoder stream is extracted. Audio endpoints a1–a4
and output-prediction position o are evaluated for Query→Audio. Audio→Query
additionally includes attribute-last, query-penultimate and query-last. MiMo
branch probes, query-first, and alternate-corpus probes are not included.
The published tables average nine category probes equally, at layers 1,10,25.

## Direction alignment

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

Keep each feature manifest and exact checkpoint revision. Historical caches
with incomplete per-file provenance can be analyzed, but input-ID/prompt
matching cannot certify their extraction-time adapter revision. Fresh
extraction with the recorded adapter paths is the reproducible route.

## MELD emotion transfer

Use the audio-only test archive from `ajyy/MELD_audio` on Hugging Face:
`archive/test.tar.gz`. Extract it so `datasets/meld/test/` contains FLAC files.
Obtain the official MELD `test_sent_emo.csv` and put it in `datasets/meld/`.
The archive revision used historically was not recorded; preserve the
revision and checksum when downloading. A checksum of the historical label
CSV is `8d37103938f7067600839fe29d5a114a6cd1bcdafb75bec101e06464c5006888`.

```bash
bash scripts/reproduce/15_meld_transfer.sh
# Aggregate existing three-model raw predictions:
ANALYZE_ONLY=1 bash scripts/reproduce/15_meld_transfer.sh
```

The exact historical prompt lists seven lettered options and asks for an
option letter; Table 22 uses the **label-word logits**, not generated text
or option-letter logits. Each space-prefixed label must tokenize as one token.
The order is anger, disgust, fear, joy, neutral, sadness, surprise. Scores use
last-position logits before generation. The three conditions share test IDs.
Sentiment, Speaking Speed adapters and other-model MELD evaluations are
excluded. Expected macro-F1 percentages: 41.7, 39.1, 30.5. The transition
summary includes the published base-Neutral → Pitch-tuned-Joy ratio.

Raw predictions, feature caches, audio and checkpoints are not committed.
Small smoke runs can use existing wrappers' sample limits, but they do not
reproduce the full paper estimates. No cluster-specific PBS scripts are needed
for these portable entrypoints.

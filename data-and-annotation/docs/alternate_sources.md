# Alternative source clips (Table 15)

The frozen selection manifests are in
`data/metadata/selection/alternate_sources/<dataset>/test_source_clips_100.csv`.
They record source IDs, audio paths, and detected onset/offset bounds used in
the experiment. Use these manifests for exact reproduction; new random
selection is a new dataset instance.

LoquaciousSet uses the shared official test partition accessible through the
`clean` configuration. The config name is not an acoustic clean/noisy label.
The 100 selected clips have 50 male and 50 female metadata labels and comprise
83 VoxPopuli and 17 Common Voice clips. Source selection shuffles candidates
within known male/female groups (unknown gender excluded). No corpus quota is
imposed. The original code's default test RNG uses seed 42 + 1; the historical
command-line overrides were not preserved. Frozen IDs avoid this uncertainty.

Common Voice uses Spontaneous Speech 4.0 (2026-06-12), with 20 clips each from
cdo, cgg, kbd, qxp and shi. Candidate audio files must exist and have metadata
duration strictly below 30 seconds; duplicate audio-file rows are deduplicated.
Speakers are partitioned by client_id before random clip sampling. The
selection uses a per-locale RNG initialized to 42 + locale index by default.
The published evaluation selection has no overlap with the separately chosen
training speakers; it is not the original corpus's predefined test split.
Gender is not balanced. Audio is decoded to 16-kHz mono PCM16 WAV.

Run from `data-and-annotation/`:

```bash
uv run python scripts/materialize_alternate_test_sources.py \
  --dataset loquaciousset \
  --manifest data/metadata/selection/alternate_sources/loquaciousset/test_source_clips_100.csv \
  --output-root data/selected/loquaciousset
uv run python scripts/materialize_alternate_test_sources.py \
  --dataset commonvoice_spontaneous \
  --manifest data/metadata/selection/alternate_sources/commonvoice_spontaneous/test_source_clips_100.csv \
  --source-root data/original/CommonVoice-Spontaneous \
  --output-root data/selected/commonvoice_spontaneous
uv run python scripts/generate_alternate_test_benchmarks.py \
  --source-csv data/selected/loquaciousset/test_source_clips_100.csv \
  --output-root data/processed/loquaciousset/test
uv run python scripts/generate_alternate_test_benchmarks.py \
  --source-csv data/selected/commonvoice_spontaneous/test_source_clips_100.csv \
  --output-root data/processed/commonvoice_spontaneous/test
```

Download the five Common Voice archives through Mozilla Data Collective and
extract the `sps-corpus-4.0-2026-06-12-<locale>` directories beneath the source
root. LoquaciousSet downloads use recorded source-row indices with an ID check;
if the upstream dataset changes, resolve the frozen clip ID rather than
silently substituting another sample.

Generators reuse the nine original category configurations: three tiers,
four curves, two directions, for 2400 clips per category (21600 total per
dataset). Recorded speech bounds are reused; no new onset fallback or clip
replacement is performed. Copy the resulting category directories to the
sibling runtime's `datasets/<dataset>/test/`. Selectors are included to explain
the original procedure, but exact reconstruction uses the frozen manifests.

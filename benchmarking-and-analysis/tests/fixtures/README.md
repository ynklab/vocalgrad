# Evaluation reference metrics

`camera_ready_direction_metrics.csv` contains reference metrics for 587 prediction
files / 1,365,600 rows: Main, prompt order/reference, trajectories, beep, few-shot,
cross-attribute, fine-tuning, paraphrases, alternate sources and audio-then-query.
Alternative-format macro-F1, probing, representation alignment and MELD use
other evaluation protocols.

Source runtime revision: `858e2aec46f9287d026f19ecc62aa1a7463a7936` (Miyabi).
Paths are relative to `outputs/raw`. `sha256` hashes the exact file bytes;
`id_sha256` hashes sorted clip IDs joined by a newline (no final newline).
`n_expected` is the required clip count per file (240 for beep, 2400 otherwise).
Accuracy and ratio columns are fractions, not percentages. Standard and
paraphrase parser identifiers are recorded separately.

Use this manifest with `scripts/reproduce/16_aggregate_results.sh` and the
corresponding prediction files. Raw predictions are not included. For a new
experiment, supply its own manifest and expected clip counts and IDs.

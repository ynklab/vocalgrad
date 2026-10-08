"""Reproduce Table 15 from original and alternate-source Kimi predictions."""

import argparse, json, csv
from pathlib import Path

CATS = [
    "speaking_speed",
    "voice_pitch",
    "volume",
    "audio_distortion",
    "audio_roughness",
    "background_noise",
    "echo",
    "voice_clarity",
    "voice_vibration",
]
p = argparse.ArgumentParser()
p.add_argument(
    "--vctk-analysis-root",
    type=Path,
    default=Path("outputs/analysis/vocalgrad/default"),
)
p.add_argument(
    "--variant-root",
    type=Path,
    default=Path("outputs/analysis/rebuttal/dataset_variants"),
)
p.add_argument(
    "--out",
    type=Path,
    default=Path("outputs/analysis/rebuttal/dataset_variants/source_accuracy.csv"),
)
a = p.parse_args()
rows = []
for dataset, root in [
    ("VCTK", a.vctk_analysis_root),
    ("LoquaciousSet", a.variant_root / "loquaciousset/benchmark"),
    ("CommonVoice", a.variant_root / "commonvoice_spontaneous/benchmark"),
]:
    scores = {}
    for category in CATS:
        payload = json.loads((root / category / "kimi-audio.json").read_text())
        if payload["overall"]["total_rows"] <= 0:
            raise ValueError("No evaluable clips")
        scores[category] = float(payload["overall"]["accuracy"])
    rows.append({"dataset": dataset, "average": sum(scores.values()) / 9, **scores})
a.out.parent.mkdir(parents=True, exist_ok=True)
with a.out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

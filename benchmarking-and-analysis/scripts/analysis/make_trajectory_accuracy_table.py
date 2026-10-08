"""Reproduce Table 5 by averaging trajectory accuracy equally across nine categories."""

import argparse, json, csv
from pathlib import Path
from statistics import mean
from make_human_model_accuracy_table import parse_direction_label, normalize_label

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
CURVES = ["jump", "linear", "quad_first_flat", "quad_last_flat"]
MODELS = [
    "gemini-3-flash",
    "kimi-audio",
    "mimo-audio",
    "step-audio-2-mini",
    "audioflamingo3",
]


def records(p):
    return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]


def curve(r):
    return (
        r.get("trajectory") or r.get("curve") or (r.get("metadata") or {}).get("curve")
    )


def accuracy(rs, c, model=False):
    selected = [r for r in rs if curve(r) == c]
    if not selected:
        raise ValueError(f"No samples for trajectory {c}")
    return sum(
        (
            parse_direction_label(r.get("raw_response"))
            == normalize_label(r.get("gold_label"))
        )
        if model
        else (r["is_correct"] is True)
        for r in selected
    ) / len(selected)


p = argparse.ArgumentParser()
p.add_argument(
    "--analysis-root", type=Path, default=Path("outputs/analysis/vocalgrad/default")
)
p.add_argument("--annotation-root", type=Path, default=Path("outputs/annotation_data"))
p.add_argument(
    "--out",
    type=Path,
    default=Path("outputs/analysis/vocalgrad/default/trajectory_accuracy.csv"),
)
p.add_argument("--models-only", action="store_true", help="Omit human row when annotations are unavailable")
a = p.parse_args()
rows = []
if not a.models_only:
    human = {c: [] for c in CURVES}
    for category in CATS:
        files = sorted(
            (a.annotation_root / category / "shared_annotation_50").glob(
                "annotator_*.jsonl"
            )
        )
        if not files:
            raise FileNotFoundError(f"No human files for {category}")
        rs = [records(f) for f in files]
        for c in CURVES:
            human[c].append(mean(accuracy(r, c) for r in rs))
    rows.append({"model": "Human Avg.", **{c: mean(human[c]) for c in CURVES}})
for model in MODELS:
    scores = {c: [] for c in CURVES}
    for category in CATS:
        payload = json.loads((a.analysis_root / category / f"{model}.json").read_text())
        for c in CURVES:
            scores[c].append(float(payload["by_curve"][c]["accuracy"]))
    rows.append({"model": model, **{c: mean(scores[c]) for c in CURVES}})
a.out.parent.mkdir(parents=True, exist_ok=True)
with a.out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

from pathlib import Path
import csv
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
p = parser.parse_args().root
r = list(csv.DictReader((p / "cosine_by_layer.csv").open()))
models = ["base", "volume", "voice_pitch"]
cats = [
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
mask = np.triu(np.ones((9, 9), bool), 1)
rows = []
for m in models:
    for l in sorted({int(x["layer"]) for x in r}):
        a = np.full((9, 9), np.nan)
        for x in r:
            if x["model"] == m and int(x["layer"]) == l:
                a[cats.index(x["category_a"]), cats.index(x["category_b"])] = float(
                    x["cosine"]
                )
        valid = np.isfinite(a).all()
        rows.append(
            dict(
                model=m,
                layer=l,
                mean_signed_cosine=float(a[mask].mean()),
                mean_absolute_cosine=float(np.abs(a[mask]).mean()),
            )
        )
with (p / "alignment_summary.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
for x in rows:
    if x["layer"] == 28:
        print(x)

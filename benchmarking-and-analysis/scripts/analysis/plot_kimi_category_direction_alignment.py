import csv
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
root = parser.parse_args().root
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
labels = [
    "speed",
    "pitch",
    "volume",
    "distortion",
    "roughness",
    "noise",
    "echo",
    "clarity",
    "vibration",
]
models = ["base", "volume", "voice_pitch"]
rows = list(csv.DictReader((root / "cosine_by_layer.csv").open()))
layers = sorted({int(x["layer"]) for x in rows})
x = np.zeros((3, len(layers), 9, 9))
for r in rows:
    if r["model"] not in models:
        continue
    x[
        models.index(r["model"]),
        layers.index(int(r["layer"])),
        cats.index(r["category_a"]),
        cats.index(r["category_b"]),
    ] = float(r["cosine"])
mask = np.triu(np.ones((9, 9), dtype=bool), 1)
fig, axs = plt.subplots(1, 3, figsize=(14, 5), layout="constrained")
for i, ax in enumerate(axs):
    im = ax.imshow(x[i, -1], vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_title(
        ["Base", "Volume-tuned", "Pitch-tuned"][i], fontsize=16, fontweight="bold"
    )
    ax.set_xticks(range(9), labels, rotation=90, fontsize=11, fontweight="bold")
    ax.set_yticks(range(9), labels, fontsize=11, fontweight="bold")
fig.colorbar(im, ax=axs, label="Signed cosine similarity")
fig.suptitle(
    f"Signed cosine similarity between category change directions (layer {layers[-1]})",
    fontsize=18,
    fontweight="bold",
)
fig.savefig(root / "final_layer_cosine.png", dpi=180)
fig.savefig(root / "final_layer_cosine.pdf")
plt.close(fig)

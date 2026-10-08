"""Paired source-balanced direction analysis of saved Kimi hidden states; no model loading."""

import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np

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
MODELS = ["base", "volume", "voice_pitch"]


def digest(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def cosine(d):
    norms = np.linalg.norm(d, axis=-1)
    u = np.divide(
        d,
        norms[..., None],
        out=np.full_like(d, np.nan, dtype=np.float64),
        where=norms[..., None] > 1e-12,
    )
    return np.einsum("...cd,...ed->...ce", u, u)


def write_csv(p, rows):
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    data = []
    reference = {}
    provenance = []
    sources_ref = None
    layers_ref = None
    for m in MODELS:
        root = a.repo / (
            "outputs/features/linear_probe_lm_text_layers/test/kimi-audio"
            if m == "base"
            else f"outputs/features/linear_probe_lm_text_layers_finetuned_kimi/test/kimi-audio-ft-{m}"
        )
        if not (root / "manifest.json").exists():
            raise ValueError(f"Missing manifest {root}")
        manifest = json.loads((root / "manifest.json").read_text())
        if m != "base" and f"/epoch1/all_linear/{m}/adapter" not in manifest.get(
            "adapter_path", ""
        ):
            raise ValueError("Unexpected adapter provenance")
        groups = []
        for c in CATS:
            p = root / f"{c}.npz"
            print("Loading", m, c, flush=True)
            with np.load(p, allow_pickle=False) as z:
                ids = z["benchmark_clip_id"].astype(str)
                labels = z["label"]
                prompts = z["prompt"].astype(str)
                layers = z["layer_index"]
                x = z["lm_text_layer_features"]
            if len(set(ids)) != len(ids):
                raise ValueError("Duplicate/truncated IDs")
            if not np.all(np.isfinite(x)):
                raise ValueError("Non-finite features")
            if layers_ref is None:
                layers_ref = layers.copy()
            if not np.array_equal(layers, layers_ref):
                raise ValueError("Layer mismatch")
            order = np.argsort(ids)
            sig = (
                ids[order].tolist(),
                labels[order].tolist(),
                prompts[order].tolist(),
                list(x.shape[1:]),
            )
            if m == "base":
                reference[c] = sig
            elif reference[c] != sig:
                raise ValueError(f"Cache mismatch {m}/{c}")
            paired = {}
            for i, s in enumerate(ids):
                key, direction = s.rsplit("__", 1)
                if direction not in ["up", "down"] or labels[i] != (
                    1 if direction == "up" else 0
                ):
                    raise ValueError("ID/label mismatch")
                paired.setdefault(key, {})[direction] = i
            by_source = {}
            for key, pair in paired.items():
                if set(pair) != {"up", "down"}:
                    raise ValueError("Unpaired sample")
                source = key.rsplit("__", 2)[0]
                by_source.setdefault(source, []).append((pair["up"], pair["down"]))
            sources = sorted(by_source)
            if sources_ref is None:
                sources_ref = sources
            if sources != sources_ref:
                raise ValueError("Source mismatch across categories/models")
            # Equal source weight; each source averages 3 intensities x 4 trajectories.
            source_d = []
            for s in sources:
                pairs = by_source[s]
                if len(pairs) != 12:
                    raise ValueError("Expected 12 direction pairs per source")
                up, down = zip(*pairs)
                source_d.append(
                    (
                        x[list(up)].astype(np.float64)
                        - x[list(down)].astype(np.float64)
                    ).mean(axis=0)
                )
            groups.append(np.stack(source_d).astype(np.float32))
            del x
            provenance.append(
                {
                    "model": m,
                    "category": c,
                    "path": str(p),
                    "sha256": digest(p),
                    "samples": len(ids),
                    "sources": len(sources),
                    "pairs": len(paired),
                    "shape": list(groups[-1].shape),
                    "manifest_sha256": digest(root / "manifest.json"),
                    "adapter_path": manifest.get("adapter_path"),
                    "manifest_lists_category": c in manifest.get("categories", []),
                }
            )
        data.append(np.stack(groups))
    # [model, category, source, layer, dimension]
    sd = np.stack(data)
    del data
    d = sd.mean(axis=2, dtype=np.float64)
    cos = cosine(d.transpose(0, 2, 1, 3))
    np.savez_compressed(
        a.out / "directions.npz",
        directions=d,
        source_directions=sd,
        cosine=cos,
        categories=CATS,
        models=MODELS,
        layers=layers_ref,
        sources=sources_ref,
    )
    rows = []
    for mi, m in enumerate(MODELS):
        for li, l in enumerate(layers_ref):
            for ci, c in enumerate(CATS):
                for ei, e in enumerate(CATS):
                    rows.append(
                        dict(
                            model=m,
                            layer=int(l),
                            category_a=c,
                            category_b=e,
                            cosine=float(cos[mi, li, ci, ei]),
                            delta_vs_base=float(
                                cos[mi, li, ci, ei] - cos[0, li, ci, ei]
                            ),
                        )
                    )
    write_csv(a.out / "cosine_by_layer.csv", rows)
    info = {
        "models": MODELS,
        "categories": CATS,
        "token_position": "last input token before answer generation",
        "input_order": "query then audio, as implemented by existing extractor; cached prompts compared exactly",
        "method": "mean paired increase-minus-decrease, equally average 12 pairs within each source then 100 sources",
        "interpretation": "Cosine measures directional alignment, not causal proof of shared representation. The analysis compares category directions within each model and does not establish causal shared representations.",
        "provenance": provenance,
    }
    (a.out / "provenance.json").write_text(json.dumps(info, indent=2))
    print("COMPLETE", a.out, flush=True)


if __name__ == "__main__":
    main()

"""Materialize the frozen Table 15 selection; preserve recorded onset/offset bounds."""

import argparse, csv, urllib.parse
from pathlib import Path
from select_loquaciousset_clips import fetch_json, download_file

p = argparse.ArgumentParser()
p.add_argument("--dataset", choices=["loquaciousset", "commonvoice_spontaneous"], required=True)
p.add_argument("--manifest", type=Path, required=True)
p.add_argument("--output-root", type=Path, required=True)
p.add_argument("--source-root", type=Path)
a = p.parse_args()
rows = list(csv.DictReader(a.manifest.open()))
if len(rows) != 100:
    raise ValueError("Expected 100 frozen test sources")
for row in rows:
    relative = Path(row["audio_path"])
    parts = relative.parts
    index = parts.index("audio")
    dest = a.output_root / "test" / Path(*parts[index:])
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        if a.dataset == "loquaciousset":
            params = urllib.parse.urlencode(
                {
                    "dataset": row["source_dataset"],
                    "config": row["source_config"],
                    "split": row["source_split"],
                    "offset": row["source_row_index"],
                    "length": 1,
                }
            )
            metadata = fetch_json("https://datasets-server.huggingface.co/rows?" + params)["rows"][
                0
            ]["row"]
            if str(metadata["ID"]) != row["clip_id"]:
                raise ValueError("Upstream row changed; resolve frozen clip ID before downloading")
            audio = metadata["wav"]
            url = audio[0]["src"]
            download_file(url, dest, overwrite=False)
        else:
            if a.source_root is None:
                raise ValueError(
                    "--source-root must contain the five downloaded Common Voice Spontaneous Speech corpora"
                )
            source = Path(row["source_audio_path"])
            parts = source.parts
            corpus = next(i for i, s in enumerate(parts) if s.startswith("sps-corpus-"))
            source = a.source_root / Path(*parts[corpus:])
            import librosa, soundfile as sf

            audio, _ = librosa.load(source, sr=16000, mono=True)
            sf.write(dest, audio, 16000, subtype="PCM_16")
    row["audio_path"] = str(dest)
a.output_root.mkdir(parents=True, exist_ok=True)
with (a.output_root / "test_source_clips_100.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

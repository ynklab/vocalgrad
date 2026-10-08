"""Reproduce Tables 18--19 from shared-decoder position-wise probe summaries."""

import argparse, json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
p.add_argument("--run-name", default="default")
p.add_argument("--out-dir", type=Path, required=True)
a = p.parse_args()
a.out_dir.mkdir(parents=True, exist_ok=True)
for order in ["query_to_audio", "audio_to_query"]:
    d = json.loads(
        (a.root / order / "kimi-audio" / a.run_name / "results.json").read_text()
    )
    columns = [("shared_audio", str(i), f"a{i + 1}") for i in range(4)]
    if order == "audio_to_query":
        columns += [
            ("text", "attribute_last", "q_attr"),
            ("text", "query_penultimate", "q_p"),
            ("text", "query_last", "q_l"),
        ]
    columns += [("text", "output", "o")]
    lines = [
        "| Layer | " + " | ".join(x[2] for x in columns) + " |",
        "|---:|" + "---:|" * len(columns),
    ]
    for layer in [1, 10, 25]:
        lines.append(
            "| "
            + str(layer)
            + " | "
            + " | ".join(
                f"{100 * d[stream][unit][str(layer)]['balanced_accuracy']:.2f}"
                for stream, unit, _ in columns
            )
            + " |"
        )
    (a.out_dir / f"{order}_table.md").write_text("\n".join(lines) + "\n")

"""Write a compact Markdown PRF table for base and fine-tuned Kimi on MELD emotion."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


MODELS = (
    ("kimi-audio", "Base Kimi"),
    ("kimi-audio-ft-volume", "Volume-tuned Kimi"),
    ("kimi-audio-ft-voice_pitch", "Pitch-tuned Kimi"),
)


def _cell(precision: float, recall: float, f1: float) -> str:
    return f"{precision * 100:.1f} / {recall * 100:.1f} / {f1 * 100:.1f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a MELD emotion logit-word precision/recall/F1 Markdown table."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "outputs/analysis/rebuttal/meld_mcq/finetune_prediction_changes/per_class_f1.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/analysis/rebuttal/meld_mcq/finetune_prediction_changes/emotion_logit_word_precision_recall_f1.md"
        ),
    )
    args = parser.parse_args()

    metrics: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    with args.input.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["task"] != "emotion" or row["model"] not in dict(MODELS):
                continue
            metrics[row["model"]][row["label"]] = {
                metric: float(row[metric]) for metric in ("precision", "recall", "f1")
            }

    labels = ("anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise")
    missing = [
        f"{model}:{label}"
        for model, _ in MODELS
        for label in labels
        if label not in metrics[model]
    ]
    if missing:
        raise ValueError(f"missing emotion metrics: {', '.join(missing)}")

    columns = ("Macro-average", *labels)
    lines = [
        "Cell format: precision / recall / F1 (%), using `logit_word` predictions.",
        "",
        "| Model | " + " | ".join(columns) + " |",
        "|---|" + "---:|" * len(columns),
    ]
    for model, display_name in MODELS:
        values = metrics[model]
        macro = {
            metric: sum(values[label][metric] for label in labels) / len(labels)
            for metric in ("precision", "recall", "f1")
        }
        cells = [_cell(**macro), *[_cell(**values[label]) for label in labels]]
        lines.append("| " + display_name + " | " + " | ".join(cells) + " |")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

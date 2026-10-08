"""Create a Markdown comparison of paraphrase-question result summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


QUESTIONS = {
    "volume": {
        "rise_fall": "Does the loudness rise or fall as the audio progresses?",
        "intensity_increase_decrease": "Does the sound intensity increase or decrease throughout the recording?",
        "intensity_increasing_decreasing": "Is the audio intensity increasing or decreasing over time?",
    },
    "voice_pitch": {
        "tone_rise_fall": "Does the tone rise or fall over time?",
        "frequency_upward_downward": "Does the sound frequency change upward or downward over time?",
        "frequency_rise_decline": "Over time, does the frequency of the voice rise or decline?",
    },
    "speaking_speed": {
        "pace_faster_slower": "Does the pace of speech become faster or slower over time?",
        "tempo_increase_decrease": "Does the speech tempo increase or decrease with time?",
        "delivery_accelerate_decelerate": "Does the pace of delivery accelerate or decelerate over time?",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write accuracy and predicted-increase ratios for each paraphrase prompt."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/analysis/rebuttal/paraphrase_questions/default"),
    )
    parser.add_argument(
        "--model",
        default="kimi-audio",
        help="Result filename stem to use within every prompt directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/analysis/rebuttal/paraphrase_questions/prompt_accuracy_increase_ratio.md"
        ),
    )
    args = parser.parse_args()

    lines = [
        "# Paraphrase question comparison",
        "",
        f"Model: {args.model}. Increase ratio is the fraction of all evaluable clips predicted as increase after normalizing each prompt's natural answer words.",
        "",
        "| Category | Query pattern | Accuracy | Increase ratio |",
        "|---|---|---:|---:|",
    ]
    for category, variants in QUESTIONS.items():
        for variant, question in variants.items():
            path = args.root / category / variant / f"{args.model}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            overall = payload["overall"]
            predicted_increase = sum(
                overall["confusion_matrix"]["table"]["pred_increase"].values()
            )
            total = int(overall["total_evaluable"])
            ratio = predicted_increase / total if total else 0.0
            lines.append(
                f"| {category} | {variant} | {float(overall['accuracy']) * 100:.2f}% | {ratio * 100:.2f}% |"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

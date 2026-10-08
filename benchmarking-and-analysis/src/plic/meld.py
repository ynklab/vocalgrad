from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


TASK_LABELS = {
    "emotion": ("anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"),
}


@dataclass(frozen=True)
class MeldSample:
    sample_id: str
    audio_path: Path
    emotion: str


def load_meld_test(dataset_root: Path) -> list[MeldSample]:
    csv_path = dataset_root / "test_sent_emo.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    samples: list[MeldSample] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            dialogue_id, utterance_id = row["Dialogue_ID"], row["Utterance_ID"]
            candidates = (
                dataset_root / "test" / f"dia{dialogue_id}_utt{utterance_id}.flac",
                dataset_root
                / "test"
                / f"final_videos_testdia{dialogue_id}_utt{utterance_id}.flac",
            )
            audio_path = next((path for path in candidates if path.exists()), None)
            if audio_path is None:
                raise FileNotFoundError(
                    f"no audio for dialogue={dialogue_id} utterance={utterance_id}"
                )
            samples.append(
                MeldSample(
                    f"dia{dialogue_id}_utt{utterance_id}", audio_path, row["Emotion"]
                )
            )
    return samples


def build_meld_prompt(task: str) -> tuple[str, dict[str, str]]:
    labels = TASK_LABELS[task]
    choices = {chr(ord("A") + idx): label for idx, label in enumerate(labels)}
    task_text = "emotion"
    options = "\n".join(f"{letter}. {label}" for letter, label in choices.items())
    return (
        f"What is the speaker's {task_text} in this audio?\n\nOptions:\n{options}\n\n"
        "Answer with only the option letter.",
        choices,
    )

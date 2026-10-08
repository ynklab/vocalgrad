"""Table 22 and the published Neutral-to-Joy prediction transition; option-letter logits."""

import argparse, csv, json, math
from pathlib import Path

LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
MODELS = ["kimi-audio", "kimi-audio-ft-volume", "kimi-audio-ft-voice_pitch"]


def read(p):
    rows = [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
    if not rows:
        raise ValueError("No MELD predictions")
    choices = dict(zip("ABCDEFG", LABELS))
    for r in rows:
        if r.get("choices") != choices:
            raise ValueError("Unexpected MELD option mapping")
        logits = r["letter_choice_logits"]
        if set(logits) != set(choices) or not all(math.isfinite(v) for v in logits.values()):
            raise ValueError("Invalid option-letter logits")
        # Resolve ties in the same A--G order as inference.
        letter = max(choices, key=lambda key: logits[key])
        prediction = choices[letter]
        for key in ("logit_letter_prediction", "logits_prediction"):
            if key in r and r[key] != prediction:
                raise ValueError("Stored prediction disagrees with option-letter logits")
        if r.get("letter_choice_argmax", letter) != letter:
            raise ValueError("Stored option disagrees with option-letter logits")
        if r.get("gold_label") not in LABELS:
            raise ValueError("Invalid MELD gold label")
        r["logit_letter_prediction"] = prediction
    out = {r["sample_id"]: r for r in rows}
    if len(out) != len(rows):
        raise ValueError("Duplicate MELD sample IDs")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path, default=Path("outputs/raw/rebuttal/meld_mcq"))
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path("outputs/analysis/rebuttal/meld_mcq/finetune_prediction_changes"),
    )
    a = p.parse_args()
    a.out_root.mkdir(parents=True, exist_ok=True)
    base = read(a.raw_root / "emotion" / f"{MODELS[0]}.jsonl")
    metrics = []
    transitions = {}
    for model in MODELS:
        rows = read(a.raw_root / "emotion" / f"{model}.jsonl")
        if set(rows) != set(base):
            raise ValueError("MELD model sample sets differ")
        for key, r in rows.items():
            if r["gold_label"] != base[key]["gold_label"]:
                raise ValueError("Gold labels differ")
            if r["logit_letter_prediction"] not in LABELS:
                raise ValueError("Invalid prediction")
        for label in LABELS:
            tp = sum(
                r["gold_label"] == label and r["logit_letter_prediction"] == label
                for r in rows.values()
            )
            fp = sum(
                r["gold_label"] != label and r["logit_letter_prediction"] == label
                for r in rows.values()
            )
            fn = sum(
                r["gold_label"] == label and r["logit_letter_prediction"] != label
                for r in rows.values()
            )
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
            metrics.append(
                dict(
                    task="emotion",
                    model=model,
                    label=label,
                    tp=tp,
                    fp=fp,
                    fn=fn,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                )
            )
        if model != MODELS[0]:
            selected = [
                key for key, r in base.items() if r["logit_letter_prediction"] == "neutral"
            ]
            n = sum(rows[key]["logit_letter_prediction"] == "joy" for key in selected)
            transitions[model] = {
                "base_neutral_count": len(selected),
                "changed_to_joy": n,
                "ratio": n / len(selected) if selected else None,
            }
    with (a.out_root / "per_class_f1.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics[0]))
        w.writeheader()
        w.writerows(metrics)
    (a.out_root / "neutral_to_joy.json").write_text(json.dumps(transitions, indent=2))


if __name__ == "__main__":
    main()

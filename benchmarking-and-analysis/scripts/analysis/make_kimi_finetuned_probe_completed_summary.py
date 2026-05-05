from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


LINEAR_ROOT = Path("outputs/analysis/linear_probe_finetuned_kimi")
LM_TEXT_ROOT = Path("outputs/analysis/linear_probe_lm_text_layers_finetuned_kimi")
DEFAULT_OUT_ROOT = Path("outputs/analysis/kimi_finetuned_probe_completed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize completed finetuned-Kimi linear-probe and lm_text layer-probe runs."
    )
    parser.add_argument("--run-name", default="epoch1")
    parser.add_argument("--linear-root", type=Path, default=LINEAR_ROOT)
    parser.add_argument("--lm-text-root", type=Path, default=LM_TEXT_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    return parser


def _train_category_from_model_stem(model_stem: str) -> str:
    prefix = "kimi-audio-ft-"
    return model_stem[len(prefix) :] if model_stem.startswith(prefix) else model_stem


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_float(value: float) -> str:
    return f"{value:.4f}"


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(rows: list[dict[str, object]], headers: list[tuple[str, str]]) -> str:
    header_line = "| " + " | ".join(label for _, label in headers) + " |"
    sep_line = "| " + " | ".join("---" for _ in headers) + " |"
    body_lines = []
    for row in rows:
        body_lines.append("| " + " | ".join(str(row[key]) for key, _ in headers) + " |")
    return "\n".join([header_line, sep_line, *body_lines])


def collect_linear_rows(root: Path, run_name: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("_"):
            continue
        results_path = model_dir / run_name / "results.json"
        if not results_path.exists():
            continue
        payload = _read_json(results_path)
        stages = payload.get("stages", {})
        if not isinstance(stages, dict):
            continue
        balanced = {
            stage: float(section["balanced_accuracy"])
            for stage, section in stages.items()
            if isinstance(section, dict) and isinstance(section.get("balanced_accuracy"), (int, float))
        }
        if not balanced:
            continue
        best_stage = max(balanced.items(), key=lambda item: item[1])[0]
        rows.append(
            {
                "train_category": _train_category_from_model_stem(model_dir.name),
                "probe_type": "linear_probe",
                "model_stem": model_dir.name,
                "best_stage": best_stage,
                "mel": _fmt_float(balanced.get("mel", float("nan"))),
                "pre_lm_audio": _fmt_float(balanced.get("pre_lm_audio", float("nan"))),
                "lm_audio": _fmt_float(balanced.get("lm_audio", float("nan"))),
                "lm_text": _fmt_float(balanced.get("lm_text", float("nan"))),
                "results_path": str(results_path),
            }
        )
    return rows


def collect_lm_text_rows(root: Path, run_name: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("_"):
            continue
        results_path = model_dir / run_name / "results.json"
        if not results_path.exists():
            continue
        payload = _read_json(results_path)
        layers = payload.get("layers", {})
        if not isinstance(layers, dict):
            continue
        balanced = {
            int(layer): float(section["balanced_accuracy"])
            for layer, section in layers.items()
            if str(layer).isdigit()
            and isinstance(section, dict)
            and isinstance(section.get("balanced_accuracy"), (int, float))
        }
        if not balanced:
            continue
        best_layer, best_value = max(balanced.items(), key=lambda item: item[1])
        rows.append(
            {
                "train_category": _train_category_from_model_stem(model_dir.name),
                "probe_type": "lm_text_layer_probe",
                "model_stem": model_dir.name,
                "best_layer": best_layer,
                "best_balanced_accuracy": _fmt_float(best_value),
                "final_layer": max(balanced),
                "final_layer_balanced_accuracy": _fmt_float(
                    balanced[max(balanced)]
                ),
                "results_path": str(results_path),
            }
        )
    return rows


def build_case_rows(
    linear_rows: list[dict[str, object]],
    lm_text_rows: list[dict[str, object]],
    out_dir: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    linprobe_heatmap = out_dir / "linprobe_heatmaps" / "balanced_accuracy.png"
    lmtext_line = out_dir / "lm_text_layer_lines" / "balanced_accuracy_by_layer.png"
    for row in linear_rows:
        rows.append(
            {
                "train_category": row["train_category"],
                "probe_type": row["probe_type"],
                "main_summary": f"best_stage={row['best_stage']}, lm_text={row['lm_text']}",
                "main_figure": str(linprobe_heatmap),
                "results_path": row["results_path"],
            }
        )
    for row in lm_text_rows:
        rows.append(
            {
                "train_category": row["train_category"],
                "probe_type": row["probe_type"],
                "main_summary": (
                    f"best_layer={row['best_layer']}, "
                    f"best_bal_acc={row['best_balanced_accuracy']}"
                ),
                "main_figure": str(lmtext_line),
                "results_path": row["results_path"],
            }
        )
    rows.sort(key=lambda item: (str(item["train_category"]), str(item["probe_type"])))
    return rows


def write_markdown(
    path: Path,
    case_rows: list[dict[str, object]],
    linear_rows: list[dict[str, object]],
    lm_text_rows: list[dict[str, object]],
    run_name: str,
) -> None:
    lines: list[str] = []
    lines.append("# Finetuned Kimi Probe Summary")
    lines.append("")
    lines.append(f"Run name: `{run_name}`")
    lines.append("")
    lines.append("## Completed Cases")
    lines.append("")
    lines.append(
        _markdown_table(
            case_rows,
            [
                ("train_category", "Train Category"),
                ("probe_type", "Probe"),
                ("main_summary", "Main Summary"),
                ("main_figure", "Main Figure"),
                ("results_path", "Results"),
            ],
        )
    )
    lines.append("")
    lines.append("## Linear Probe")
    lines.append("")
    if linear_rows:
        lines.append(
            _markdown_table(
                linear_rows,
                [
                    ("train_category", "Train Category"),
                    ("model_stem", "Model"),
                    ("best_stage", "Best Stage"),
                    ("mel", "mel"),
                    ("pre_lm_audio", "pre_lm_audio"),
                    ("lm_audio", "lm_audio"),
                    ("lm_text", "lm_text"),
                ],
            )
        )
    else:
        lines.append("No completed linear-probe runs found.")
    lines.append("")
    lines.append("## LM-Text Layer Probe")
    lines.append("")
    if lm_text_rows:
        lines.append(
            _markdown_table(
                lm_text_rows,
                [
                    ("train_category", "Train Category"),
                    ("model_stem", "Model"),
                    ("best_layer", "Best Layer"),
                    ("best_balanced_accuracy", "Best Balanced Accuracy"),
                    ("final_layer", "Final Layer"),
                    ("final_layer_balanced_accuracy", "Final Layer Balanced Accuracy"),
                ],
            )
        )
    else:
        lines.append("No completed lm_text layer-probe runs found.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    out_dir = args.out_root / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    linear_rows = collect_linear_rows(args.linear_root, args.run_name)
    lm_text_rows = collect_lm_text_rows(args.lm_text_root, args.run_name)
    case_rows = build_case_rows(linear_rows, lm_text_rows, out_dir)

    _write_csv(
        out_dir / "completed_cases.csv",
        case_rows,
        ["train_category", "probe_type", "main_summary", "main_figure", "results_path"],
    )
    _write_csv(
        out_dir / "linear_probe_summary.csv",
        linear_rows,
        ["train_category", "probe_type", "model_stem", "best_stage", "mel", "pre_lm_audio", "lm_audio", "lm_text", "results_path"],
    )
    _write_csv(
        out_dir / "lm_text_layer_probe_summary.csv",
        lm_text_rows,
        [
            "train_category",
            "probe_type",
            "model_stem",
            "best_layer",
            "best_balanced_accuracy",
            "final_layer",
            "final_layer_balanced_accuracy",
            "results_path",
        ],
    )
    write_markdown(
        out_dir / "summary.md",
        case_rows,
        linear_rows,
        lm_text_rows,
        args.run_name,
    )
    print(f"wrote completed finetuned-Kimi probe summary to {out_dir}")


if __name__ == "__main__":
    main()

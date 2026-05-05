from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

BASE = Path('outputs/analysis/vocalgrad')
VARIANTS = ['default', 'swap-order', 'audio-ref']
OUT = BASE / 'prompt_sensitivity_summary.md'


def load_variant(root: Path) -> dict[str, dict[str, float]]:
    data: dict[str, dict[str, float]] = {}
    for category_dir in sorted(root.iterdir()):
        if not category_dir.is_dir() or category_dir.name.startswith('_'):
            continue
        category = category_dir.name
        data[category] = {}
        for json_file in sorted(category_dir.glob('*.json')):
            try:
                obj = json.loads(json_file.read_text(encoding='utf-8'))
            except Exception:
                continue
            overall = obj.get('overall')
            if not isinstance(overall, dict):
                continue
            acc = overall.get('accuracy')
            if isinstance(acc, (int, float)):
                data[category][json_file.stem] = float(acc)
    return data


def all_models(variant_data: dict[str, dict[str, dict[str, float]]]) -> list[str]:
    models = set()
    for by_category in variant_data.values():
        for metrics in by_category.values():
            models.update(metrics)
    return sorted(models)


def variant_mean(by_category: dict[str, dict[str, float]], model: str) -> float | None:
    vals = [metrics[model] for metrics in by_category.values() if model in metrics]
    return mean(vals) if vals else None


def wins(by_category: dict[str, dict[str, float]], model: str) -> int:
    total = 0
    for metrics in by_category.values():
        if not metrics or model not in metrics:
            continue
        best = max(metrics.values())
        if metrics[model] == best:
            total += 1
    return total


def biggest_shift(default_data: dict[str, dict[str, float]], variant_data: dict[str, dict[str, float]], model: str) -> tuple[str, float] | None:
    best: tuple[str, float] | None = None
    for category, default_metrics in default_data.items():
        if model not in default_metrics:
            continue
        variant_metrics = variant_data.get(category, {})
        if model not in variant_metrics:
            continue
        delta = variant_metrics[model] - default_metrics[model]
        if best is None or abs(delta) > abs(best[1]):
            best = (category, delta)
    return best


def fmt(value: float | None) -> str:
    if value is None:
        return '-'
    return f'{value:.4f}'


def fmt_signed(value: float | None) -> str:
    if value is None:
        return '-'
    return f'{value:+.4f}'


def table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---'] * len(headers)) + ' |',
    ]
    for row in rows:
        lines.append('| ' + ' | '.join(row) + ' |')
    return lines


def main() -> None:
    variant_data = {variant: load_variant(BASE / variant) for variant in VARIANTS}
    models = all_models(variant_data)
    mean_rows: list[list[str]] = []
    best_variant_rows: list[list[str]] = []
    shift_rows: list[list[str]] = []

    for model in models:
        default_mean = variant_mean(variant_data['default'], model)
        swap_mean = variant_mean(variant_data['swap-order'], model)
        audio_ref_mean = variant_mean(variant_data['audio-ref'], model)
        mean_rows.append([
            model,
            fmt(default_mean),
            fmt(swap_mean),
            fmt_signed(None if default_mean is None or swap_mean is None else swap_mean - default_mean),
            fmt(audio_ref_mean),
            fmt_signed(None if default_mean is None or audio_ref_mean is None else audio_ref_mean - default_mean),
            str(wins(variant_data['default'], model)),
            str(wins(variant_data['swap-order'], model)),
            str(wins(variant_data['audio-ref'], model)),
        ])

        candidates = [(variant, variant_mean(variant_data[variant], model)) for variant in VARIANTS]
        candidates = [(variant, value) for variant, value in candidates if value is not None]
        if candidates:
            best_variant, best_value = max(candidates, key=lambda item: item[1])
            worst_variant, worst_value = min(candidates, key=lambda item: item[1])
            best_variant_rows.append([
                model,
                best_variant,
                fmt(best_value),
                worst_variant,
                fmt(worst_value),
                fmt_signed(best_value - worst_value),
            ])

        for variant in ['swap-order', 'audio-ref']:
            shift = biggest_shift(variant_data['default'], variant_data[variant], model)
            if shift is not None:
                category, delta = shift
                shift_rows.append([model, variant, category, fmt_signed(delta)])

    leader_rows: list[list[str]] = []
    categories = sorted(variant_data['default'])
    for category in categories:
        row = [category]
        for variant in VARIANTS:
            metrics = variant_data[variant].get(category, {})
            if not metrics:
                row.append('-')
                continue
            best_model, best_value = max(metrics.items(), key=lambda item: item[1])
            row.append(f'{best_model} ({best_value:.4f})')
        leader_rows.append(row)

    lines = [
        '# VocalGrad Prompt Sensitivity Summary',
        '',
        'Comparison across the three prompt settings currently available under `outputs/analysis/vocalgrad`.',
        '',
        '- `default`: original wording',
        '- `swap-order`: asks `decrease or increase`',
        '- `audio-ref`: explicitly says `of this audio clip`',
        '',
        '## Mean Accuracy By Variant',
        '',
    ]
    lines.extend(table(
        ['model', 'default', 'swap-order', 'delta_vs_default', 'audio-ref', 'delta_vs_default', 'wins_default', 'wins_swap', 'wins_audio_ref'],
        mean_rows,
    ))
    lines.extend([
        '',
        '## Best And Worst Variant Per Model',
        '',
    ])
    lines.extend(table(
        ['model', 'best_variant', 'best_mean_accuracy', 'worst_variant', 'worst_mean_accuracy', 'gap'],
        best_variant_rows,
    ))
    lines.extend([
        '',
        '## Largest Category Shift Vs Default',
        '',
    ])
    lines.extend(table(
        ['model', 'variant', 'category', 'delta_vs_default'],
        shift_rows,
    ))
    lines.extend([
        '',
        '## Category Leaders By Variant',
        '',
    ])
    lines.extend(table(
        ['category', 'default', 'swap-order', 'audio-ref'],
        leader_rows,
    ))

    OUT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()

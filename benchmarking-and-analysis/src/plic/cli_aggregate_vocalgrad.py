"""Aggregate binary-direction predictions without loading models."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
from .cli_summarize_vocalgrad import summarize_rows
from .direction_evaluation import read_rows, validate_rows, SCHEMA_VERSION


def digest_ids(rows):
    ids = [r.get("benchmark_clip_id", r.get("sample_id")) for r in rows]
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True,
                   help="CSV listing path (relative to raw-root), n_expected, parser; optional sha256/id_sha256/expected metrics")
    args = p.parse_args()
    if args.out_root.exists():
        raise FileExistsError(f"Use a new output directory; preserving {args.out_root}")
    with args.manifest.open() as f:
        entries = list(csv.DictReader(f))
    if not entries or len({e["path"] for e in entries}) != len(entries):
        raise ValueError("Manifest must contain unique prediction files")
    seen_outputs = set()
    for entry in entries:
        rel = Path(entry["path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"Expected a relative path: {rel}")
        output = rel.with_suffix(".json")
        if output in seen_outputs:
            raise ValueError(f"Output path collision: {output}")
        seen_outputs.add(output)
    checked = []
    for entry in entries:
        rel = Path(entry["path"])
        path = args.raw_root / rel
        raw_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if entry.get("sha256") and raw_sha != entry["sha256"]:
            raise ValueError(f"Raw checksum mismatch: {rel}")
        rows = read_rows(path)
        expected = int(entry["n_expected"])
        validate_rows(rows, expected_count=expected, require_ids=True)
        ids_sha = digest_ids(rows)
        if entry.get("id_sha256") and ids_sha != entry["id_sha256"]:
            raise ValueError(f"Clip ID set mismatch: {rel}")
        parser = entry["parser"]
        if parser not in {"main-unique-direction-v1", "synonym-unique-direction-v1"}:
            raise ValueError(f"Unknown parser: {parser}")
        if any(bool(r.get("answer_words")) != (parser == "synonym-unique-direction-v1") for r in rows):
            raise ValueError(f"Parser/row metadata mismatch: {rel}")
        summary = summarize_rows(rows, str(path))
        summary.update(n_expected=expected, n_rows=len(rows), sha256=raw_sha, id_sha256=ids_sha)
        summary["overall"]["n_expected"] = expected
        for key in ("accuracy", "n_parsed", "n_unparsed", "n_correct", "n_ambiguous", "n_error", "balanced_accuracy", "increase_ratio_parsed"):
            if entry.get(key) and abs(summary["overall"][key] - float(entry[key])) > 1e-12:
                raise ValueError(f"Reference mismatch: {rel}: {key}")
        out = args.out_root / rel.with_suffix(".json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2) + "\n")
        checked.append(dict(path=str(rel), n_rows=len(rows), sha256=raw_sha, id_sha256=ids_sha, parser=parser))
    report = dict(schema_version=SCHEMA_VERSION, files=len(checked), rows=sum(r["n_rows"] for r in checked), manifest=str(args.manifest), inputs=checked)
    (args.out_root / "aggregation_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Validated and aggregated {report['files']} files / {report['rows']} rows into {args.out_root}")


if __name__ == "__main__":
    main()

"""Generate the nine-category test grid from frozen alternative-source selections."""

import argparse, csv, subprocess, sys
from pathlib import Path
from generate_all_benchmarks import ACTIVE_CATEGORY_SPECS

p = argparse.ArgumentParser()
p.add_argument("--source-csv", type=Path, required=True)
p.add_argument("--output-root", type=Path, required=True)
p.add_argument("--limit", type=int)
p.add_argument("--overwrite", action="store_true")
a = p.parse_args()
rows = list(csv.DictReader(a.source_csv.open()))
for r in rows:
    if r.get("onset_detected", "").lower() not in ["true", "1"] or not 0 <= float(
        r["onset_sec"]
    ) < float(r["offset_sec"]):
        raise ValueError("Frozen source lacks valid detected speech bounds")
for category, script, config in ACTIVE_CATEGORY_SPECS:
    cmd = [
        sys.executable,
        str(script),
        "--source-csv",
        str(a.source_csv),
        "--config",
        str(config),
        "--output-root",
        str(a.output_root / category),
    ]
    if a.limit is not None:
        cmd += ["--limit", str(a.limit)]
    if a.overwrite:
        cmd += ["--overwrite"]
    subprocess.run(cmd, check=True)

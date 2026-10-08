#!/usr/bin/env bash
# CPU-only; writes a new tree and does not overwrite existing summaries.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
"${ANALYSIS_PYTHON_BIN:-python3}" -m plic.cli_aggregate_vocalgrad "$@"

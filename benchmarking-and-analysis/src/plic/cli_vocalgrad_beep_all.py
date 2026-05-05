from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from .cli_vocalgrad_all import build_parser, run_from_args


DEFAULT_ABLATION_BEEP_DATASET_ROOT = Path("datasets/vocalgrad/ablation_beep")
DEFAULT_ABLATION_BEEP_RAW_ROOT = Path("outputs/raw/vocalgrad_ablation_beep/default")
DEFAULT_ABLATION_BEEP_ANALYSIS_ROOT = Path("outputs/analysis/vocalgrad_ablation_beep/default")


def main() -> None:
    load_dotenv()
    args = build_parser(
        description=(
            "Run VocalGrad evaluation across all ablation-beep categories in one command."
        ),
        default_dataset_root=DEFAULT_ABLATION_BEEP_DATASET_ROOT,
        default_raw_root=DEFAULT_ABLATION_BEEP_RAW_ROOT,
        default_analysis_root=DEFAULT_ABLATION_BEEP_ANALYSIS_ROOT,
    ).parse_args()
    run_from_args(args)


if __name__ == "__main__":
    main()

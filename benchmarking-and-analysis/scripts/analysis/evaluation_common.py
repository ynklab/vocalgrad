"""Shared evaluation imports for scripts run directly from a checkout."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from plic.direction_evaluation import parse_direction_label, prediction_for_row

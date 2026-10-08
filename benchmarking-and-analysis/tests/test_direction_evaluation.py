import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from plic.direction_evaluation import (parse_direction_label, parse_paraphrase_label,
    binary_metrics, validate_rows)
from plic.cli_summarize_vocalgrad import summarize_rows
from plic.kimi_finetune import summarize_direction_rows, summarize_prediction_file


def row(raw, gold="increase", **extra):
    return dict(raw_response=raw, gold_label=gold, metadata=dict(curve="linear", tier="low", direction=gold), **extra)


class DirectionEvaluationTest(unittest.TestCase):
    def test_whole_words_and_unique_direction(self):
        for raw, expected in [("INCREASES.", "increase"), ("decreases", "decrease"),
                              ("increase increases", "increase"), ("increase or decrease", None),
                              ("increased", None), ("decreased", None), ("increasing", None),
                              ("preincrease", None), (None, None), ("", None),
                              ("does not increase", "increase")]:
            with self.subTest(raw=raw):
                self.assertEqual(parse_direction_label(raw), expected)

    def test_paraphrase_vocabulary_and_ambiguity(self):
        self.assertEqual(parse_paraphrase_label("accelerating"), "increase")
        self.assertEqual(parse_paraphrase_label("decline"), "decrease")
        self.assertIsNone(parse_paraphrase_label("rise then fall"))
        self.assertIsNone(parse_paraphrase_label("decelerated"))
        m = binary_metrics([row("rises", answer_words={"increase": "rise", "decrease": "fall"})])
        self.assertEqual(m["accuracy"], 1)

    def test_all_clips_metrics_use_raw_response(self):
        rows = [row("increase", prediction_label="decrease"), row(None, prediction_label="increase")]
        summary = summarize_rows(rows, "fixture")
        for m in [summary["overall"], summary["by_tier"]["low"], summary["by_curve"]["linear"], summary["by_direction"]["increase"]]:
            self.assertEqual(m["accuracy"], .5)
            self.assertEqual(m["accuracy_parsed"], 1)
            self.assertEqual(m["recall_increase"], .5)
            self.assertEqual(m["n_unparsed"], 1)
        self.assertEqual(summarize_direction_rows(rows).accuracy, .5)
        with tempfile.TemporaryDirectory() as d:
            src, out = Path(d)/"raw.jsonl", Path(d)/"out.json"
            src.write_text("\n".join(json.dumps(r) for r in rows))
            self.assertEqual(summarize_prediction_file(src, out)["overall"]["accuracy"], .5)

    def test_one_direction_failures_and_errors(self):
        m = binary_metrics([row("increase"), row("decrease", "decrease", error="runtime error")])
        self.assertEqual(m["balanced_accuracy"], .5)
        self.assertEqual(m["accuracy_down"], 0)
        self.assertEqual(m["n_error"], 1)
        self.assertEqual(m["accuracy_parsed"], 1)

    def test_invalid_inputs(self):
        for rows, options in [([row("increase", "constant")], {}),
                              ([row("increase", benchmark_clip_id="a")]*2, {}),
                              ([row("increase")], {"expected_count":2}),
                              ([row("increase")], {"require_ids":True})]:
            with self.assertRaises(ValueError):
                validate_rows(rows, **options)

    def test_alternative_format_macro_f1_keeps_unparsed_fn(self):
        spec = importlib.util.spec_from_file_location("formats", ROOT / "scripts/analysis/make_rebuttal_question_format_weighted_accuracy_table.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)/"volume"; folder.mkdir()
            (folder/"kimi.json").write_text(json.dumps({"category":"volume", "gold_label_counts":{"yes":2,"no":1}, "confusion_matrix":{"table":{"yes":{"yes":1,"no":0},"no":{"yes":0,"no":1}}}}))
            self.assertAlmostEqual(module._macro_f1_by_augmentation(Path(d))["volume"], 5/6)


if __name__ == "__main__":
    unittest.main()

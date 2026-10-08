"""CPU-only checks for MELD option-letter scoring and saved-logit analysis."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run = load("meld_runner", "scripts/run/evaluate_meld_mcq.py")
analysis = load("meld_analysis", "scripts/analysis/analyze_meld_emotion_transfer.py")


class MeldOptionLetterTest(unittest.TestCase):
    def test_candidates_have_ascii_space_and_no_special_tokens(self):
        seen = []
        class Tokenizer:
            def encode(self, text, **kwargs):
                seen.append((text, kwargs))
                return [100 + ord(text[-1])]
        runtime = SimpleNamespace(model=SimpleNamespace(prompt_manager=SimpleNamespace(text_tokenizer=Tokenizer())), backend="kimia")
        ids = run._candidate_ids(runtime, dict(zip("ABCDEFG", "ABCDEFG")), kind="option letter")
        self.assertEqual(list(ids), list("ABCDEFG"))
        self.assertEqual(seen, [(" " + c, {"bos": False, "eos": False}) for c in "ABCDEFG"])
        runtime.model.prompt_manager.text_tokenizer.encode = lambda *args, **kwargs: [1, 2]
        with self.assertRaises(RuntimeError):
            run._candidate_ids(runtime, {"A": "A"}, kind="option letter")

    def test_saved_logits_ties_and_invalid_inputs(self):
        row = dict(sample_id="a", gold_label="anger", choices=dict(zip("ABCDEFG", analysis.LABELS)),
                   letter_choice_logits={c: 1.0 for c in "ABCDEFG"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_text(json.dumps(row) + "\n")
            self.assertEqual(analysis.read(path)["a"]["logit_letter_prediction"], "anger")
            for changes in [{"logit_letter_prediction": "joy"}, {"gold_label": "invalid"},
                            {"letter_choice_logits": {c: float("nan") for c in "ABCDEFG"}},
                            {"letter_choice_argmax": "G"}]:
                path.write_text(json.dumps(dict(row, **changes)) + "\n")
                with self.assertRaises(ValueError):
                    analysis.read(path)
            path.write_text((json.dumps(row) + "\n") * 2)
            with self.assertRaises(ValueError):
                analysis.read(path)


if __name__ == "__main__":
    unittest.main()

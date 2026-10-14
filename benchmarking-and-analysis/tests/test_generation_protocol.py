import asyncio
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import json
import unittest
from unittest.mock import Mock, AsyncMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts/analysis'))
from plic.cli_vocalgrad import maybe_retry_same_prompt, maybe_retry_same_prompt_async
from plic.direction_evaluation import parse_direction_label
from make_original_source_increase_response_table import collect_source_rates


class GenerationProtocolTest(unittest.TestCase):
    def test_retry_and_scoring_are_separate_sync_async(self):
        for raw, retry in [('increase or decrease', False), ('increased', True), ('increase', False)]:
            for asynchronous in (False, True):
                runtime = SimpleNamespace(
                    generate_text=Mock(return_value='decrease'),
                    generate_text_async=AsyncMock(return_value='decrease'))
                kwargs = dict(runtime=runtime, sample=SimpleNamespace(audio_path=Path('clip.wav')),
                    raw_text=raw, retry_max_new_tokens=256, temperature=0, top_k=5, prompt='direction?')
                if asynchronous:
                    result = asyncio.run(maybe_retry_same_prompt_async(**kwargs))
                    calls = runtime.generate_text_async.call_count
                else:
                    result = maybe_retry_same_prompt(**kwargs)
                    calls = runtime.generate_text.call_count
                self.assertEqual(calls, int(retry))
                self.assertEqual(result, ('decrease', 'decrease') if retry else (parse_direction_label(raw), raw))

    def test_source_summary_uses_parsed_denominator(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, 'model.json').write_text(json.dumps({'category_summary': {
                'volume': {'preferred_increase_rate': .25, 'parsed_preferred_increase_rate': .5}}}))
            self.assertEqual(collect_source_rates(Path(d), set()), {'model': {'volume': .5}})

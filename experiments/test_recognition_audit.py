"""Run in the existing CCV retrieval environment; exercise the native parser."""
import json
from pathlib import Path
import tempfile
import unittest

from hipporag.rerank import DSPyFilter
from experiments.encoder_robustness import RecognitionAudit


class RecognitionTests(unittest.TestCase):
    def test_native_historical_replay_preserves_parser_result(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/malformed_recognition_20261002.json").read_text())
        native = DSPyFilter.__new__(DSPyFilter)
        native.llm_infer_fn = lambda **kwargs: (fixture["response"], {}, True)
        expected = native.parse_filter(fixture["response"])
        self.assertEqual(expected, fixture["native_expected"])
        with tempfile.TemporaryDirectory() as directory:
            audit = RecognitionAudit(native, Path(directory) / "audit.jsonl", policy="historical_replay")
            try:
                self.assertEqual(native.parse_filter(fixture["response"]), expected)
                audit.check()
                self.assertEqual(len(audit.failures), 1)
                self.assertEqual(next(iter(audit.responses.values()))["status"], "parse_failure")
            finally:
                audit.close()

    def test_fresh_recognition_still_rejects_malformed_response(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/malformed_recognition_20261002.json").read_text())
        native = DSPyFilter.__new__(DSPyFilter)
        native.llm_infer_fn = lambda **kwargs: (fixture["response"], {}, False)
        with tempfile.TemporaryDirectory() as directory:
            audit = RecognitionAudit(native, Path(directory) / "audit.jsonl", policy="strict")
            try:
                native.parse_filter(fixture["response"])
                with self.assertRaises(RuntimeError):
                    audit.check()
            finally:
                audit.close()

    def test_valid_empty_facts_are_not_parse_failure(self):
        native = DSPyFilter.__new__(DSPyFilter)
        native.llm_infer_fn = lambda **kwargs: None
        with tempfile.TemporaryDirectory() as directory:
            audit = RecognitionAudit(native, Path(directory) / "audit.jsonl")
            try:
                self.assertEqual(native.parse_filter('[[ ## fact_after_filter ## ]]\n{"fact": []}\n[[ ## completed ## ]]'), [])
                audit.check()
                self.assertEqual(audit.failures, [])
            finally:
                audit.close()


if __name__ == "__main__":
    unittest.main()

"""Read-only reporting tests using the native parser and captured response."""
import json
from pathlib import Path
import tempfile
import unittest

from experiments.audit_encoder_retrieval import audit_recognition_log


class RecognitionLogAuditTests(unittest.TestCase):
    def audit(self, response, policy, cached):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'recognition.jsonl'
            path.write_text(json.dumps(dict(request={}, response=[response, {}, cached])) + '\n')
            return audit_recognition_log(path, policy)

    def test_valid_empty_recognition_is_valid(self):
        result = self.audit('[[ ## fact_after_filter ## ]]\n{"fact": []}\n[[ ## completed ## ]]', 'strict', False)
        self.assertEqual(result['calls'], 1)
        self.assertEqual(list(result['responses'].values()), [dict(status='valid', facts=0)])

    def test_historical_malformed_response_is_explicitly_reported(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures/malformed_recognition_20261002.json').read_text())
        result = self.audit(fixture['response'], 'historical_replay', True)
        self.assertEqual(next(iter(result['responses'].values()))['status'], 'parse_failure')

    def test_fresh_malformed_response_is_rejected(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures/malformed_recognition_20261002.json').read_text())
        with self.assertRaisesRegex(RuntimeError, 'Malformed recognition'):
            self.audit(fixture['response'], 'strict', False)

    def test_uncached_historical_inference_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'uncached inference'):
            self.audit('unused', 'historical_replay', False)


if __name__ == '__main__':
    unittest.main()

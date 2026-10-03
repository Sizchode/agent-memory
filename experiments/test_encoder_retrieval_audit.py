import copy
import unittest

from experiments.audit_encoder_retrieval import evidence_metrics, validate_rankings


class RetrievalAuditTests(unittest.TestCase):
    def setUp(self):
        self.passages = {f'p{i}': f'text{i}' for i in range(15)}
        self.traces = [dict(case_id='q', method=method, k=k, task='task', group_id='group',
                           ids=list(self.passages)[:k], passages=list(self.passages.values())[:k])
                       for method in ('amor', 'without_recommendation', 'dense') for k in (5, 10, 15)]

    def check(self, traces):
        validate_rankings(traces, {'q'}, self.passages, 'task', 'group')

    def test_complete_grid(self):
        self.check(self.traces)

    def test_missing_depth(self):
        with self.assertRaisesRegex(ValueError, 'Missing'):
            self.check(self.traces[:-1])

    def test_repeated_condition(self):
        with self.assertRaisesRegex(ValueError, 'duplicated'):
            self.check(self.traces + [self.traces[0]])

    def test_source_identity_and_text_mismatch(self):
        traces = copy.deepcopy(self.traces)
        traces[0]['passages'][0] = 'wrong source'
        with self.assertRaisesRegex(ValueError, 'disagree'):
            self.check(traces)

    def test_duplicate_source_rejected(self):
        traces = copy.deepcopy(self.traces)
        traces[0]['ids'][1] = traces[0]['ids'][0]
        with self.assertRaisesRegex(ValueError, 'duplicated sources'):
            self.check(traces)

    def test_incorrect_cutoff_rejected(self):
        traces = copy.deepcopy(self.traces)
        traces[0]['ids'].pop()
        with self.assertRaisesRegex(ValueError, 'budget'):
            self.check(traces)

    def test_unresolved_gold_counts_as_miss(self):
        self.assertEqual(evidence_metrics(['p1', 'missing'], ['p1', 'p2']),
                         dict(precision=50., recall=50., complete=0.))

    def test_complete_gold(self):
        self.assertEqual(evidence_metrics(['p1', 'p1'], ['p1', 'p2']),
                         dict(precision=50., recall=100., complete=100.))

    def test_empty_gold_rejected(self):
        with self.assertRaises(ValueError):
            evidence_metrics([], ['p1'])


if __name__ == '__main__':
    unittest.main()

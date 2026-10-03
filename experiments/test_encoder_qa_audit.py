import copy
from types import SimpleNamespace
import unittest

from experiments.audit_encoder_qa import check_predictions


class PredictionAuditTests(unittest.TestCase):
    def setUp(self):
        self.rows = [SimpleNamespace(group_id='history', case=SimpleNamespace(case_id=f'q{i}'),
                     top_k=5, retrieved=[SimpleNamespace(text=f'passage{i}-{j}', metadata=dict(
                         source_passage=f'p{j}', context_representation='original_source_and_frozen_window'))
                         for j in range(5)]) for i in range(2)]
        self.predictions = [dict(group_id='history', case_id=f'q{i}', prediction='answer',
                                retrieved=[dict(text=f'passage{i}-{j}') for j in range(5)], metrics={'f1': .5}) for i in range(2)]
        self.generation = dict(max_tokens=50, temperature=.4, top_k=10, top_p=.9)
        self.usage = [dict(case_id=f'q{i}', seed=42, generation_settings=self.generation.copy(),
                           input_tokens=100, output_tokens=5) for i in range(2)]

    def run_check(self, rows=None, expected=None, predictions=None, usage=None):
        return check_predictions(self.rows if rows is None else rows,
            self.rows if expected is None else expected,
            self.predictions if predictions is None else predictions,
            self.usage if usage is None else usage, count=2, metric='f1',
            score_fn=lambda *args: {'f1': .5}, generation_fn=lambda case: self.generation)

    def test_complete_matched_records(self):
        values, inputs, outputs = self.run_check()
        self.assertEqual(values, {('history', 'q0'): .5, ('history', 'q1'): .5})
        self.assertEqual((inputs, outputs), (200, 10))

    def test_missing_prediction(self):
        with self.assertRaisesRegex(ValueError, 'coverage'):
            self.run_check(predictions=self.predictions[:1])

    def test_duplicate_identity(self):
        rows = [self.rows[0], self.rows[0]]
        with self.assertRaisesRegex(ValueError, 'identity'):
            self.run_check(rows=rows, expected=rows, predictions=[self.predictions[0]] * 2,
                           usage=[self.usage[0]] * 2)

    def test_changed_context(self):
        rows = copy.deepcopy(self.rows)
        rows[0].retrieved[0].text = 'different'
        with self.assertRaisesRegex(ValueError, 'context'):
            self.run_check(rows=rows)

    def test_changed_saved_score(self):
        predictions = copy.deepcopy(self.predictions)
        predictions[0]['metrics']['f1'] = 1.0
        with self.assertRaisesRegex(ValueError, 'native evaluator'):
            self.run_check(predictions=predictions)

    def test_changed_seed_or_generation(self):
        for field, replacement in [('seed', 43), ('generation_settings', {'max_tokens': 2048})]:
            usage = copy.deepcopy(self.usage)
            usage[0][field] = replacement
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.run_check(usage=usage)

    def test_context_overflow(self):
        usage = copy.deepcopy(self.usage)
        usage[0]['input_tokens'] = 32768
        with self.assertRaisesRegex(ValueError, 'context limit'):
            self.run_check(usage=usage)

    def test_historical_cutoff_counts_augmented_entries(self):
        expected = copy.deepcopy(self.rows)
        for row, prediction in zip(expected, self.predictions):
            row.retrieved.append(SimpleNamespace(text='appended fact',
                metadata={'context_representation': 'retrieved_triples'}))
            prediction['retrieved'].append({'text': 'appended fact'})
        historical = copy.deepcopy(expected)
        for row in historical:
            row.top_k = len(row.retrieved)
        self.run_check(rows=historical, expected=expected)

    def test_actual_passage_budget_change_fails(self):
        rows = copy.deepcopy(self.rows)
        rows[0].retrieved[-1].metadata['context_representation'] = 'retrieved_triples'
        with self.assertRaisesRegex(ValueError, 'passage budget'):
            self.run_check(rows=rows)

    def test_actual_source_identity_change_fails(self):
        rows = copy.deepcopy(self.rows)
        rows[0].retrieved[-1].metadata['source_passage'] = 'another-source'
        with self.assertRaisesRegex(ValueError, 'passage identities'):
            self.run_check(rows=rows)


if __name__ == '__main__':
    unittest.main()

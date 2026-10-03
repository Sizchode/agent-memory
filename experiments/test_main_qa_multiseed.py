import copy
import json
import unittest

from experiments.main_qa_multiseed import FIXTURE, check_scope, comparable_reader, verify_usage


class MainQASeedProtocolTests(unittest.TestCase):
    def setUp(self):
        self.table = json.loads(FIXTURE.read_text())
        self.generation = {"max_tokens": 50, "temperature": 0.4, "top_k": 10, "top_p": 0.9}
        self.cost = dict(case_id="question-1", seed=52, sampling_seed=52, raw_answer="London",
                         generation_settings=self.generation, input_tokens=300, output_tokens=2)

    def test_complete_user_table(self):
        check_scope(self.table)
        self.assertEqual(sum(len(row) for rows in self.table["scores_percent"].values() for row in rows), 240)
        self.assertEqual(self.table["scores_percent"]["Qwen/Qwen3.5-4B"][7], [93,59,75,14,49.63,59.14])

    def test_reject_incomplete_or_duplicate_table(self):
        incomplete = copy.deepcopy(self.table)
        incomplete["scores_percent"]["Qwen/Qwen3.5-4B"].pop()
        with self.assertRaises(ValueError):
            check_scope(incomplete)
        repeated = copy.deepcopy(self.table)
        repeated["tasks"][0] = repeated["tasks"][1]
        with self.assertRaises(ValueError):
            check_scope(repeated)

    def test_seed_52_must_reach_sampler(self):
        verify_usage(self.cost, "question-1", self.generation, 52, new_run=True)
        for updates in ({"seed":42}, {"sampling_seed":42}, {"sampling_seed":None}, {"raw_answer":""}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                verify_usage(dict(self.cost, **updates), "question-1", self.generation, 52, new_run=True)

    def test_native_generation_and_context_limit_preserved(self):
        changed = dict(self.cost, generation_settings=dict(self.generation, temperature=0.8))
        with self.assertRaises(ValueError):
            verify_usage(changed, "question-1", self.generation, 52, new_run=True)
        with self.assertRaises(ValueError):
            verify_usage(dict(self.cost, input_tokens=32768), "question-1", self.generation, 52, new_run=True)

    def test_historical_seed_metadata_is_not_invented(self):
        historical = {k:v for k,v in self.cost.items() if k not in {"sampling_seed", "raw_answer"}}
        historical["seed"] = 42
        verify_usage(historical, "question-1", self.generation, 42, new_run=False)
        with self.assertRaises(ValueError):
            verify_usage(historical, "question-1", self.generation, 42, new_run=True)

    def test_only_seed_and_allocation_are_excluded_from_reader_comparison(self):
        a = dict(model="model", revision="pinned", vllm="version", seed=42, gpu="gpu1")
        b = dict(a, seed=52, gpu="gpu2")
        self.assertEqual(comparable_reader(a), comparable_reader(b))
        self.assertNotEqual(comparable_reader(a), comparable_reader(dict(b, revision="changed")))


if __name__ == "__main__":
    unittest.main()

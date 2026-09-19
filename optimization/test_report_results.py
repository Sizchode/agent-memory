"""Check complete-matrix targets without substituting per-reader candidates."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from optimization import report_results


class ReportResultsTests(unittest.TestCase):
    def test_usage_missing_is_not_zero_and_partial_usage_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.assertIsNone(report_results.qa_usage(directory, 2))
            row = dict(input_tokens=10, output_tokens=2, seconds=0.5)
            path = directory / "qa_usage.jsonl"
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                report_results.qa_usage(directory, 2)
            self.assertEqual(report_results.qa_usage(directory, 1), dict(calls=1, **row))
            row["input_tokens"] = None
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                report_results.qa_usage(directory, 1)

    def test_full_target_requires_same_variant_six_wins_for_every_reader(self):
        tasks = {f"task{i}": (1, "score") for i in range(6)}
        models = [f"reader{i}" for i in range(3)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, baseline, output = root / "source", root / "baseline", root / "output"
            extra = root / "extra"

            def prediction(directory, score):
                directory.mkdir(parents=True)
                (directory / "summary.json").write_text(json.dumps({"score": score}))
                row = dict(group_id="group", case_id="case", metrics={"score": score})
                (directory / "predictions.jsonl").write_text(json.dumps(row) + "\n")

            for index, task in enumerate(tasks):
                reference = source / "hipporag2" / task
                reference.mkdir(parents=True)
                row = dict(group_id="group", case={"case_id": "case"})
                (reference / "retrieval.jsonl").write_text(json.dumps(row) + "\n")
                for model in models:
                    baseline_directory = baseline if model != models[-1] else extra / "control"
                    prediction(baseline_directory / task / "evaluations" / model, 0.5)
                    if model == models[-1] and index == 0:
                        prediction(baseline / task / "evaluations" / model, 0.9)
                    for variant in ("full", "five", "partial", "mixed"):
                        if variant == "partial" and index == 5:
                            continue
                        tie = variant == "five" and index == 5
                        tie |= variant == "mixed" and model == models[-1] and index == 5
                        prediction(output / variant / task / "evaluations" / model,
                                   0.5 if tie else 0.6)

            with patch.multiple(report_results, SOURCE=source, MODELS=models,
                                TASK_METRICS=tasks, BASELINES={"control": baseline}):
                with redirect_stdout(io.StringIO()):
                    report_results.report(output, ["full"])
                current = json.loads((output / "comparison.json").read_text())
                self.assertEqual(set(current["variants"]["full"]), set(models[:2]))
                with redirect_stdout(io.StringIO()):
                    report_results.report(output, ["full", "five", "partial", "mixed"], models=models,
                                          reference=output / "full", additional_baseline_roots=[extra])
                with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "baseline is missing"):
                    report_results.report(output, ["full"], models=models)
                with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "Incomplete main-method"):
                    report_results.report(output, ["five"], models=models, reference=output / "partial",
                                          additional_baseline_roots=[extra])
            result = json.loads((output / "comparison.json").read_text())
            self.assertEqual(result["target_wins"], 6)
            self.assertEqual(result["milestone_wins"], 5)
            self.assertTrue(result["variant_status"]["full"]["achieved"])
            self.assertTrue(result["variant_status"]["five"]["milestone_achieved"])
            for variant in ("five", "partial", "mixed"):
                self.assertFalse(result["variant_status"][variant]["achieved"])
            self.assertFalse(result["variant_status"]["partial"]["milestone_achieved"])
            self.assertEqual(result["variants"]["five"][models[0]]["wins"], 5)
            self.assertAlmostEqual(result["variants"]["five"][models[0]]["delta_from_reference"]["task5"], -0.1)
            self.assertIsNone(result["variants"]["partial"][models[0]]["delta_from_reference"]["task5"])
            self.assertEqual(result["variants"]["full"][models[0]]["delta_from_reference"]["task5"], 0.0)
            self.assertIn("-10.00", (output / "results.md").read_text())
            self.assertEqual(result["baseline_sources"][models[-1]]["control"], str(extra / "control"))
            self.assertEqual(result["baseline_sources"][models[0]]["control"], str(baseline))
            self.assertEqual(result["baselines"][models[-1]]["task0"]["best"], 0.5)


if __name__ == "__main__":
    unittest.main()

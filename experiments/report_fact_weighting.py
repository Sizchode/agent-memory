"""Verify native answer scores for matched graph-weighting comparisons."""

import argparse
import json
import math
from pathlib import Path

from experiments.mine_cases import read_lines, csv_rows, write_csv, write_json
from experiments.report_memory_findings import native_score
from experiments.report_paper_costs import TASKS, COUNTS, MODELS


def report(results, document):
    root = results / "analysis/fact_weighting"
    assert json.loads((root / "inputs/complete.json").read_text())["complete"]
    questions = {(q["task"], q["case_id"]): q for q in read_lines(results / "analysis/failure_questions.jsonl")}
    original = csv_rows(Path("/oscar/scratch/zliu328/agent-memory-outputs/optimization_paper_ablation_seed42_20260927/connection_controls/qa_results.csv"))
    current = csv_rows(results / "qa_results.csv")
    comparisons = [("unit", "amor"), ("unit", "unit_released"),
                   ("amor", "amor_released"), ("unit_released", "amor_released")]
    unchanged = {}
    for task in TASKS:
        old, = [r for r in original if r["task"] == task and r["variant"] == "without_projection"
                and r["model"].replace("/", "_") == MODELS[0]]
        full, = [r for r in current if r["task"] == task and r["variant"] == "full"
                 and r["setting"] == "one_shot" and r["model"].replace("/", "_") == MODELS[0]]
        paths = {"unit": Path(old["source"]) / "retrieval.jsonl",
                 "amor": Path(full["source"]) / "retrieval.jsonl"}
        paths.update({c: root / "inputs" / c / task / "retrieval.jsonl"
                      for c in ["unit_released", "amor_released"]})
        for before, after in comparisons:
            unchanged[task, before, after] = set()
        ids = set()
        # Identical cases/order also preserve LoCoMo's seeded option order.
        for group in zip(*(read_lines(p) for p in paths.values()), strict=True):
            rows = dict(zip(paths, group))
            base = rows["amor"]
            assert all(r["case"] == base["case"] for r in rows.values())
            cid = base["case"]["case_id"]
            assert cid not in ids
            ids.add(cid)
            texts = {c: [i["text"] for i in r["retrieved"]] for c, r in rows.items()}
            for before, after in comparisons:
                if texts[before] == texts[after]:
                    unchanged[task, before, after].add(cid)
        assert len(ids) == COUNTS[task]
    summaries, paired = [], []
    checks = 0
    for model in MODELS:
        assert json.loads((root / "qa" / model / "complete.json").read_text())["complete"]
        for task in TASKS:
            scores = {"amor": {cid: q["conditions"][model]["libra"]["score"]
                                for (t, cid), q in questions.items() if t == task}}
            old, = [r for r in original if r["task"] == task and r["variant"] == "without_projection"
                    and r["model"].replace("/", "_") == model]
            paths = {"unit": Path(old["source"]) / "evaluations" / model / "predictions.jsonl"}
            for condition in ["unit_released", "amor_released"]:
                directory = root / "qa" / model / condition / task
                marker = json.loads((directory / "qa_complete.json").read_text())
                assert marker["complete"] and marker["questions"] == COUNTS[task] and not marker["pilot"]
                paths[condition] = directory / "evaluations" / model / "predictions.jsonl"
            for condition, path in paths.items():
                values = {}
                for row in read_lines(path):
                    question = questions[task, row["case_id"]]
                    value = native_score(question, row["prediction"])
                    metric = "f1" if task == "LoCoMo" else "answer_f1" if task == "2WikiMultiHopQA" else "substring_exact_match"
                    assert math.isclose(value, row["metrics"][metric], abs_tol=1e-12)
                    assert row["case_id"] not in values
                    values[row["case_id"]] = value
                    checks += 1
                assert values.keys() == scores["amor"].keys() and len(values) == COUNTS[task]
                scores[condition] = values
            for condition, values in scores.items():
                summaries.append(dict(task=task, model=model, condition=condition,
                                      questions=len(values), score=100*sum(values.values())/len(values)))
            for before, after in comparisons:
                differences = [scores[after][cid]-scores[before][cid] for cid in scores[before]]
                repeated = [scores[after][cid]-scores[before][cid] for cid in unchanged[task, before, after]]
                paired.append(dict(task=task, model=model, before=before, after=after,
                    questions=len(differences), higher=sum(d>0 for d in differences),
                    lower=sum(d<0 for d in differences), equal=sum(d==0 for d in differences),
                    difference=100*sum(differences)/len(differences),
                    identical_prompt_questions=len(repeated),
                    identical_prompt_score_changes=sum(d!=0 for d in repeated),
                    identical_prompt_higher=sum(d>0 for d in repeated),
                    identical_prompt_lower=sum(d<0 for d in repeated)))
    data = document / "data"
    write_csv(data / "fact_weighting_answers.csv", summaries)
    write_csv(data / "fact_weighting_answer_pairs.csv", paired)
    labels = {"unit": "Unit weights", "amor": "AMOR",
              "unit_released": "Unit + key-fact boost", "amor_released": "AMOR + key-fact boost"}
    task_labels = ["SH-Doc", "MH-Doc", "FC-SH", "FC-MH", "LoCoMo", "2Wiki"]
    latex = [r"\begin{table*}[t]", r"\centering", r"\small", r"\begin{tabular}{llrrrr}",
             r"\toprule", r"Task & Weighting condition & Qwen4 & Qwen9 & Gemma & Llama \\", r"\midrule"]
    for task, label in zip(TASKS, task_labels):
        for i, condition in enumerate(["unit", "unit_released", "amor", "amor_released"]):
            values = [next(r["score"] for r in summaries
                          if r["task"] == task and r["model"] == model and r["condition"] == condition)
                      for model in MODELS]
            latex.append((label if i == 0 else "") + " & " + labels[condition] + " & "
                         + " & ".join(f"{v:.2f}" for v in values) + r" \\")
        latex.append(r"\addlinespace[3pt]")
    latex.extend([r"\bottomrule", r"\end{tabular}",
        r"\caption{Full configured data and native answer metrics for the matched weighting comparison. "
        r"Key-fact boost is the transferred released CatRAG operator, not the full system. "
        r"LoCoMo is the aggregate of its native category scores; 2Wiki is answer F1; "
        r"document and FC tasks use substring EM. Source connections, query initialization, "
        r"and context construction are unchanged. All values are percentages.}",
        r"\label{tab:weighting-answers}", r"\end{table*}"])
    (document / "weighting_answers.tex").write_text("\n".join(latex)+"\n")
    write_json(data / "fact_weighting_answer_verification.json", dict(complete=True,
        native_score_checks=checks, questions=sum(COUNTS.values()), readers=4,
        conditions=["unit", "amor", "unit_released", "amor_released"],
        source_contexts_verified=str(root / "inputs/complete.json")))
    print("Verified", checks, "native weighting answer scores", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--document", type=Path, required=True)
    args = parser.parse_args()
    report(args.results, args.document)

"""Verify native scores and assemble tables for the ACL memory analysis."""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from experiments.mine_cases import read_lines, csv_rows, write_csv, write_json
from experiments.report_paper_costs import TASKS, COUNTS, MODELS, METHODS
from utils.metrics import substring_exact_match
from utils.locomo_metrics import locomo_qa_f1
from utils.hipporag_metrics import hipporag_answer_f1


def native_score(question, prediction):
    task = question["task"]
    if task == "LoCoMo":
        return locomo_qa_f1(prediction, question["answers"][0], question["category"])
    if task == "2WikiMultiHopQA":
        return hipporag_answer_f1(prediction, question["answers"])
    return float(substring_exact_match(prediction, question["answers"]))


def report(results, output):
    output.mkdir(parents=True, exist_ok=True)
    analysis = results / "analysis"
    questions = {}
    totals, pairs = defaultdict(list), defaultdict(list)
    count = 0
    methods = METHODS + ["without_propagation", "without_context_augmentation"]
    for row in read_lines(analysis / "failure_questions.jsonl"):
        key = row["task"], row["case_id"]
        assert key not in questions
        questions[key] = row
        assert set(row["conditions"]) == set(MODELS)
        for model, conditions in row["conditions"].items():
            assert set(conditions) == set(methods)
            for method, value in conditions.items():
                actual = native_score(row, value["prediction"])
                assert math.isclose(actual, value["score"], abs_tol=1e-12)
                totals[row["task"], model, method].append(actual)
                pairs[row["task"], model, method].append(conditions["libra"]["score"] - actual)
                count += 1
    assert Counter(t for t, _ in questions) == COUNTS
    summary, paired = [], []
    for task in TASKS:
        for model in MODELS:
            for method in methods:
                values = totals[task, model, method]
                assert len(values) == COUNTS[task]
                metric = ("LoCoMo native mixed score" if task == "LoCoMo" else
                          "Answer F1" if task == "2WikiMultiHopQA" else "Substring EM")
                summary.append(dict(task=task, model=model, method=method, questions=len(values),
                                    metric=metric, score=100*sum(values)/len(values)))
                differences = pairs[task, model, method]
                paired.append(dict(task=task, model=model, comparator=method, questions=len(values),
                                   amor_higher=sum(x>0 for x in differences), equal=sum(x==0 for x in differences),
                                   amor_lower=sum(x<0 for x in differences),
                                   score_difference=100*sum(differences)/len(differences)))
    write_csv(output / "native_scores.csv", summary)
    write_csv(output / "paired_outcomes.csv", paired)

    context_values = defaultdict(dict)
    for model in MODELS:
        for key, question in questions.items():
            if key[0] != "LoCoMo":
                continue
            for name, method in [("full", "libra"), ("centers_only", "without_context_augmentation")]:
                context_values[model, name][key[1]] = question["conditions"][model][method]["score"]
        path = analysis / "context_components" / model / "predictions.jsonl"
        for row in read_lines(path):
            question = questions["LoCoMo", row["case_id"]]
            actual = native_score(question, row["prediction"])
            assert math.isclose(actual, row["metrics"]["f1"], abs_tol=1e-12)
            key = model, row["condition"]
            assert row["case_id"] not in context_values[key]
            context_values[key][row["case_id"]] = actual
            count += 1
    context_summary, context_pairs = [], []
    for model in MODELS:
        for category in ["ALL", 1, 2, 3, 4, 5]:
            ids = [cid for (task, cid), q in questions.items() if task == "LoCoMo"
                   and (category == "ALL" or q["category"] == category)]
            for condition in ["centers_only", "facts_only", "neighbors_only", "full"]:
                assert len(context_values[model, condition]) == COUNTS["LoCoMo"]
                values = context_values[model, condition]
                context_summary.append(dict(model=model, condition=condition, category=category,
                    questions=len(ids), score=100*sum(values[cid] for cid in ids)/len(ids)))
            for before, after in [("centers_only", "neighbors_only"), ("facts_only", "full"),
                                  ("centers_only", "facts_only"), ("neighbors_only", "full")]:
                differences = [context_values[model, after][cid] - context_values[model, before][cid] for cid in ids]
                context_pairs.append(dict(model=model, category=category, before=before, after=after,
                    questions=len(ids), higher=sum(d>0 for d in differences), equal=sum(d==0 for d in differences),
                    lower=sum(d<0 for d in differences), difference=100*sum(differences)/len(ids)))
    for row in csv_rows(analysis / "context_components.csv"):
        match, = [r for r in context_summary if r["model"] == row["model"].replace("/", "_")
                  and r["condition"] == row["condition"] and str(r["category"]) == row["category"]]
        assert math.isclose(match["score"], float(row["score"]), abs_tol=1e-9)
    write_csv(output / "context_operations.csv", context_summary)
    write_csv(output / "context_operation_pairs.csv", context_pairs)

    interventions = []
    for model in MODELS:
        records = list(read_lines(analysis / "memory_interventions" / model / "predictions.jsonl"))
        assert len(records) == 207
        for row in records:
            question = questions[row["task"], row["case_id"]]
            actual = native_score(question, row["prediction"])
            assert actual == row["metrics"]["substring_exact_match"]
            interventions.append(dict(model=model, task=row["task"], condition=row["condition"],
                scope=row["scope"], case_id=row["case_id"], prediction=row["prediction"], score=actual))
            count += 1
    write_csv(output / "fact_interventions.csv", interventions)
    population = []
    for task, condition in [("FactConsolidation-MH", "all_fact_candidates"),
                            ("FactConsolidation-SH", "all_context_facts")]:
        for model in MODELS:
            control = [r for r in interventions if r["task"] == task and r["model"] == model and r["condition"] == condition]
            assert len(control) == 100
            current = totals[task, model, "libra"]
            population.append(dict(task=task, model=model, condition=condition, questions=100,
                current=sum(current), control=sum(r["score"] for r in control)))
    write_csv(output / "fact_population.csv", population)
    write_json(output / "verification.json", dict(complete=True, unique_questions=len(questions),
        tasks=COUNTS, score_checks=count, native_scores=True, no_new_metric=True,
        sources=[str(analysis / "failure_questions.jsonl"), str(analysis / "context_components"),
                 str(analysis / "memory_interventions")],
        scope="All six existing settings; three benchmark suites and four underlying data origins."
        " The 1000-query 2Wiki release and seed-42 development evaluation are unchanged."))
    print("Verified", count, "native scores across", len(questions), "questions", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report(args.results, args.output)

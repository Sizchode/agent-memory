"""Audit complete native QA predictions and report fixed graph comparisons."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import zipfile

from experiments.runner import _read_retrieval_records, _score
from optimization.ircot import BASE, MODELS, write_json
from optimization.report_results import BASELINES, TASK_METRICS, audited_score
from optimization.run_graph import SOURCE, TASKS


QA_ROOT = BASE / "optimization_fact_graph_main_qa_seed42_20260919"
IRCOT_ROOT = BASE / "optimization_ircot_fact_graph_seed42_20260919"
CAPS = (1, 3, 5)
VARIANTS = ("bm25", "fact_graph", "fact_graph_without_synonyms")


def audit_cell(target, model, task, expected):
    count, metric = TASK_METRICS[task]
    marker = json.loads((target / "qa_complete.json").read_text())
    assert marker["complete"] and not marker["pilot"] and marker["questions"] == count
    assert marker["native_evaluator"] and marker["scores_recomputed"]
    rows = list(_read_retrieval_records(target / "retrieval.jsonl"))
    assert [(row.group_id, row.case) for row in rows] == [(row.group_id, row.case) for row in expected]
    output = target / "evaluations" / model.replace("/", "_")
    predictions = [json.loads(line) for line in (output / "predictions.jsonl").open()]
    for row, prediction in zip(rows, predictions, strict=True):
        assert (row.group_id, row.case.case_id) == (prediction["group_id"], prediction["case_id"])
        assert prediction["metrics"] == _score(row.case, prediction["prediction"], row.retrieved, row.top_k)
        assert [item.text for item in row.retrieved] == [item["text"] for item in prediction["retrieved"]]
    score = audited_score(output, task, {(row.group_id, row.case.case_id) for row in expected})
    assert score is not None and score == marker["score"]
    usage_rows = [json.loads(line) for line in (output / "qa_usage.jsonl").open()]
    assert len(usage_rows) == count
    for row, usage_row in zip(rows, usage_rows, strict=True):
        assert row.case.case_id == usage_row["case_id"]
        assert all(isinstance(usage_row[key], int) and usage_row[key] >= 0
                   for key in ("input_tokens", "output_tokens", "cached_input_tokens"))
    usage = {key: sum(row[key] for row in usage_rows) for key in ("input_tokens", "output_tokens", "cached_input_tokens")}
    assert all(usage[key] == marker[key] for key in ("input_tokens", "output_tokens"))
    return dict(score=score, metric=metric, questions=count, usage=usage)


def comparison(left, right):
    differences = [a-b for a, b in zip(left, right, strict=True)]
    return dict(wins=sum(value > 0 for value in differences), ties=sum(value == 0 for value in differences),
                losses=sum(value < 0 for value in differences))


def report_components(root, models, controls):
    """Reuse native auditing for complete reader/control/task experiment folders."""
    if (not models or len(set(models)) != len(models) or not set(models).issubset(MODELS)
            or not controls or len(set(controls)) != len(controls)):
        raise ValueError("Select unique supported readers and unique controls")
    if any(Path(control).name != control or control in (".", "..") for control in controls):
        raise ValueError("Controls must be directory names")
    root = Path(root)
    expected = {task: list(_read_retrieval_records(SOURCE / "hipporag2" / task / "retrieval.jsonl"))
                for task in TASK_METRICS}
    for task, rows in expected.items():
        assert len(rows) == len({(r.group_id, r.case.case_id) for r in rows}) == TASK_METRICS[task][0]
    result = dict(complete=False, seed=42, test_as_dev=True, scores_recomputed=True,
                  controls=controls, scores={}, comparisons={}, reference_baselines={})
    lines = ["# 组件对照结果", "", "六任务全量；seed 42；test-as-dev；按未舍入原生分数比较，不跨任务求平均。",
             "同批组件对照与先前完整九项 baseline 分开报告；不同批次的小幅差值可能包含生成变化。", ""]
    with zipfile.ZipFile(root / f"report_code_{os.environ['SLURM_JOB_ID']}.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in (Path(__file__), Path("experiments/runner.py"), Path("optimization/report_results.py")):
            archive.write(path, path.name if path.is_absolute() else str(path))
        assert archive.testzip() is None
    for model in models:
        slug = model.replace("/", "_")
        directory = root / slug
        complete = json.loads((directory / "complete.json").read_text())
        metadata = json.loads((directory / "reader.json").read_text())
        assert complete["complete"] and not complete["pilot"] and complete["model"] == model
        assert set(complete["tasks"]) == set(TASK_METRICS)
        assert all(set(values) == set(controls) for values in complete["tasks"].values())
        reference = QA_ROOT / slug / "main"
        assert metadata == json.loads((reference / "reader.json").read_text())
        assert metadata["revision"] == MODELS[model]
        result["scores"][model], result["reference_baselines"][model] = {}, {}
        lines += ["## " + model, "", "### 本次完整组件对照", "",
                  "| 任务 | " + " | ".join(controls) + " |", "|---|" + "---:|" * len(controls)]
        for task in TASK_METRICS:
            values = {}
            for control in controls:
                target = directory / control / task
                values[control] = audit_cell(target, model, task, expected[task])
                values[control]["native_summary"] = json.loads((target / "qa_complete.json").read_text())["summary"]
            result["scores"][model][task] = values
            reference_values = {name: audit_cell(reference / name / task, model, task, expected[task])
                                for name in BASELINES}
            result["reference_baselines"][model][task] = reference_values
            lines.append("| " + task + " | " + " | ".join(f"{100 * values[c]['score']:.2f}" for c in controls) + " |")
        values = result["scores"][model]
        result["comparisons"][model] = {
            left: {right: comparison([values[t][left]["score"] for t in TASK_METRICS],
                                    [values[t][right]["score"] for t in TASK_METRICS])
                   for right in controls if right != left} for left in controls}
        lines += ["", "### 先前九项 Baseline 参考", "",
                  "来源：`" + str(reference) + "`。这批 baseline 未在本次重新生成答案；不称为同批比较。", "",
                  "| 任务 | " + " | ".join(BASELINES) + " |", "|---|" + "---:|" * len(BASELINES)]
        for task in TASK_METRICS:
            baselines = result["reference_baselines"][model][task]
            lines.append("| " + task + " | " + " | ".join(f"{100 * baselines[b]['score']:.2f}" for b in BASELINES) + " |")
        best = [max(v["score"] for v in result["reference_baselines"][model][t].values()) for t in TASK_METRICS]
        result.setdefault("against_prior_best_of_nine", {})[model] = {
            c: comparison([values[t][c]["score"] for t in TASK_METRICS], best) for c in controls}
        lines += ["", "对先前逐任务最佳的胜/平/负（跨批次参考）：`" +
                  json.dumps(result["against_prior_best_of_nine"][model], sort_keys=True) + "`", "",
                  "### 2Wiki 原生段落 Recall@5", "", "| 条件 | Recall@5 (%) |", "|---|---:|"]
        for control in controls:
            score = values["2WikiMultiHopQA"][control]["native_summary"]["passage_recall_at_5"]
            lines.append(f"| {control} | {100 * score:.2f} |")
        lines += ["", "逐题原生评分与上下文身份已核对；QA token 用量及完整原生汇总见 `comparison.json`。", ""]
    result["complete"] = True
    write_json(root / "comparison.json", result)
    (root / "results.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result["comparisons"], indent=2), flush=True)


def report(kind):
    root = QA_ROOT if kind == "one-shot" else IRCOT_ROOT
    with zipfile.ZipFile(root / f"report_code_{os.environ['SLURM_JOB_ID']}.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in (Path(__file__), Path("experiments/runner.py"),
                     Path("optimization/report_results.py"), Path("optimization/ircot.py")):
            archive.write(path, path.name if path.is_absolute() else str(path))
        assert archive.testzip() is None
    expected = {task: list(_read_retrieval_records(SOURCE / "hipporag2" / task / "retrieval.jsonl")) for task in TASK_METRICS}
    for task, rows in expected.items():
        assert len(rows) == len({(row.group_id, row.case.case_id) for row in rows}) == TASK_METRICS[task][0]
    report = dict(complete=False, kind=kind, seed=42, test_as_dev=True, scores_recomputed=True,
                  native_metrics=True, scores={}, comparisons={})
    lines = ["# " + kind + " results", "", "Full six-task data; seed 42; test-as-dev.",
             "Native scores are percentages; no cross-metric average. All comparisons use unrounded scores.", ""]
    for model, revision in MODELS.items():
        slug = model.replace("/", "_")
        directory = root / slug / "main" if kind == "one-shot" else root / "main" / slug
        complete = json.loads((directory / "complete.json").read_text())
        metadata = json.loads((directory / "reader.json").read_text())
        assert complete["complete"] and not complete["pilot"] and complete["model"] == model
        assert metadata["revision"] == revision
        report["scores"][model] = {}
        lines += ["## " + model, ""]
        if kind == "one-shot":
            controls = tuple(BASELINES) + ("fact_graph",)
            assert set(complete["sources"]) == set(controls) and complete["questions"] == 3386
            lines += ["| Task | " + " | ".join(controls) + " |", "|---|" + "---:|" * len(controls)]
            candidate, best, by_baseline = [], [], {name: [] for name in BASELINES}
            for task in TASK_METRICS:
                values = {control: audit_cell(directory / control / task, model, task, expected[task]) for control in controls}
                report["scores"][model][task] = values
                candidate.append(values["fact_graph"]["score"])
                best.append(max(values[name]["score"] for name in BASELINES))
                for name in BASELINES:
                    by_baseline[name].append(values[name]["score"])
                lines.append("| " + task + " | " + " | ".join(f"{100*values[name]['score']:.2f}" for name in controls) + " |")
            report["comparisons"][model] = dict(best_of_nine=comparison(candidate, best),
                individual={name: comparison(candidate, values) for name, values in by_baseline.items()})
        else:
            assert complete["caps"] == list(CAPS) and complete["variants"] == list(VARIANTS)
            assert complete["seed"] == 42 and complete["tasks"] == list(TASKS)
            pilot = json.loads((root / "pilot" / slug / "complete.json").read_text())
            assert pilot["complete"] and pilot["native_controller_equivalence_checked"]
            lines += ["| Task | Backend | Cap 1 | Cap 3 | Cap 5 |", "|---|---|---:|---:|---:|"]
            for task in TASK_METRICS:
                values = {variant: {str(cap): audit_cell(directory / f"cap_{cap}" / variant / task, model, task, expected[task])
                                    for cap in CAPS} for variant in VARIANTS}
                report["scores"][model][task] = values
                for variant in VARIANTS:
                    lines.append("| " + task + " | " + variant + " | " +
                                 " | ".join(f"{100*values[variant][str(cap)]['score']:.2f}" for cap in CAPS) + " |")
            report["comparisons"][model] = {}
            for cap in CAPS:
                values = report["scores"][model]
                candidate = [values[task]["fact_graph_without_synonyms"][str(cap)]["score"] for task in TASK_METRICS]
                report["comparisons"][model][str(cap)] = {baseline: comparison(candidate,
                    [values[task][baseline][str(cap)]["score"] for task in TASK_METRICS]) for baseline in VARIANTS[:2]}
            report.setdefault("trace_usage", {})[model] = trace_usage(directory)
            plot(model, report["scores"][model], root)
        lines += ["", "Win/tie/loss: `" + json.dumps(report["comparisons"][model], sort_keys=True) + "`", ""]
    report["complete"] = True
    write_json(root / "comparison.json", report)
    (root / "results.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report["comparisons"], indent=2), flush=True)


def trace_usage(directory):
    totals = {}
    for variant in VARIANTS:
        totals[variant] = {}
        for cap in CAPS:
            aggregate = Counter()
            for task in TASK_METRICS:
                question_count = 0
                for path in (directory / "traces" / variant / task).glob("*.json"):
                    saved = json.loads(path.read_text())
                    assert saved["complete"] and not saved["pilot"]
                    for trace in saved["traces"]:
                        question_count += 1
                        steps = trace["rounds"][:cap]
                        aggregate["reasoning_requests"] += len(steps)
                        for step in steps:
                            for key in ("input_tokens", "output_tokens", "cached_input_tokens"):
                                aggregate["reasoning_" + key] += step["reasoning"]["usage"][key]
                            for call in step["recognition"]:
                                assert call["status"] == "completed" and call["usage_available"]
                                aggregate["new_recognition_calls"] += 1
                                aggregate["recognition_input_tokens"] += call["prompt_tokens"]
                                aggregate["recognition_output_tokens"] += call["completion_tokens"]
                assert question_count == TASK_METRICS[task][0]
            totals[variant][str(cap)] = dict(aggregate)
    return dict(totals=totals, caveat="Recognition counts are incremental provider calls with reused and shared caches, not independent cold runs")


def plot(model, scores, root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = root / "figures"
    directory.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    styles = (("bm25", "IRCoT + BM25", "#3268a8"), ("fact_graph", "Fact graph, all edges", "#777777"),
              ("fact_graph_without_synonyms", "Fact graph, no synonym edges", "#c15242"))
    for ax, task in zip(axes.flat, TASK_METRICS, strict=True):
        for variant, label, color in styles:
            ax.plot(CAPS, [100*scores[task][variant][str(cap)]["score"] for cap in CAPS], marker="o", label=label, color=color)
        ax.set_title(task)
        ax.set_xlabel("Maximum retrieval rounds")
        ax.set_ylabel(TASK_METRICS[task][1] + " (%)")
        ax.set_xticks(CAPS)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(model.split("/")[-1])
    for extension in ("pdf", "png"):
        fig.savefig(directory / (model.replace("/", "_") + "." + extension), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("one-shot", "ircot", "components"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--models", nargs="+", choices=tuple(MODELS))
    parser.add_argument("--controls", nargs="+")
    args = parser.parse_args()
    if args.kind == "components":
        if args.root is None or not args.models or not args.controls:
            parser.error("components requires --root, --models and --controls")
        report_components(args.root, args.models, args.controls)
    else:
        if args.root is not None or args.models or args.controls:
            parser.error("--root, --models and --controls apply only to components")
        report(args.kind)

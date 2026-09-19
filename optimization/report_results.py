"""Audit complete prediction files and report each reader's strict task wins."""

import argparse
import json
import math
from pathlib import Path

from optimization.run_graph import BASE, MODELS, SOURCE, write_json


TASK_METRICS = {
    "SH-Doc_QA": (100, "substring_exact_match"),
    "MH-Doc_QA": (100, "substring_exact_match"),
    "FactConsolidation-SH": (100, "substring_exact_match"),
    "FactConsolidation-MH": (100, "substring_exact_match"),
    "LoCoMo": (1986, "f1"),
    "2WikiMultiHopQA": (1000, "answer_f1"),
}
BASELINES = {name: SOURCE / name for name in ("bm25", "dense", "hipporag2", "mem0", "lightmem")}
BASELINES.update({
    "lightmem_offline": BASE / "lightmem_offline_20260911T184409Z/lightmem",
    "anchormem_dense": BASE / "anchormem_qa_controls_seed42_20260911/anchormem_dense_top5",
    "anchormem_official": BASE / "anchormem_qa_controls_seed42_20260911/anchormem_official",
    "catrag": BASE / "catrag_hypermem_top5_seed42_20260912_h100_batch8/catrag_top5",
})


def audited_score(directory, task, expected_keys):
    count, metric = TASK_METRICS[task]
    summary_path = directory / "summary.json"
    predictions_path = directory / "predictions.jsonl"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    with predictions_path.open() as stream:
        predictions = [json.loads(line) for line in stream]
    keys = [(r["group_id"], r["case_id"]) for r in predictions]
    if len(keys) != count or len(set(keys)) != count or set(keys) != expected_keys:
        raise ValueError(f"Coverage mismatch: {directory}")
    score = sum(float(r["metrics"][metric]) for r in predictions) / count
    if abs(score - summary[metric]) > 1e-12:
        raise ValueError(f"Summary disagrees with predictions: {directory}")
    return score


def qa_usage(directory, count):
    path = directory / "qa_usage.jsonl"
    if not path.exists():
        return None
    with path.open() as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != count:
        raise ValueError(f"QA usage does not cover the completed predictions: {directory}")
    totals = {}
    for field in ("input_tokens", "output_tokens", "seconds"):
        values = [row.get(field) for row in rows]
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
               for value in values):
            raise ValueError(f"Invalid or missing QA usage {field}: {directory}")
        totals[field] = sum(values)
    return dict(calls=count, **totals)


def report(root, variants, models=None, reference=None, additional_baseline_roots=()):
    models = list(MODELS[:2] if models is None else models)
    if not models or len(set(models)) != len(models) or not set(models).issubset(MODELS):
        raise ValueError("Select unique supported readers")
    if (root / "INVALID.json").exists():
        raise ValueError(f"Experiment is explicitly invalid: {root}")
    expected = {}
    for task, (count, _) in TASK_METRICS.items():
        with (SOURCE / "hipporag2" / task / "retrieval.jsonl").open() as stream:
            rows = [json.loads(line) for line in stream]
        expected[task] = {(r["group_id"], r["case"]["case_id"]) for r in rows}
        if len(rows) != count or len(expected[task]) != count:
            raise ValueError(f"Reference is incomplete: {task}")
    baselines, baseline_sources = {}, {}
    for model in models:
        sources = {}
        for name, primary in BASELINES.items():
            candidates = [primary] + [path / name for path in additional_baseline_roots]
            complete = [path for path in candidates if all(
                (path / task / "evaluations" / model.replace("/", "_") / "summary.json").is_file()
                for task in TASK_METRICS)]
            if not complete:
                raise ValueError(f"A declared complete baseline is missing: {model}, {name}")
            sources[name] = complete[0]
        baseline_sources[model] = {name: str(path) for name, path in sources.items()}
        baselines[model] = {}
        for task in TASK_METRICS:
            scores = {name: audited_score(path / task / "evaluations" / model.replace("/", "_"),
                                          task, expected[task]) for name, path in sources.items()}
            if any(score is None for score in scores.values()):
                raise ValueError(f"A declared complete baseline is missing: {model}, {task}")
            baselines[model][task] = dict(scores=scores, best=max(scores.values()))
    result = dict(development_set_results=True, target_wins=6, milestone_wins=5,
                  baselines=baselines, baseline_sources=baseline_sources, variants={}, variant_status={})
    lines = ["# 构图优化结果", "", "既有 test set 用作开发集；缺失成绩不补零。", "",
             "目标包括较大上下文预算的 AnchorMem 官方设置，不能统称等上下文比较。", "",
             "报告 reader：" + "、".join(models) + "。同一候选全部 6/6 为完整胜出，5/6 为阶段里程碑。", ""]
    if reference is not None:
        result["reference"] = dict(directory=str(reference), scores={})
        lines += ["## 完整方法参考", "", "| 模型 | " + " | ".join(TASK_METRICS) + " |",
                  "|---|" + "---:|" * len(TASK_METRICS)]
        for model in models:
            scores = {task: audited_score(reference / task / "evaluations" / model.replace("/", "_"),
                                         task, expected[task]) for task in TASK_METRICS}
            if any(score is None for score in scores.values()):
                raise ValueError(f"Incomplete main-method reference: {model}, {reference}")
            result["reference"]["scores"][model] = scores
            lines.append("| " + model.split("/")[-1] + " | " +
                         " | ".join(f"{100 * score:.2f}" for score in scores.values()) + " |")
        lines += ["", "下表差值为当前配置减完整方法，单位为百分点；不同任务指标不求平均。", ""]
    for variant in variants:
        result["variants"][variant] = {}
        lines += [f"## {variant}", "", "| 模型 | " + " | ".join(TASK_METRICS) + " | 严格胜出 |",
                  "|---|" + "---:|" * (len(TASK_METRICS) + 1)]
        for model in models:
            scores, costs, wins, complete = {}, {}, 0, 0
            for task in TASK_METRICS:
                directory = root / variant / task / "evaluations" / model.replace("/", "_")
                score = audited_score(directory, task, expected[task])
                scores[task] = score
                costs[task] = qa_usage(directory, TASK_METRICS[task][0]) if score is not None else None
                if score is not None:
                    complete += 1
                    wins += score > baselines[model][task]["best"]
            result["variants"][variant][model] = dict(scores=scores, qa_usage=costs, wins=wins, complete_tasks=complete,
                                                      milestone_achieved=complete == 6 and wins >= 5,
                                                      achieved=complete == 6 and wins == 6)
            values = ["未完成" if score is None else f"{100 * score:.2f}" for score in scores.values()]
            lines.append("| " + model.split("/")[-1] + " | " + " | ".join(values) + f" | {wins}/6 ({complete}/6 完成) |")
            if reference is not None:
                deltas = {task: None if score is None else score - result["reference"]["scores"][model][task]
                          for task, score in scores.items()}
                result["variants"][variant][model]["delta_from_reference"] = deltas
                values = ["未完成" if delta is None else f"{100 * delta:+.2f}" for delta in deltas.values()]
                lines.append("| 相对完整方法 | " + " | ".join(values) + " | |")
        reader_results = result["variants"][variant].values()
        result["variant_status"][variant] = dict(
            milestone_achieved=all(row["milestone_achieved"] for row in reader_results),
            achieved=all(row["achieved"] for row in reader_results))
        lines.append("")
        lines += ["QA 成本仅汇总已完成且有 usage 的任务；不是建图/检索或总 wall time，跨 GPU 耗时不作等硬件比较。", "",
                  "| 模型 | 有 usage 的完整任务 | 调用数 | 输入 token | 输出 token | QA 调用秒数 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for model in models:
            costs = [value for value in result["variants"][variant][model]["qa_usage"].values() if value is not None]
            if costs:
                totals = {field: sum(value[field] for value in costs)
                          for field in ("calls", "input_tokens", "output_tokens", "seconds")}
                lines.append(f"| {model.split('/')[-1]} | {len(costs)}/6 | {totals['calls']} | "
                             f"{totals['input_tokens']} | {totals['output_tokens']} | {totals['seconds']:.1f} |")
            else:
                lines.append(f"| {model.split('/')[-1]} | 0/6 | 未完成或缺失 | 未完成或缺失 | 未完成或缺失 | 未完成或缺失 |")
        lines.append("")
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "comparison.json", result)
    (root / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS)
    parser.add_argument("--reference-dir", type=Path,
                        help="Frozen main-method variant directory with complete evaluations")
    parser.add_argument("--additional-baseline-root", type=Path, action="append", default=[],
                        help="Additional baseline experiment roots; use the first complete six-task directory, never select by score")
    args = parser.parse_args()
    report(args.output_root, args.variants, args.models, args.reference_dir, args.additional_baseline_root)


if __name__ == "__main__":
    main()

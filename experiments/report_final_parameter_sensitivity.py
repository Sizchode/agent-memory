"""Verify final-graph QA and produce one eight-panel parameter bar plot."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from experiments.encoder_robustness import digest, json_save
from experiments.final_parameter_sensitivity import READERS, SETTINGS, TASKS


def verify_condition(root, model, setting, task):
    from experiments.runner import _read_retrieval_records, _score
    from optimization.report_results import TASK_METRICS

    directory = root / "qa" / model.replace("/", "_") / setting / task
    marker = json.loads((directory / "verified.json").read_text())
    evaluation = directory / "evaluations" / model.replace("/", "_")
    if not marker["complete"] or marker["pilot"]:
        raise ValueError("Only complete full evaluations can be reported")
    if digest(directory / "retrieval.jsonl") != marker["retrieval_sha256"]:
        raise ValueError("Retrieval changed after evaluation")
    if digest(evaluation / "predictions.jsonl") != marker["predictions_sha256"]:
        raise ValueError("Predictions changed after evaluation")
    rows = list(_read_retrieval_records(directory / "retrieval.jsonl"))
    predictions = [json.loads(line) for line in (evaluation / "predictions.jsonl").open()]
    count, metric = TASK_METRICS[task]
    if not len(rows) == len(predictions) == marker["questions"] == count:
        raise ValueError("Question coverage differs")
    values = {}
    for row, prediction in zip(rows, predictions, strict=True):
        key = (row.group_id, row.case.case_id)
        if key in values or key != (prediction["group_id"], prediction["case_id"]):
            raise ValueError("Question identities differ")
        if [r.text for r in row.retrieved] != [r["text"] for r in prediction["retrieved"]]:
            raise ValueError("Prediction context differs")
        actual = _score(row.case, prediction["prediction"], row.retrieved, row.top_k)
        if actual != prediction["metrics"]:
            raise ValueError("Saved metrics disagree with native scorer")
        value = actual[metric]
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid score")
        values[key] = value
    mean = 100 * float(np.mean(list(values.values())))
    if abs(mean - 100 * marker["score"]) > 1e-9:
        raise ValueError("Aggregate differs from native evaluation")
    return mean, values, dict(model=model, setting=setting, task=task, questions=count,
        predictions_sha256=marker["predictions_sha256"], native_scores_recomputed=True)


def paired_interval(baseline, treatment, *, cluster, repeats=5000):
    if baseline.keys() != treatment.keys():
        raise ValueError("Unpaired question identities")
    keys = sorted(baseline)
    units = sorted({key[0] for key in keys}) if cluster else keys
    deltas = np.asarray([treatment[key] - baseline[key] for key in keys])
    if cluster:
        totals = np.asarray([sum(deltas[i] for i, key in enumerate(keys) if key[0] == unit) for unit in units])
        counts = np.asarray([sum(key[0] == unit for key in keys) for unit in units])
    else:
        totals, counts = deltas, np.ones(len(keys))
    rng = np.random.default_rng(42)
    draws = []
    for _ in range(repeats):
        sample = rng.integers(0, len(units), len(units))
        draws.append(100 * totals[sample].sum() / counts[sample].sum())
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def plot(rows, target):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    matplotlib.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8,
        "axes.titlesize": 9, "axes.labelsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    labels = ["SH-Doc", "MH-Doc", "FC-SH", "FC-MH", "LoCoMo", "2Wiki"]
    reader_labels = ["Qwen3.5-4B", "Qwen3.5-9B", "Gemma-3-4B-it", "Llama-3.1-8B"]
    groups = [("alpha_025", "alpha_075"), ("lambda_001", "lambda_010")]
    colors = ("#2467A8", "#D9822B")
    lookup = {(r["model"], r["setting"], r["task"]): r for r in rows}
    maximum = max(abs(r["difference_pp"]) for r in rows)
    limit = max(1.0, math.ceil(maximum * 1.15))
    fig, axes = plt.subplots(2, 4, figsize=(8.0, 4.1), sharey=True)
    for row_index, settings in enumerate(groups):
        for column, model in enumerate(READERS):
            ax = axes[row_index, column]
            x = np.arange(len(TASKS))
            for arm, setting in enumerate(settings):
                values = [lookup[model, setting, task]["difference_pp"] for task in TASKS]
                ax.bar(x + (arm - .5) * .36, values, width=.34, color=colors[arm],
                    edgecolor="white", linewidth=.3, zorder=3)
            ax.axhline(0, color="#333333", linewidth=.7)
            ax.set_ylim(-limit, limit)
            ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=7)
            ax.grid(axis="y", alpha=.16, linewidth=.6, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
            if row_index == 0:
                ax.set_title(reader_labels[column], pad=6)
            if column == 0:
                ax.set_ylabel((r"$\alpha$" if row_index == 0 else r"$\lambda$") + " sensitivity\nScore change (pp)")
    handles = [Patch(facecolor=c) for c in colors]
    fig.legend(handles, [r"$\alpha=0.25$", r"$\alpha=0.75$"], loc="upper center",
        bbox_to_anchor=(.55, 1.01), ncol=2, frameon=False)
    fig.legend(handles, [r"$\lambda=0.01$", r"$\lambda=0.10$"], loc="upper center",
        bbox_to_anchor=(.55, .49), ncol=2, frameon=False)
    fig.subplots_adjust(left=.075, right=.995, bottom=.12, top=.86, wspace=.16, hspace=.95)
    fig.savefig(target / "parameter_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(target / "parameter_sensitivity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def report(root):
    protocol = json.loads((root / "protocol.json").read_text())
    if protocol["graph_variant"] != "without_projected_connections" or not protocol["fresh_qa_reference"]:
        raise ValueError("Not the declared final-graph experiment")
    rows, audits = [], []
    for model in READERS:
        for task in TASKS:
            reference, base, audit = verify_condition(root, model, "reference", task)
            audits.append(audit)
            for setting, (alpha, weight) in SETTINGS.items():
                mean, values, audit = verify_condition(root, model, setting, task)
                if setting != "reference":
                    audits.append(audit)
                low, high = paired_interval(base, values, cluster=task == "LoCoMo")
                rows.append(dict(model=model, task=task, setting=setting, alpha=alpha, lambda_value=weight,
                    questions=len(values), score=mean, reference_score=reference, difference_pp=mean-reference,
                    paired_low=low, paired_high=high))
    target = root / "report"
    target.mkdir(parents=True, exist_ok=True)
    with (target / "scores.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_save(target / "audit.json", dict(conditions=audits, questions=sum(a["questions"] for a in audits),
        all_120_conditions=True, bootstrap_repeats=5000,
        bootstrap_units="questions except LoCoMo conversations; seed42",
        figure="point estimate differences in native EM/F1 percentage points, no cross-task averaging"))
    plot(rows, target)
    json_save(target / "complete.json", dict(complete=True, conditions=len(rows),
        scores_sha256=digest(target / "scores.csv"), pdf_sha256=digest(target / "parameter_sensitivity.pdf"),
        visual_review_pending=True))
    print("Verified full final-graph sensitivity; plot awaits visual review", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    report(parser.parse_args().root)

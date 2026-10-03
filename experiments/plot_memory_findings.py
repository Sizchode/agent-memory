"""Render native-metric comparisons and export their exact plotted values."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.mine_cases import csv_rows, read_lines, write_csv
from experiments.report_paper_costs import MODELS


def plot(results, document):
    data = document / "data"
    assert json.loads((data / "verification.json").read_text())["complete"]
    weight_dir = results / "analysis/fact_weighting"
    assert json.loads((weight_dir / "complete.json").read_text())["complete"]
    official = {r["_id"]: r for r in json.loads(Path(
        "baseline_algorithms/HippoRAG/reproduce/dataset/2wikimultihopqa.json").read_text())}
    rows = list(read_lines(weight_dir / "2WikiMultiHopQA.jsonl"))
    assert len(rows) == len(official) == 1000
    assert {r["case_id"] for r in rows} == set(official)
    conditions = ["unit", "unit_released", "unit_paper", "amor", "amor_released", "amor_paper"]
    coverage = []
    for category in ["ALL", "comparison", "compositional", "bridge_comparison", "inference"]:
        subset = [r for r in rows if category == "ALL" or official[r["case_id"]]["type"] == category]
        for condition in conditions:
            complete = [set(r["gold"]).issubset(r["selected"][condition]) for r in subset]
            reference = [set(r["gold"]).issubset(r["selected"]["amor"]) for r in subset]
            coverage.append(dict(category=category, condition=condition, questions=len(subset),
                complete_at_5=100*sum(complete)/len(subset),
                recall_at_5=100*sum(len(set(r["gold"]) & set(r["selected"][condition]))/len(set(r["gold"])) for r in subset)/len(subset),
                amor_gains=sum(a and not c for a,c in zip(reference,complete)),
                amor_losses=sum(c and not a for a,c in zip(reference,complete))))
    write_csv(data / "fact_weighting_by_type.csv", coverage)
    target = document / "figures"
    target.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65), gridspec_kw={"width_ratios": [1.05, 1]})
    labels = ["Unit", "Unit +\nkey-fact", "AMOR", "AMOR +\nkey-fact"]
    chosen = ["unit", "unit_released", "amor", "amor_released"]
    values = [next(r["complete_at_5"] for r in coverage if r["category"] == "ALL" and r["condition"] == c) for c in chosen]
    ax = axes[0]
    bars = ax.bar(range(4), values, color=["#8f9baa", "#d99b69", "#477ba7", "#69a2ba"], width=.66)
    ax.bar_label(bars, labels=[f"{v:.1f}" for v in values], padding=3, fontsize=8)
    ax.set_ylim(0, 82)
    ax.set_xticks(range(4), labels)
    ax.set_ylabel("Complete support@5 (%)")
    ax.set_title("(a) Same graph, same recognized facts\n2Wiki: all 1,000 questions", fontsize=9)
    ax.axhline(66.7, color="#62686f", linestyle=":", linewidth=1)
    ax.text(.02,.95,"Original graph minus synonyms: 66.7", transform=ax.transAxes, va="top", fontsize=7)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color="#e8ebed", linewidth=.5)
    context = csv_rows(data / "context_operations.csv")
    ax = axes[1]
    positions = np.arange(4)
    plotted = []
    for category, label, color, shift in [(4,"Single-hop F1 (841)","#477ba7",-.15),
                                         (5,"Refusal accuracy (446)","#ba655d",.15)]:
        values = []
        for model in MODELS:
            lookup = {r["condition"]: float(r["score"]) for r in context
                      if r["model"] == model and r["category"] == str(category)}
            difference = lookup["neighbors_only"] - lookup["centers_only"]
            values.append(difference)
            plotted.append(dict(model=model, category=category, before=lookup["centers_only"],
                                after=lookup["neighbors_only"], difference=difference))
        bars = ax.bar(positions+shift, values, width=.27, color=color, label=label)
        ax.bar_label(bars, labels=[f"{v:+.1f}" for v in values], padding=2, fontsize=7)
    ax.axhline(0, color="#62686f", linewidth=.7)
    ax.set_ylim(-28,23)
    ax.set_xticks(positions,["Qwen4", "Qwen9", "Gemma", "Llama"])
    ax.set_ylabel("Change in native score (points)")
    ax.set_title("(b) Same five selected conversation turns\nAdd neighboring turns, no appended facts",fontsize=9)
    ax.legend(loc="upper center", bbox_to_anchor=(.5,-.15), frameon=False, fontsize=7, ncol=1)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color="#e8ebed", linewidth=.5)
    fig.tight_layout(w_pad=2.0)
    for suffix in ["pdf", "png"]:
        fig.savefig(target / ("memory_operations."+suffix), dpi=230, bbox_inches="tight")
    plt.close(fig)
    write_csv(data / "plotted_neighbor_effects.csv", plotted)
    print("Rendered", target / "memory_operations.pdf", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--document", type=Path, required=True)
    args = parser.parse_args()
    plot(args.results, args.document)

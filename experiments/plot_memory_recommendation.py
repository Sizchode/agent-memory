"""Plot official question types and a verified recommendation intervention."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

from experiments.mine_cases import csv_rows, write_csv


def plot(results, document):
    assert json.loads((document / "data/verification.json").read_text())["complete"]
    rows = csv_rows(results / "analysis/wiki_question_types.csv")
    cases = csv_rows(document / "data/fact_interventions.csv")
    outcomes = {}
    for condition in ["current", "old_authorship", "restored"]:
        selected = [r for r in cases if r["task"] == "FactConsolidation-MH"
                    and r["case_id"] == "factconsolidation_mh_262k_no45" and r["condition"] == condition]
        assert len(selected) == 4
        outcomes[condition] = sum(float(r["score"]) for r in selected)
    assert outcomes == {"current": 4, "old_authorship": 0, "restored": 4}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.85), gridspec_kw={"width_ratios": [1.18, 1]})
    ax = axes[0]
    types = ["comparison", "compositional", "bridge_comparison", "inference"]
    labels = ["Comparison\n244; 2 supports", "Compositional\n413; 2 supports",
              "Bridge comp.\n235; 4 supports", "Inference\n108; 2 supports"]
    methods = [("without_propagation", "w/o recommendation", "#a9b1ba"),
               ("hipporag2", "HippoRAG 2", "#d59b73"), ("libra", "AMOR", "#477ba7")]
    exported = []
    for i, (method, label, color) in enumerate(methods):
        chosen = [next(r for r in rows if r["question_type"] == t and r["method"] == method) for t in types]
        values = [float(r["all_support_at_5"]) for r in chosen]
        bars = ax.bar(np.arange(4)+(i-1)*.24, values, width=.22, color=color, label=label)
        if method == "libra":
            ax.bar_label(bars, labels=[f"{v:.1f}" for v in values], padding=2, fontsize=7)
        exported.extend(chosen)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks(np.arange(4), labels, fontsize=6.8)
    ax.set_ylabel("Complete annotated support@5 (%)")
    ax.set_title("(a) Evidence dependencies matter\nAll 1,000 2Wiki questions", fontsize=9)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color="#e8ebed", linewidth=.6)
    ax.legend(loc="upper center", bbox_to_anchor=(.5,-.19), frameon=False, fontsize=7, ncol=2)

    ax = axes[1]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("(b) Same question; change one accepted fact\nWhere did Dubliners' author die?", fontsize=9)
    def box(x, y, text, color):
        patch = FancyBboxPatch((x, y), .42, .16, boxstyle="round,pad=0.01,rounding_size=0.015",
                              facecolor=color, edgecolor="#6d7c89", linewidth=.6)
        ax.add_patch(patch)
        ax.text(x+.21, y+.08, text, ha="center", va="center", fontsize=8)
    for x, author, source, answer, color in [
        (.03, "Author: George Eliot", "Eliot death source\nLondon supplied", "4/4 readers correct", "#e5eff7"),
        (.55, "Author: James Joyce", "Eliot death source\nLondon absent", "0/4 readers correct", "#f6e9e2")]:
        box(x,.72,author,color)
        box(x,.43,source,color)
        box(x,.14,answer,color)
        for bottom, top in [(.59,.72),(.30,.43)]:
            ax.annotate("",xy=(x+.21,bottom),xytext=(x+.21,top),
                        arrowprops=dict(arrowstyle="->",color="#6d7c89",lw=.8))
    ax.text(.5,.655,"Recommendation",ha="center",va="center",fontsize=7,
            bbox=dict(facecolor="white",edgecolor="none",pad=1))
    ax.text(.5,.36,"Native QA",ha="center",va="center",fontsize=7,
            bbox=dict(facecolor="white",edgecolor="none",pad=1))
    ax.text(.5,.025,"Corrected Eliot authorship stays in BOTH inputs.\nRestore accepted fact: 4/4 correct again.",
            ha="center",va="center",fontsize=7)
    fig.tight_layout(w_pad=1.6)
    destination = document / "figures"
    destination.mkdir(exist_ok=True)
    for suffix in ["pdf", "png"]:
        fig.savefig(destination / ("memory_recommendation."+suffix), dpi=240, bbox_inches="tight")
    plt.close(fig)
    write_csv(document / "data/plotted_recommendation_types.csv", exported)
    print("Rendered", destination / "memory_recommendation.pdf", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--document", type=Path, required=True)
    args = parser.parse_args()
    plot(args.results, args.document)

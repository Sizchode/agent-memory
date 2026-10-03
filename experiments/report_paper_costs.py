"""Report measured construction, QA, IRCoT costs and passage recall from saved runs."""

import argparse
import csv
import json
import math
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


TASKS = ["SH-Doc_QA", "MH-Doc_QA", "FactConsolidation-SH",
         "FactConsolidation-MH", "LoCoMo", "2WikiMultiHopQA"]
SHORT = dict(zip(TASKS, ["SH-Doc", "MH-Doc", "FC-SH", "FC-MH", "LoCoMo", "2Wiki"]))
COUNTS = dict(zip(TASKS, [100, 100, 100, 100, 1986, 1000]))
METRICS = dict(zip(TASKS, ["substring_exact_match"] * 4 + ["f1", "answer_f1"]))
MODELS = ["Qwen_Qwen3.5-4B", "Qwen_Qwen3.5-9B", "google_gemma-3-4b-it",
          "meta-llama_Llama-3.1-8B-Instruct"]
MODEL_LABELS = ["Qwen3.5-4B", "Qwen3.5-9B", "Gemma-3-4B", "Llama-3.1-8B"]
METHODS = ["bm25", "dense", "hipporag2", "catrag", "mem0", "lightmem_offline",
           "anchormem_official", "libra"]
LABELS = dict(zip(METHODS, ["BM25", "Dense", "HippoRAG 2", "CatRAG", "Mem0",
                          "LightMem", "AnchorMem", "AMOR"]))
LABELS.update(bm25_native="BM25 w/o context augmentation",
              bm25_same_context="BM25 w/ context augmentation",
              sources="AMOR w/o context augmentation",
              full="AMOR w/ context augmentation",
              without_propagation="AMOR w/o recommendation")
STYLES = dict(zip(METHODS, [("#F2A36B", "o"), ("#F0CC57", "D"), ("#99A2B2", "^"),
                          ("#A5C896", "s"), ("#BB91B2", "v"), ("#67ACAB", "P"),
                          ("#C48B76", "p"), ("#6488B3", "h")]))
STYLES.update(bm25_native=STYLES["bm25"], bm25_same_context=("#B6AC80", "X"),
              sources=("#7EADB9", "*"), full=STYLES["libra"],
              without_propagation=("#BB91B2", "^"))
VARIANT = "statement_projection_loop_free_retained_index_rrf_window"


def read(path):
    return json.loads(path.read_text())


def lines(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    text_fields = {"model", "task", "method", "group", "stage", "source", "coverage",
                   "directory", "trace_directory", "setting", "cutoff"}
    with path.open() as stream:
        return [{k: (v if k in text_fields else None if v == "" else float(v))
                 for k, v in row.items()} for row in csv.DictReader(stream)]


def qa_record(directory, task, expected_score):
    evaluations = list(directory.glob("evaluations/*/qa_usage.jsonl"))
    assert len(evaluations) == 1, (directory, evaluations)
    usage = list(lines(evaluations[0]))
    n = COUNTS[task]
    assert len(usage) == n and len({x["case_id"] for x in usage}) == n
    total = {k: sum(x[k] for x in usage) for k in ("input_tokens", "output_tokens")}
    assert all(isinstance(x[k], (int, float)) and math.isfinite(x[k]) and x[k] >= 0
               for x in usage for k in total)
    prediction_path = evaluations[0].with_name("predictions.jsonl")
    ids, score = [], 0.0
    for prediction in lines(prediction_path):
        ids.append(prediction["case_id"])
        score += prediction["metrics"][METRICS[task]]
    assert len(ids) == n and len(set(ids)) == n
    assert set(ids) == {x["case_id"] for x in usage}
    assert math.isclose(score / n, expected_score, abs_tol=1e-10), directory
    return dict(questions=n, score=100 * score / n, **total,
                qa_input_per_question=total["input_tokens"] / n,
                qa_output_per_question=total["output_tokens"] / n,
                qa_total_per_question=sum(total.values()) / n, directory=str(directory))


def one_shot(root):
    rows = []
    for model in MODELS:
        llama = model == MODELS[-1]
        ours = root / ("optimization_fact_context_seed42_20260920/source_and_fact_context/main" if llama
                       else "optimization_source_fact_transfer_seed42_20260920/main") / model
        native = root / ("optimization_context_reranking_seed42_20260920/one_shot/main" if llama
                         else "optimization_native_readers_seed42_20260920/main") / model
        comparison = read(ours / "comparison.json")["main_comparison"]
        for task in TASKS:
            for method in METHODS:
                score = comparison[task]["ours"] if method == "libra" else comparison[task]["native_baselines"][method]
                directory = (ours / "optimized_graph" / task if llama else ours / task) if method == "libra" else native / method
                if method != "libra":
                    directory = directory / "original" / task if llama else directory / task
                rows.append(dict(model=model, task=task, method=method,
                                 **qa_record(directory, task, score)))
    return rows


def trace_cost(root, model, task, variant, cap):
    directory = root / "optimization_ircot_hybrid_seed42_20260918/main" / model / "traces" / variant / task
    result = dict(reasoning_input_tokens=0, reasoning_output_tokens=0, reasoning_calls=0,
                  recognition_input_observed=0, recognition_output_observed=0,
                  recognition_calls_observed=0, recognition_missing_usage=0,
                  reasoning_batch_seconds=0.0, retrieval_seconds_observed=0.0)
    ids = set()
    for path in sorted(directory.glob("*.json")):
        data = read(path)
        assert data["complete"] and not data["pilot"]
        for trace in data["traces"]:
            assert trace["case_id"] not in ids
            ids.add(trace["case_id"])
            assert trace["rounds"]
            for step in trace["rounds"][:cap]:
                usage = step["reasoning"]["usage"]
                result["reasoning_input_tokens"] += usage["input_tokens"]
                result["reasoning_output_tokens"] += usage["output_tokens"]
                result["reasoning_calls"] += 1
                result["retrieval_seconds_observed"] += step["retrieval_seconds"]
                for call in step["recognition"]:
                    result["recognition_calls_observed"] += 1
                    if call.get("prompt_tokens") is None or call.get("completion_tokens") is None:
                        result["recognition_missing_usage"] += 1
                    else:
                        result["recognition_input_observed"] += call["prompt_tokens"]
                        result["recognition_output_observed"] += call["completion_tokens"]
        result["reasoning_batch_seconds"] += sum(x["seconds"] for x in data["batch_timings"] if x["round"] <= cap)
    assert len(ids) == COUNTS[task], (directory, len(ids))
    result["trace_directory"] = str(directory)
    return result


def iterative(root):
    rows, cache = [], {}
    summary = read(root / "optimization_method_analysis_seed42_20260921/ablation_summary.json")
    assert summary["complete"] and not summary["missing"]
    for unit in summary["results"]:
        if unit["setting"] == "one_shot":
            continue
        model, task, cap = unit["model"].replace("/", "_"), unit["task"], int(unit["setting"].split("_")[1])
        for method in ("bm25_native", "bm25_same_context", "sources", "full"):
            condition = unit["conditions"][method]
            qa = qa_record(Path(condition["directory"]), task, condition["score"])
            variant = "bm25" if method.startswith("bm25") else "optimized_graph"
            key = (model, task, variant, cap)
            if key not in cache:
                cache[key] = trace_cost(root, *key)
            costs = cache[key]
            reader_tokens = qa["input_tokens"] + qa["output_tokens"] + costs["reasoning_input_tokens"] + costs["reasoning_output_tokens"]
            rows.append(dict(model=model, task=task, cap=cap, method=method, **qa, **costs,
                             reader_total_per_question=reader_tokens / COUNTS[task]))
    assert len(rows) == 4 * 6 * 3 * 4
    return rows


def one_shot_context_ablation(root, one, data_dir, table_dir):
    summary = read(root / "optimization_method_analysis_seed42_20260921/ablation_summary.json")
    conditions = ["sources", "sources_facts", "sources_neighbors", "full"]
    rows = []
    for unit in summary["results"]:
        if unit["setting"] != "one_shot":
            continue
        model, task = unit["model"].replace("/", "_"), unit["task"]
        for method in conditions:
            condition = unit["conditions"][method]
            qa = qa_record(Path(condition["directory"]), task, condition["score"])
            rows.append(dict(model=model, task=task, method=method, **qa))
            if method == "full":
                reference, = [r for r in one if r["model"] == model and r["task"] == task
                              and r["method"] == "libra"]
                for key in ("score", "questions", "input_tokens", "output_tokens"):
                    assert math.isclose(qa[key], reference[key], abs_tol=1e-10)
    lookup = {(r["model"], r["task"], r["method"]): r for r in rows}
    assert len(rows) == len(lookup) == len(MODELS) * len(TASKS) * len(conditions)
    write_csv(data_dir / "one_shot_context_ablation.csv", rows)
    labels = ["Sources only", "Sources + facts", "Sources + neighbors", "AMOR (both)"]
    tex = [r"\begin{table*}[t]", r"\centering\small",
           r"\caption{One-shot context ablations (\%). We hold selected source identities and order, prompts, and generation settings fixed and vary whether selected facts and neighboring text are displayed. Source metadata is preserved. Facts are selected once using the full source and neighbor pool and are not reselected when neighbors are hidden. Input length is not held fixed. Bold indicates the best condition for each model and task, including ties.}",
           r"\label{tab:one-shot-context-ablation}",
           r"\begin{tabular}{lrrrrrr}", r"\toprule",
           "Context & " + " & ".join(SHORT[t] for t in TASKS) + r" \\"]
    comparisons = {method: [0, 0, 0] for method in conditions[:-1]}
    for model, model_label in zip(MODELS, MODEL_LABELS):
        tex.extend([r"\midrule", r"\multicolumn{7}{c}{\textbf{" + model_label + r"}} \\"])
        for method, label in zip(conditions, labels):
            cells = []
            for task in TASKS:
                score = lookup[model, task, method]["score"]
                best = max(lookup[model, task, m]["score"] for m in conditions)
                cell = f"{score:.2f}"
                cells.append(r"\textbf{" + cell + "}" if abs(score - best) < 1e-10 else cell)
                if method != "full":
                    delta = lookup[model, task, "full"]["score"] - score
                    comparisons[method][0 if delta > 1e-10 else 2 if delta < -1e-10 else 1] += 1
            tex.append(label + " & " + " & ".join(cells) + r" \\")
    tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (table_dir / "one_shot_context_ablation.tex").write_text("\n".join(tex) + "\n")
    win, tie, loss = comparisons["sources"]
    text = rf"""\subsection{{Context Ablations for One-Shot QA}}
\label{{app:one-shot-context}}
Table~\ref{{tab:one-shot-context-ablation}} separates the effects of presenting selected facts and neighboring text. All four conditions reuse AMOR's selected sources, preserving their order and metadata. We independently remove facts, neighboring text, or both from the final answer context, without changing the question, prompt template, model, or scoring procedure. Facts and their order are fixed before this intervention; the facts-only condition can therefore contain assertions extracted from neighbors whose original text is not displayed.

Compared with sources alone, full AMOR improves {win} settings, ties in {tie}, and declines in {loss} across four LLMs and six tasks. These comparisons measure the effects of context presentation, including changes in input length; they do not isolate graph construction or compare equal token budgets. They also differ from the BM25 versus AMOR controls in the iterative experiments.

\input{{tables/one_shot_context_ablation}}
"""
    (table_dir / "one_shot_context_analysis.tex").write_text(text)
    print("Validated one-shot context conditions:", len(rows), comparisons, flush=True)


def usage_totals(rows, nested=False):
    calls = list(rows)
    tokens = [r.get("usage") if nested else r for r in calls]
    missing = sum(not u or u.get("prompt_tokens") is None or u.get("completion_tokens") is None for u in tokens)
    return dict(calls=len(calls), input_tokens=None if missing else sum(u["prompt_tokens"] for u in tokens),
                output_tokens=None if missing else sum(u["completion_tokens"] for u in tokens), missing_usage=missing)


def generation_delta(before, after):
    a, b = before["generation_client"], after["generation_client"]
    missing = b["responses_without_usage"] - a["responses_without_usage"]
    return dict(calls=b["client_calls_started"] - a["client_calls_started"],
                input_tokens=None if missing else b["prompt_tokens"] - a["prompt_tokens"],
                output_tokens=None if missing else b["completion_tokens"] - a["completion_tokens"],
                missing_usage=missing)


def construction(root):
    rows, retrieval = [], []
    original = root / "final_qwen3_30b_seed42_clean_20260910"
    zero = {"generation_client": dict(client_calls_started=0, responses_without_usage=0,
                                     prompt_tokens=0, completion_tokens=0)}
    for task in TASKS:
        for method in ("bm25", "dense", "hipporag2", "mem0", "lightmem"):
            path = original / method / task / "efficiency.jsonl"
            records = list(lines(path))
            assert len(records) == (10 if task == "LoCoMo" else 1)
            for record in records:
                assert record["status"] == "completed" and record["build_memory"]
                before, after = record["generator_statistics_before_build"], record["generator_statistics_after_build"]
                costs = generation_delta(before, after) if before else dict(calls=0, input_tokens=0, output_tokens=0, missing_usage=0)
                common = dict(task=task, method=method, group=record["group_id"], source=str(path))
                rows.append(dict(**common, stage="base_index", **costs, seconds=record["build_seconds"],
                                 records=record["input_memory_items"], coverage="recorded build"))
                if after:
                    retrieval.append(dict(**common, **generation_delta(after, record["generator_statistics_after_retrieval"]),
                                          coverage="observed provider calls; caches may bypass recorder"))
        path = root / "lightmem_offline_20260911T184409Z/lightmem" / task / "consolidation.jsonl"
        for record in lines(path):
            if record["status"] != "completed":
                continue
            costs = generation_delta(zero, record["generator_statistics"])
            rows.append(dict(task=task, method="lightmem_offline", stage="additional_consolidation",
                             group=record["group_id"], **costs,
                             seconds=record["queue_seconds"] + record["offline_update_seconds"], source=str(path),
                             coverage="add to LightMem base_index; queue is similarity preparation, not Slurm wait"))
        for path in sorted((root / "anchormem_full_seed42_20260911/anchormem" / task).glob("*/construction.jsonl")):
            lineage, directory = [], path.parent
            while directory is not None:
                assert directory not in lineage, "Cyclic AnchorMem recovery lineage"
                lineage.append(directory)
                recovered = read(directory / "settings.json").get("recover_from")
                directory = Path(recovered) if recovered else None
            for index, directory in enumerate(lineage):
                for record in lines(directory / "construction.jsonl"):
                    if record["status"] not in ("completed", "failed"):
                        continue
                    costs = generation_delta(zero, record["generator_statistics"])
                    stage = "inherited_index" if index else "base_index"
                    rows.append(dict(task=task, method="anchormem_official",
                                     stage=stage if record["status"] == "completed" else "failed_attempt",
                                     group=record["group_id"], **costs,
                                     seconds=record["elapsed_seconds_including_initialization"],
                                     source=str(directory / "construction.jsonl"),
                                     coverage="provider usage along explicit recovery lineage, including initialization"))
        path = root / "catrag_hypermem_top5_seed42_20260912_h100_batch8/catrag_top5" / task / "usage.jsonl"
        records = list(lines(path))
        for stage in ("build", "access"):
            selected = [r for r in records if r.get("stage") == stage and r["event"] == "provider_call"]
            row = dict(task=task, method="catrag", group="all", stage=stage, **usage_totals(selected), source=str(path),
                       seconds=sum(r["elapsed_seconds"] for r in records if r.get("stage") == stage and r["event"] == "stage"),
                       coverage="observed calls; resumed build may exclude inherited cache")
            (rows if stage == "build" else retrieval).append(row)
        compiled = root / "optimization_retained_fact_index_seed42_20260914/compiled_sources" / task
        for path in sorted(compiled.glob("*/construction.json")):
            record = read(path)
            schema = Path(record["source_schema"])
            phase2 = read(schema / "complete.json")
            phase1 = Path(phase2["source_schema"]).parent
            for stage, directory in (("relation_names", phase1), ("relation_equivalence", schema)):
                complete = read(directory / "complete.json")
                seconds = complete.get("seconds") if stage == "relation_names" else sum(complete[k] for k in ("embedding_seconds", "clustering_seconds", "normalization_seconds"))
                rows.append(dict(task=task, method="libra", stage=stage, group=path.parent.name,
                                 **usage_totals(lines(directory / "batches.jsonl"), nested=True), seconds=seconds,
                                 source=str(directory), coverage="additional to shared extraction/index"))
            rows.append(dict(task=task, method="libra", stage="graph_projection", group=path.parent.name,
                             calls=0, input_tokens=0, output_tokens=0, missing_usage=0,
                             seconds=record["seconds"], records=record["original_passages_preserved"],
                             source=str(path), coverage="cached extraction and normalization; not cold build",
                             source_statements=record["source_statements"], retained_support=record["retained_statement_support"]))
        directory = root / "optimization_retained_fact_index_seed42_20260914" / VARIANT / task
        paths = list(directory.glob("runtime/*/recognition_usage.jsonl"))
        calls = [r for path in paths for r in lines(path)]
        retrieval.append(dict(task=task, method="libra", group="all", **usage_totals(calls), source=str(directory / "runtime"),
                              coverage="cache misses only; NOT complete cold-query recognition cost"))
    return rows, retrieval


def retrieval_metrics(root):
    original = root / "final_qwen3_30b_seed42_clean_20260910"
    paths = {m: original / m / "2WikiMultiHopQA/retrieval.jsonl" for m in ("bm25", "dense", "hipporag2")}
    paths["catrag"] = root / "catrag_hypermem_top5_seed42_20260912_h100_batch8/catrag_top5/2WikiMultiHopQA/retrieval.jsonl"
    paths["libra"] = root / "optimization_retained_fact_index_seed42_20260914" / VARIANT / "2WikiMultiHopQA/retrieval.jsonl"
    rows, gold_reference = [], None
    for method, path in paths.items():
        data = list(lines(path))
        gold = {r["case"]["case_id"]: set(r["case"]["gold_passages"]) for r in data}
        assert len(data) == len(gold) == 1000 and all(gold.values())
        if gold_reference is None:
            gold_reference = gold
        assert gold == gold_reference
        recalls, precisions, complete = [], [], 0
        for record in data:
            texts = [x.get("metadata", {}).get("original_source_text", x["text"]) for x in record["retrieved"][:5]]
            relevant = gold[record["case"]["case_id"]]
            hits = len(set(texts) & relevant)
            recalls.append(hits / len(relevant))
            precisions.append(hits / len(texts) if texts else 0)
            complete += relevant.issubset(set(texts))
        rows.append(dict(setting="one_shot", model="shared", method=method, cap=None, questions=1000,
                         recall=100 * sum(recalls) / 1000, precision=100 * sum(precisions) / 1000,
                         all_support_percent=complete / 10, all_support_count=complete, source=str(path),
                         cutoff="top 5 original passages"))
    content_path = root / "optimization_retained_fact_index_seed42_20260914/compiled_sources/2WikiMultiHopQA/hipporag-2wikimultihopqa/contents.json"
    contents = read(content_path)
    sources = {v["source_position"]: v["original_source_text"] for v in contents.values()}
    assert len(sources) == 6119 and set(sources) == set(range(6119))
    with (root / "optimization_amem_transfer_seed42_20260918/inputs/2WikiMultiHopQA.pkl").open("rb") as stream:
        groups = pickle.load(stream)
    assert len(groups) == 1
    assert list(groups[0].memory_items) == [sources[i] for i in range(6119)]
    for model in MODELS:
        for method in ("bm25", "optimized_graph"):
            path = root / "optimization_ircot_hybrid_seed42_20260918/main" / model / "traces" / method / "2WikiMultiHopQA/hipporag-2wikimultihopqa.json"
            traces = read(path)["traces"]
            assert len(traces) == 1000 and {r["case_id"] for r in traces} == set(gold_reference)
            for cap in (1, 3, 5):
                recall, complete, count = 0.0, 0, 0
                for trace in traces:
                    step = trace["rounds"][min(cap, len(trace["rounds"])) - 1]
                    selected = {sources[i] for i in step["selected_sources"]}
                    assert len(selected) <= 15
                    gold = gold_reference[trace["case_id"]]
                    recall += len(selected & gold) / len(gold)
                    complete += gold.issubset(selected)
                    count += len(selected)
                rows.append(dict(setting="ircot", model=model, method=method, cap=cap, questions=1000,
                                 recall=recall / 10, all_support_percent=complete / 10, all_support_count=complete,
                                 mean_passages=count / 1000, source=str(path), cutoff="accumulated set; at most 15 passages"))
    return rows


def save(fig, directory, name):
    fig.savefig(directory / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(directory / (name + ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def legend(directory, name, methods, ncol, labels=None, fontsize=9, **layout):
    labels = LABELS if labels is None else labels
    handles = [Line2D([], [], color=STYLES[m][0], marker=STYLES[m][1], markersize=6,
                      markeredgecolor="#444444", markeredgewidth=0.45, linestyle="None", label=labels[m]) for m in methods]
    save_legend(directory, name, handles, ncol, fontsize=fontsize, **layout)


def save_legend(directory, name, handles, ncol, fontsize=9, **layout):
    fig = plt.figure(figsize=(7.2, 0.24 * math.ceil(len(handles) / ncol) + 0.1))
    fig.legend(handles=handles, loc="center", ncol=ncol, frameon=True,
               fancybox=False, edgecolor="#9AA2AF",
               prop={"weight": "bold", "size": fontsize}, **layout)
    save(fig, directory, name)


def grid_plot(rows, directory, name, methods, x, xlabel, curves, tasks=TASKS):
    # Each task keeps its own metric; no averaging across columns or readers.
    compact = len(tasks) == 3
    fig, axes = plt.subplots(4, len(tasks), figsize=(7.2, 6.5) if compact else (15.2, 8.2))
    fig.subplots_adjust(left=0.11 if compact else 0.065, right=0.995, top=0.96,
                        bottom=0.095, wspace=0.33, hspace=0.48)
    for i, model in enumerate(MODELS):
        for j, task in enumerate(tasks):
            ax = axes[i, j]
            for method in methods:
                selected = sorted([r for r in rows if r["model"] == model and r["task"] == task and r["method"] == method], key=lambda r: r.get("cap", 0))
                assert len(selected) == (3 if curves else 1)
                color, marker = STYLES[method]
                ax.plot([r[x] / 1000 if x.endswith("per_question") else r[x] for r in selected],
                        [r["score"] for r in selected], color=color, marker=marker,
                        markeredgecolor="#444444", markeredgewidth=0.45, markersize=5.0,
                        linewidth=1.1, linestyle="-" if curves else "None", zorder=4 if method in ("libra", "full") else 3)
            if x == "cap":
                ax.set_xticks([1, 3, 5])
            if i == 0:
                ax.set_title(SHORT[task], fontweight="bold")
            if j == 0:
                ax.set_ylabel(MODEL_LABELS[i] + "\nQA score (%)", fontweight="bold")
            ax.grid(color="#D7DDE5", linewidth=0.6)
            ax.set_axisbelow(True)
            ax.margins(x=0.13, y=0.18)
            low, high = ax.get_ylim()
            ax.set_ylim(max(-0.5, low), min(101, high))
            if i == 3:
                ax.set_xlabel(xlabel)
    save(fig, directory, name)


def iteration_cost_plots(rows, directory, methods, suffix=""):
    # Keep panel scales fixed across iteration caps and main/ablation figures.
    metric_labels = ["Substring EM (%)"] * 4 + ["F1 / accuracy (%)", "Answer F1 (%)"]
    for cap in (1, 3, 5):
        fig, axes = plt.subplots(4, 6, figsize=(7.2, 4.4))
        fig.subplots_adjust(left=0.085, right=0.99, top=0.89, bottom=0.14,
                            wspace=0.62, hspace=0.28)
        for i, model in enumerate(MODELS):
            for j, task in enumerate(TASKS):
                ax = axes[i, j]
                reference = [r for r in rows if r["model"] == model and r["task"] == task
                             and r["method"] in methods]
                assert len(reference) == 3 * len(methods)
                for method in methods:
                    selected = [r for r in reference if r["cap"] == cap and r["method"] == method]
                    assert len(selected) == 1
                    row = selected[0]
                    color, marker = STYLES[method]
                    ax.plot(row["reader_total_per_question"] / 1000, row["score"],
                            color=color, marker=marker, linestyle="None", markersize=4.5,
                            markeredgecolor="#444444", markeredgewidth=0.4,
                            zorder=4 if method == "full" else 3)
                xs = [r["reader_total_per_question"] / 1000 for r in reference]
                ys = [r["score"] for r in reference]
                dx = max(max(xs) - min(xs), 1) * 0.13
                dy = max(max(ys) - min(ys), 1) * 0.18
                ax.set_xlim(max(0, min(xs) - dx), max(xs) + dx)
                ax.set_ylim(max(-0.5, min(ys) - dy), min(101, max(ys) + dy))
                ax.xaxis.set_major_locator(MaxNLocator(nbins=2))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
                ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
                ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
                ax.tick_params(labelsize=6.5, pad=2)
                for tick in ax.get_xticklabels() + ax.get_yticklabels():
                    tick.set_fontweight("bold")
                ax.grid(color="#D7DDE5", linewidth=0.5)
                ax.set_axisbelow(True)
                if i == 0:
                    ax.set_title(SHORT[task] + "\n" + metric_labels[j],
                                 fontsize=7, fontweight="bold", pad=7)
                if j == 0:
                    position = ax.get_position()
                    fig.text(0.012, (position.y0 + position.y1) / 2, MODEL_LABELS[i],
                             rotation=90, ha="center", va="center", fontsize=8, fontweight="bold")
        fig.supxlabel("Tokens per Question (k)",
                      fontsize=8, fontweight="bold", y=0.065)
        save(fig, directory, f"ircot_cost_cap_{cap}{suffix}")


def iterative_ablation_plots(output, ablation):
    """Join audited QA scores with measured costs from each ablation trajectory."""
    assert read(ablation / "complete.json")["complete"]
    rows = read_csv(output / "data/ircot_cost.csv")
    assert len(rows) == 288
    with (ablation / "qa_results.csv").open() as stream:
        audited = list(csv.DictReader(stream))
    lookup = {(r["model"].replace("/", "_"), r["task"], r["setting"], r["variant"]): r
              for r in audited}
    assert len(lookup) == len(audited)
    for row in rows:
        if row["method"] in ("full", "sources"):
            variant = "full" if row["method"] == "full" else "without_context_augmentation"
            unit = lookup[row["model"], row["task"], f"cap_{int(row['cap'])}", variant]
            assert math.isclose(row["score"], float(unit["score"]), abs_tol=1e-9)
            assert row["input_tokens"] + row["output_tokens"] == int(unit["qa_tokens"])
    for model in MODELS:
        for task in TASKS:
            directory = ablation / "ircot" / model / "without_propagation/main/traces" / task
            costs = {cap: [0, 0] for cap in (1, 3, 5)}
            ids = set()
            for path in sorted(directory.glob("*/*.json")):
                data = read(path)
                assert data["complete"] and not data["pilot"]
                for trace in data["traces"]:
                    assert trace["case_id"] not in ids and trace["rounds"]
                    ids.add(trace["case_id"])
                    for cap in costs:
                        for step in trace["rounds"][:cap]:
                            usage = step["reasoning"]["usage"]
                            for i, field in enumerate(("input_tokens", "output_tokens")):
                                value = usage[field]
                                assert isinstance(value, int) and value >= 0
                                costs[cap][i] += value
            assert len(ids) == COUNTS[task], (directory, len(ids))
            for cap, (inputs, outputs) in costs.items():
                unit = lookup[model, task, f"cap_{cap}", "without_propagation"]
                qa = qa_record(Path(unit["source"]), task, float(unit["score"]) / 100)
                assert qa["input_tokens"] + qa["output_tokens"] == int(unit["qa_tokens"])
                usage_path, = Path(unit["source"]).glob("evaluations/*/qa_usage.jsonl")
                assert ids == {r["case_id"] for r in lines(usage_path)}
                rows.append(dict(model=model, task=task, cap=cap, method="without_propagation",
                    **qa, reasoning_input_tokens=inputs, reasoning_output_tokens=outputs,
                    trace_directory=str(directory),
                    reader_total_per_question=(int(unit["qa_tokens"]) + inputs + outputs) / COUNTS[task]))
    assert len(rows) == 360
    write_csv(output / "data/ircot_cost_with_ablation.csv", rows)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                         "pdf.fonttype": 42, "axes.edgecolor": "#7C8492", "axes.linewidth": 0.7})
    methods = ["bm25_native", "bm25_same_context", "sources", "full", "without_propagation"]
    iteration_cost_plots(rows, output / "figures", methods)
    iterative_legend(output / "figures")
    caption = (
        "Iterative memory retrieval with a maximum of {cap} IRCoT iterations.\n"
        "Rows correspond to LLMs and columns to tasks.\n"
        "The horizontal axis reports average input and output tokens for intermediate reasoning "
        "and final answer generation per question, in thousands.\n"
        "Offline construction, embedding computation, and separate fact recognition calls are excluded.\n"
        "SH-Doc, MH-Doc, FC-SH, and FC-MH use substring exact match; 2Wiki uses answer F1.\n"
        "LoCoMo averages token F1 for non-adversarial questions and binary abstention accuracy "
        "for adversarial questions.\n"
        "BM25 and AMOR are shown with and without context augmentation.\n"
        "AMOR w/o recommendation replaces graph propagation with dense ranking while retaining "
        "BM25 fusion and context augmentation; its reasoning trajectories are generated separately.\n")
    for cap in (1, 3, 5):
        (output / "tables" / f"ircot_cost_cap_{cap}_caption.tex").write_text(caption.format(cap=cap))
    print("Validated and plotted 360 model/task/cap/condition points", flush=True)


def iterative_results_plots(output, results):
    """Replace AMOR points with audited results while preserving BM25 controls."""
    audit = read(results / "audit.json")
    assert audit["complete"] and not audit["missing"]
    assert audit["native_metrics_recomputed"] and audit["contexts_and_trajectories_checked"]
    rows = [r for r in read_csv(output / "data/ircot_cost.csv")
            if r["method"] in ("bm25_native", "bm25_same_context")]
    assert len(rows) == len(MODELS) * len(TASKS) * 3 * 2
    variants = {"full": "full", "without_context_augmentation": "sources",
                "without_propagation": "without_propagation"}
    with (results / "qa_results.csv").open() as stream:
        for unit in csv.DictReader(stream):
            if unit["setting"] == "one_shot":
                continue
            cap = int(unit["setting"].removeprefix("cap_"))
            assert cap in (1, 3, 5)
            count = int(unit["questions"])
            assert count == COUNTS[unit["task"]]
            qa_tokens = int(unit["qa_tokens"])
            inputs = int(unit["reasoning_input_tokens"])
            outputs = int(unit["reasoning_output_tokens"])
            assert min(qa_tokens, inputs, outputs) >= 0
            per_question = (qa_tokens + inputs + outputs) / count
            assert math.isclose(per_question, float(unit["reader_tokens_per_question"]), abs_tol=1e-9)
            rows.append(dict(model=unit["model"].replace("/", "_"), task=unit["task"],
                cap=cap, method=variants[unit["variant"]], questions=count,
                score=float(unit["score"]), qa_tokens=qa_tokens,
                reasoning_input_tokens=inputs, reasoning_output_tokens=outputs,
                reader_total_per_question=per_question, directory=unit["source"],
                trace_directory=unit["trajectory_source"]))
    methods = ["bm25_native", "bm25_same_context", "sources", "full", "without_propagation"]
    expected = {(model, task, cap, method) for model in MODELS for task in TASKS
                for cap in (1, 3, 5) for method in methods}
    actual = {(r["model"], r["task"], r["cap"], r["method"]) for r in rows}
    assert actual == expected and len(rows) == len(expected)
    assert all(r["questions"] == COUNTS[r["task"]] and 0 <= r["score"] <= 100 for r in rows)
    write_csv(output / "data/ircot_cost_simplified.csv", rows)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                         "pdf.fonttype": 42, "axes.edgecolor": "#7C8492", "axes.linewidth": 0.7})
    iteration_cost_plots(rows, output / "figures", methods)
    print(f"Validated and plotted {len(rows)} points; BM25 controls and legend unchanged", flush=True)


def iterative_legend(directory):
    methods = ["full", "sources", "without_propagation", "bm25_same_context", "bm25_native"]
    legend(directory, "legend_iterative", methods, 5, fontsize=12,
           columnspacing=0.8, handletextpad=0.4, handlelength=1.2)


def construction_and_qa(one, offline):
    rows = []
    for row in one:
        method = row["method"]
        stages = [r for r in offline if r["task"] == row["task"] and r["stage"] != "failed_attempt"
                  and (r["method"] == method
                       or method == "libra" and r["method"] == "hipporag2"
                       or method == "lightmem_offline" and r["method"] == "lightmem")]
        assert stages and all(r["missing_usage"] == 0 for r in stages)
        build = sum(r["input_tokens"] + r["output_tokens"] for r in stages)
        total = build + row["input_tokens"] + row["output_tokens"]
        rows.append(dict(row, construction_tokens=build, construction_qa_tokens=total,
                         construction_qa_per_question=total / row["questions"],
                         construction_cache_incomplete=int(method == "catrag")))
    return rows


def plots(one, iterative_rows, retrieval, construction_rows, directory):
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
                         "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 9, "pdf.fonttype": 42,
                         "axes.edgecolor": "#7C8492", "axes.linewidth": 0.7})
    methods = ["bm25_native", "bm25_same_context", "sources", "full"]
    legend(directory, "legend_iterative", methods, 2)
    iteration_cost_plots(iterative_rows, directory, methods)
    combined = construction_and_qa(one, construction_rows)
    write_csv(directory.parent / "data/one_shot_construction_qa_cost.csv", combined)
    construction_totals = []
    for method in METHODS:
        rows = [r for r in combined if r["method"] == method and r["model"] == MODELS[0]]
        assert len(rows) == len(TASKS)
        construction_totals.append(dict(method=method, tasks=len(rows),
                                        construction_tokens=sum(r["construction_tokens"] for r in rows),
                                        construction_cache_incomplete=int(method == "catrag")))
    write_csv(directory.parent / "data/construction_by_method.csv", construction_totals)
    plotted = [r for r in construction_totals if r["method"] not in ("bm25", "dense")]
    construction_methods = [r["method"] for r in plotted]
    fig, ax = plt.subplots(figsize=(3.35, 2.35), layout="constrained")
    values = [r["construction_tokens"] / 1e6 for r in plotted]
    labels = [LABELS[m] for m in construction_methods]
    bars = ax.bar(labels, values, color=[STYLES[m][0] for m in construction_methods],
                  edgecolor="#555555", linewidth=0.5, width=0.65)
    for method, bar, value in zip(construction_methods, bars, values):
        label = f"{value:.2f}"
        ax.annotate(label, (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    fontsize=8, fontweight="bold" if method == "libra" else "normal")
    ax.set_ylabel("Construction tokens (M)", fontsize=9)
    ax.set_ylim(0, max(values) * 1.15)
    ax.set_yticks([0, 40, 80, 120, 160])
    ax.grid(axis="y", color="#D7DDE5", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")
    save(fig, directory, "construction_by_method")


def tables(one, multi, retrieval, offline, directory):
    def table(name, caption, columns, header, body):
        text = [r"\begin{table*}[t]", r"\centering\small", r"\caption{" + caption + "}",
                r"\label{tab:" + name.replace("_", "-") + "}",
                r"\begin{tabular}{" + columns + "}", r"\toprule", header + r" \\", r"\midrule"]
        text += [" & ".join(row) + r" \\" for row in body]
        text += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
        (directory / (name + ".tex")).write_text("\n".join(text) + "\n")

    body, summaries = [], []
    for task in TASKS:
        base = [r for r in offline if r["task"] == task and r["method"] == "hipporag2"]
        names = [r for r in offline if r["task"] == task and r["method"] == "libra" and r["stage"] != "graph_projection"]
        projection = [r for r in offline if r["task"] == task and r["stage"] == "graph_projection"]
        assert len(projection) == (10 if task == "LoCoMo" else 1)
        assert len(names) == 2 * len(projection)
        tokens = lambda rs: sum(r["input_tokens"] + r["output_tokens"] for r in rs)
        row = dict(task=task, sources=sum(r["records"] for r in projection),
                   shared_index_tokens=tokens(base), normalization_tokens=tokens(names),
                   normalization_calls=sum(r["calls"] for r in names),
                   index_seconds=sum(r["seconds"] for r in base),
                   normalization_seconds=sum(r["seconds"] for r in names),
                   projection_seconds=sum(r["seconds"] for r in projection))
        summaries.append(row)
        body.append([SHORT[task], f'{row["sources"]:.0f}', f'{row["shared_index_tokens"]/1e6:.3f}',
                     f'{row["normalization_tokens"]/1e6:.3f}', f'{row["normalization_calls"]:.0f}',
                     f'{row["projection_seconds"]:.1f}'])
    table("construction_cost", "Construction costs of our method. Shared extraction/indexing and additional relation normalization are reported separately in millions of LLM tokens. Projection time is the recorded CPU-stage elapsed time with extraction and normalization cached, not end-to-end construction latency. No additional LLM calls are made by projection.",
          "lrrrrr", "Task & Records & Shared tokens (M) & Normalization (M) & Norm. calls & Projection (s)", body)
    write_csv(directory.parent / "data/construction_summary.csv", summaries)
    body = []
    for method in METHODS:
        rs = [r for r in offline if
              (r["method"] == method or method == "lightmem_offline" and r["method"] == "lightmem"
               or method == "libra" and r["method"] == "hipporag2") and r["stage"] != "failed_attempt"]
        if not rs:
            continue
        sums = [sum(r[k] for r in rs) if all(r.get(k) is not None for r in rs) else None
                for k in ("input_tokens", "output_tokens", "calls")]
        note = "Observed; cache-dependent" if method == "catrag" else "Recorded build stages"
        if method == "anchormem_official":
            note = "Includes recovered build logs"
        body.append([LABELS[method], "--" if sums[0] is None else f'{sums[0]/1e6:.3f}',
                     "--" if sums[1] is None else f'{sums[1]/1e6:.3f}',
                     "--" if sums[2] is None else f'{sums[2]:.0f}', note])
    table("offline_baseline_cost", "Recorded offline LLM usage across all six tasks, in millions of input/output tokens. Our method includes the shared extraction/indexing cost plus both normalization stages. LightMem includes offline consolidation. AnchorMem includes provider usage from explicitly referenced recovered builds. CatRAG's resumed-run logs may omit inherited cache costs, so its values are observed usage rather than a complete cold-build comparison. Embedding computation is excluded. Failed attempts are preserved separately in the CSV.",
          "lrrrl", "Method & Input (M) & Output (M) & Calls & Coverage", body)
    body = []
    for model, label in zip(MODELS, MODEL_LABELS):
        for method in METHODS:
            rows = [r for r in one if r["model"] == model and r["method"] == method]
            assert len(rows) == 6
            total = sum(r["input_tokens"] + r["output_tokens"] for r in rows)
            body.append([label, LABELS[method], f'{sum(r["input_tokens"] for r in rows)/1e6:.3f}',
                         f'{sum(r["output_tokens"] for r in rows)/1e3:.1f}', f'{total/3386:.1f}'])
    table("qa_cost", "Final one-shot QA token usage over all 3,386 questions. Input tokens are in millions and output tokens in thousands. The per-question total includes both. Costs exclude offline construction, online fact recognition and embeddings; token counts from different model families are not equivalent compute costs.",
          "llrrr", "Model & Method & Input (M) & Output (k) & Tokens/question", body)
    body = [[LABELS[r["method"]], f'{r["recall"]:.3f}', f'{r["precision"]:.3f}',
             f'{r["all_support_percent"]:.2f}'] for r in retrieval if r["setting"] == "one_shot"]
    table("retrieval_results", "One-shot passage retrieval on all 1,000 2Wiki questions. All-support@5 is the percentage of questions with all annotated supporting passages in the top five original passages. Facts appended to QA context are not counted as passages. Methods returning generated memories are not scored without a verified passage mapping.",
          "lrrr", "Method & Recall@5 & Precision@5 & All-support@5", body)
    body = []
    for model, label in zip(MODELS, MODEL_LABELS):
        for cap in (1, 3, 5):
            for method in ("bm25", "optimized_graph"):
                r, = [r for r in retrieval if r["model"] == model and r["method"] == method and r["cap"] == cap]
                body.append([label, "BM25" if method == "bm25" else "AMOR", str(cap),
                             f'{r["recall"]:.2f}', f'{r["all_support_percent"]:.2f}', f'{r["mean_passages"]:.2f}'])
    table("ircot_retrieval", "IRCoT supporting-passage coverage on 2Wiki. Recall is measured over the accumulated evidence set (at most 15 passages), not a top-5 ranking. Caps are maximum iterations; trajectories may terminate earlier. The raw and enhanced-context variants of each retriever share these retrieved passages.",
          "llrrrr", "Model & Retriever & Cap & Recall (\\%) & All-support (\\%) & Mean passages", body)
    body = []
    for model, label in zip(MODELS, MODEL_LABELS):
        for cap in (1, 3, 5):
            for method in ("bm25_native", "bm25_same_context", "full"):
                rs = [r for r in multi if r["model"] == model and r["cap"] == cap and r["method"] == method]
                totals = [sum(r[k] for r in rs) for k in ("reasoning_input_tokens", "reasoning_output_tokens", "input_tokens", "output_tokens")]
                body.append([label, LABELS[method], str(cap), f'{(totals[0]+totals[1])/1e6:.3f}',
                             f'{(totals[2]+totals[3])/1e6:.3f}', f'{sum(totals)/3386:.1f}'])
    table("ircot_reader_cost", "IRCoT reasoning and final QA token usage over 3,386 questions. Reasoning cost is counted once per evaluated trajectory prefix; larger caps are separate settings, not summed as independent trajectories. Online graph recognition and offline construction are excluded and reported separately.",
          "llrrrr", "Model & Method & Cap & Reasoning (M) & Final QA (M) & Tokens/question", body)
    body = []
    recognition = read_csv(directory.parent / "data/recognition_observed.csv")
    for task in TASKS:
        r, = [r for r in recognition if r["method"] == "libra" and r["task"] == task]
        body.append(["One-shot", SHORT[task], "--", f'{r["calls"]:.0f}',
                     f'{r["input_tokens"]/1e6:.3f}', f'{r["output_tokens"]/1e6:.3f}'])
    for model, label in zip(MODELS, MODEL_LABELS):
        for cap in (1, 3, 5):
            rs = [r for r in multi if r["model"] == model and r["cap"] == cap and r["method"] == "full"]
            assert sum(r["recognition_missing_usage"] for r in rs) == 0
            body.append(["IRCoT", label, str(cap), f'{sum(r["recognition_calls_observed"] for r in rs):.0f}',
                         f'{sum(r["recognition_input_observed"] for r in rs)/1e6:.3f}',
                         f'{sum(r["recognition_output_observed"] for r in rs)/1e6:.3f}'])
    table("recognition_cost", "Observed fact-recognition usage of our method for Qwen3-30B-A3B-Instruct-2507. One-shot rows are per task and shared across readers; IRCoT rows cover all six tasks for each reader and cap. The logger runs below the cache: these are recorded provider calls, not complete cold-query costs. Missing cache-hit usage is not treated as zero cold-run cost. Tokens are in millions.",
          "llrrrr", "Setting & Task / reader & Cap & Calls & Input (M) & Output (M)", body)
    captions = [
        ("construction_by_method", "Recorded offline construction LLM token usage across six tasks, including input and output tokens and counting each index once. BM25 and dense retrieval require no construction-stage LLM calls and are omitted. Embedding computation and online retrieval and QA are excluded."),
        ("one_shot_construction_qa_cost", "One-shot QA performance versus offline construction plus QA LLM tokens, amortized over all evaluated questions of each task. Each panel counts one index construction and the complete QA workload for the specified LLM; the same physical index is reused across LLMs. Construction includes shared extraction and additional normalization for our method, offline consolidation for LightMem, and recovered build logs for AnchorMem. CatRAG construction usage is a lower bound because inherited cache costs may be absent. Online retrieval calls, embedding computation and CPU work are excluded. Tokens from extraction and QA models are summed without compute or price weighting; this is not an end-to-end compute or dollar comparison."),
        ("ircot_iterations", "Iterative QA across four LLMs and six tasks. Curves show maximum iteration counts of 1, 3 and 5. Our method uses graph and BM25 retrieval with factual context at final QA. We also report BM25 with the same final context procedure and graph retrieval with records only. All conditions use one seed; no error bars are shown."),
        ("ircot_reader_cost", "Cost--performance tradeoff for iterative QA. The horizontal axis counts reasoning plus final QA input and output tokens per question, in thousands. Points along each curve correspond to caps 1, 3 and 5, in that order. Offline indexing, embeddings and online fact recognition are excluded; this is reader token usage, not total system compute or dollar cost."),
        ("one_shot_qa_cost", "One-shot QA performance versus final QA tokens per question. All seven reported baselines and our method use the same task instances. LightMem includes offline consolidation; AnchorMem uses official retrieval. Token usage excludes construction and retrieval. Each panel uses its task metric, with no cross-task score averaging."),
        ("ircot_document_tasks", "IRCoT results on document QA tasks. Curves show maximum iteration counts of 1, 3 and 5 for all four LLMs. Raw and enhanced-context variants share each retriever's fixed trajectories. One seed; no error bars."),
        ("ircot_memory_tasks", "IRCoT results on factual consolidation and conversational memory. All four LLMs and iteration caps 1, 3 and 5 are reported. Context enhancement is applied only at final QA. One seed; no error bars."),
        ("ircot_2wiki_recall_qa", "Evidence retrieval and downstream QA on 2Wiki. The horizontal axis measures supporting-passage recall over the accumulated IRCoT evidence set (at most 15 passages); the vertical axis is answer F1. Within each curve, points from left to right correspond to iteration caps 1, 3 and 5. Raw and enhanced context share a retriever's trajectory. One seed; no error bars."),
        ("ours_construction_tokens", "Offline LLM token usage for our method, separating shared information extraction and indexing from additional relation normalization. Graph projection adds no LLM calls. Embedding-model computation and CPU projection time are not represented by these token counts.")]
    for cap in (1, 3, 5):
        captions.append((f"ircot_cost_cap_{cap}",
                         f"IRCoT with BM25 or AMOR at a maximum of {cap} {'iteration' if cap == 1 else 'iterations'}. "
                         "Both selection methods are evaluated with and without final context augmentation, "
                         "which adds selected facts and neighboring conversational text. "
                         "Augmented and unaugmented variants reuse each method's reasoning trajectory. "
                         "Rows show four LLMs and columns show six tasks. The vertical axis is substring exact match "
                         "for SH-Doc, MH-Doc, FC-SH and FC-MH, and answer token F1 for 2Wiki. "
                         "LoCoMo reports the question-level mean of token F1 for non-adversarial questions "
                         "and binary abstention accuracy for adversarial questions. All metrics are percentages. "
                         "The horizontal axis counts mean LLM input plus output tokens per question, in thousands, "
                         "summed over intermediate reasoning calls and final answer generation. "
                         "Corresponding panels use identical axis limits across iteration caps. "
                         "Offline construction, embeddings, and separate fact-recognition calls are excluded. "
                         "All results use one seed; no error bars are shown."))
        captions.append((f"ircot_cost_cap_{cap}_ablation",
                         f"Context ablations for IRCoT with a maximum of {cap} iterations. "
                         "BM25 selects original passages by lexical matching; AMOR combines graph and lexical rankings. "
                         "Context augmentation adds selected facts and neighboring conversational text only for final QA. "
                         "The unaugmented and augmented variants share each method's reasoning trajectory. "
                         "Axes and token accounting follow the corresponding main comparison."))
    captions = [(name, caption) for name, caption in captions
                if name in {"construction_by_method", "ircot_cost_cap_1", "ircot_cost_cap_3", "ircot_cost_cap_5"}]
    blocks, caption_blocks = [], []
    for name, caption in captions:
        legend_name = "legend_construction" if name == "ours_construction_tokens" else "legend_shared"
        if name.startswith("ircot_"):
            legend_name = "legend_iterative_ablation"
        if name.startswith("ircot_cost_cap_"):
            legend_name = "legend_iterative_ablation" if name.endswith("_ablation") else "legend_iterative"
        caption_blocks.extend(["% " + name + ".pdf", r"\caption{" + caption + "}",
                               r"\label{fig:" + name.replace("_", "-") + "}", ""])
        environment = "figure" if name == "construction_by_method" else "figure*"
        width = r"\linewidth" if name == "construction_by_method" else r"\textwidth"
        blocks.extend([r"\begin{" + environment + "}[t]", r"\centering"])
        if name != "construction_by_method":
            blocks.append(r"\includegraphics[width=\textwidth]{fig/" + legend_name + r".pdf}\par")
        blocks.extend([r"\includegraphics[width=" + width + "]{fig/" + name + ".pdf}",
                       r"\caption{" + caption + "}", r"\label{fig:" + name.replace("_", "-") + "}",
                       r"\end{" + environment + "}", ""])
    (directory / "figures.tex").write_text("\n".join(blocks))
    (directory / "captions.tex").write_text("\n".join(caption_blocks))
    base = sum(r["shared_index_tokens"] for r in summaries)
    norm = sum(r["normalization_tokens"] for r in summaries)
    rec = {r["method"]: r for r in retrieval if r["setting"] == "one_shot"}
    def wins(reference):
        comparisons = []
        for full in [r for r in multi if r["method"] == "full"]:
            other, = [r for r in multi if r["method"] == reference and all(r[k] == full[k] for k in ("model", "task", "cap"))]
            comparisons.append(full["score"] - other["score"])
        return sum(v > 1e-10 for v in comparisons), sum(abs(v) <= 1e-10 for v in comparisons), sum(v < -1e-10 for v in comparisons)
    ratios = []
    for model in MODELS:
        total = lambda m: sum(r["reader_total_per_question"] * r["questions"] for r in multi
                              if r["model"] == model and r["method"] == m and r["cap"] == 3)
        ratios.append(total("full") / total("bm25_native"))
    combined = construction_and_qa(one, offline)
    frontier_count = 0
    for row in [r for r in combined if r["method"] == "libra"]:
        alternatives = [r for r in combined if r["model"] == row["model"] and r["task"] == row["task"]]
        dominated = any(r["score"] >= row["score"]
                        and r["construction_qa_per_question"] <= row["construction_qa_per_question"]
                        and (r["score"] > row["score"]
                             or r["construction_qa_per_question"] < row["construction_qa_per_question"])
                        for r in alternatives)
        frontier_count += not dominated
    text = rf"""\subsection{{Construction Cost}}
Table~\ref{{tab:construction-cost}} separates shared extraction and indexing from our additional graph construction stages. Across the six tasks, shared extraction and indexing consume {base/1e6:.2f} million LLM tokens. Relation normalization adds {norm/1e6:.2f} million tokens ({100*norm/base:.1f}\%), while graph projection requires no further LLM calls. Its recorded CPU-stage time totals {sum(r['projection_seconds'] for r in summaries):.1f} seconds across all 15 source groups, with extraction and normalization already cached. Normalization overhead varies substantially by task, so the cached projection time should not be interpreted as the cost of constructing memory from raw text.

\subsection{{Cost--Performance Tradeoff}}
Figure~\ref{{fig:one-shot-construction-qa-cost}} reports offline construction plus QA LLM tokens per question. For each task and downstream LLM, we count one index construction and QA over all evaluated questions, then divide by the number of questions. These panels represent separate deployment scenarios; the shared index is not rebuilt for each LLM in the actual experiments. Our method lies on the observed token--performance Pareto frontier in {frontier_count} of 24 settings. This does not establish optimality under a fixed compute budget: extraction and QA tokens are summed without price or compute weighting, and online retrieval, embeddings and CPU work are excluded. CatRAG construction usage is a lower bound where inherited cache costs are missing.

Figures~\ref{{fig:one-shot-qa-cost}} and~\ref{{fig:ircot-reader-cost}} report performance against token usage within each task and LLM. For IRCoT, we count both intermediate reasoning and final QA tokens. At a three-iteration cap, our method uses {min(ratios):.2f}--{max(ratios):.2f} times the reader tokens of native IRCoT with BM25, aggregated over the same 3,386 questions. This comparison excludes separate fact-recognition calls, embedding computation, and offline construction; it is not a total-system compute comparison. Table~\ref{{tab:recognition-cost}} reports observed fact-recognition usage separately, with its cache-coverage limitations. Our method improves many task scores, but does not uniformly reduce QA context cost.

\subsection{{Evidence Retrieval}}
Table~\ref{{tab:retrieval-results}} evaluates original-passage retrieval independently of answer generation. On 2Wiki, our method achieves {rec['libra']['recall']:.3f}\% Recall@5, compared with {rec['hipporag2']['recall']:.3f}\% for HippoRAG 2 and {rec['bm25']['recall']:.3f}\% for BM25. The percentage of questions with all supporting passages retrieved increases from {rec['hipporag2']['all_support_percent']:.1f}\% with HippoRAG 2 to {rec['libra']['all_support_percent']:.1f}\%. Under IRCoT, our method has higher supporting-passage recall than BM25 for all four LLMs at each reported cap. However, higher recall does not always yield higher answer F1, as shown in Figure~\ref{{fig:ircot-2wiki-recall-qa}}. Evidence coverage and successful answer generation are therefore distinct outcomes.

\subsection{{Iterative Question Answering}}
Across the 72 task--model--iteration settings, our method records {wins('bm25_native')[0]} wins, {wins('bm25_native')[1]} ties and {wins('bm25_native')[2]} losses against native IRCoT with BM25. Against BM25 using the same final context construction procedure, the counts are {wins('bm25_same_context')[0]}, {wins('bm25_same_context')[1]} and {wins('bm25_same_context')[2]}, respectively. Figures~\ref{{fig:ircot-document-tasks}} and~\ref{{fig:ircot-memory-tasks}} retain all iteration caps and tasks, including settings where additional iterations do not improve performance.

\paragraph{{Reporting Scope.}}
All results use one seed, and the evaluation data were used during method development. No confidence intervals or significance claims are made. Token counts include full logical input and output tokens, without discounts for prefix caching. Historical stage timings were collected with different execution and cache conditions and are not matched-hardware latency benchmarks. Missing or cache-bypassed usage is not imputed as zero cold-run cost. Prompt formats and task-specific scoring are unchanged.
"""
    (directory / "paper_analysis.tex").write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/oscar/scratch/zliu328/agent-memory-outputs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render-only", action="store_true", help="Reuse validated CSV files to update plots and tables")
    parser.add_argument("--iterative-ablation", type=Path,
                        help="Update only IRCoT figures and legend using audited component ablations")
    parser.add_argument("--iterative-results", type=Path,
                        help="Update the three IRCoT figures from audited QA and reasoning token totals")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data_dir, figure_dir, table_dir = [args.output / name for name in ("data", "figures", "tables")]
    for directory in (data_dir, figure_dir, table_dir):
        directory.mkdir(exist_ok=True)
    if args.iterative_results:
        iterative_results_plots(args.output, args.iterative_results)
        return
    if args.iterative_ablation:
        iterative_ablation_plots(args.output, args.iterative_ablation)
        return
    if args.render_only:
        one, multi, offline, retrieval = [read_csv(data_dir / (name + ".csv")) for name in
            ("one_shot_qa_cost", "ircot_cost", "construction_stages", "retrieval")]
        one_shot_context_ablation(args.root, one, data_dir, table_dir)
        plots(one, multi, retrieval, offline, figure_dir)
        tables(one, multi, retrieval, offline, table_dir)
        return
    one = one_shot(args.root)
    one_shot_context_ablation(args.root, one, data_dir, table_dir)
    print("Validated one-shot QA:", len(one), flush=True)
    multi = iterative(args.root)
    print("Validated iterative QA:", len(multi), flush=True)
    offline, recognition = construction(args.root)
    retrieval = retrieval_metrics(args.root)
    for name, rows in [("one_shot_qa_cost", one), ("ircot_cost", multi), ("construction_stages", offline),
                       ("recognition_observed", recognition), ("retrieval", retrieval)]:
        write_csv(data_dir / (name + ".csv"), rows)
    plots(one, multi, retrieval, offline, figure_dir)
    tables(one, multi, retrieval, offline, table_dir)
    print("Artifacts written to", args.output, flush=True)


if __name__ == "__main__":
    main()

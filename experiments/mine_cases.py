"""Audit paired QA outcomes and export real contexts for qualitative analysis."""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from experiments.analyze_failures import case_key, reference_answer_fields
from experiments.report_paper_costs import COUNTS, METHODS, MODELS, SHORT, TASKS
from utils.hipporag_metrics import hipporag_answer_f1
from utils.locomo_metrics import locomo_qa_f1
from utils.metrics import substring_exact_match


LABELS = dict(zip(METHODS, ["BM25", "Dense", "HippoRAG 2", "CatRAG", "Mem0",
                          "LightMem", "AnchorMem", "AMOR"]))
LABELS.update(full="AMOR", sources="AMOR w/o context augmentation",
              bm25_native="BM25 w/o context augmentation", bm25_same_context="BM25 w/ context augmentation",
              without_propagation="AMOR w/o recommendation",
              without_context_augmentation="AMOR w/o context augmentation",
              without_projection="AMOR w/o projection")
COLORS = {"higher": "#6488B3", "equal": "#E3E6EB", "lower": "#C8877D"}


def read_lines(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def csv_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score(case, prediction):
    if case["metric"] == "subem":
        return "substring_exact_match", float(substring_exact_match(prediction, case["answers"]))
    if case["metric"] == "locomo":
        return "f1", locomo_qa_f1(prediction, case["answers"][0], case["category"])
    return "answer_f1", hipporag_answer_f1(prediction, case["answers"])


def manifest(root):
    entries = {}
    costs = root / "paper_costs_20260925/data"
    for row in csv_rows(costs / "one_shot_qa_cost.csv"):
        entries["one_shot", row["model"], row["task"], row["method"]] = row
    for row in csv_rows(root / "optimization_paper_ablation_seed42_20260927/qa_results.csv"):
        if row["variant"] not in ("without_propagation", "without_projection", "without_context_augmentation"):
            continue
        if row["setting"] != "one_shot":
            continue
        key = (row["setting"], row["model"].replace("/", "_"), row["task"], row["variant"])
        entries[key] = dict(row, directory=row["source"])
    for row in csv_rows(costs / "ircot_cost_with_ablation.csv"):
        entries[f"cap_{int(float(row['cap']))}", row["model"], row["task"], row["method"]] = row
    return entries


def audit_predictions(entry, cases, task, method):
    directory = Path(entry["directory"])
    path, = directory.glob("evaluations/*/predictions.jsonl")
    records = {}
    for row in read_lines(path):
        key = case_key(row)
        assert key in cases and key not in records, (path, key)
        case = cases[key]
        metric, value = score(case, row["prediction"])
        assert math.isclose(value, row["metrics"][metric], abs_tol=1e-12)
        item = dict(prediction=row["prediction"], score=value, metric=metric)
        if task == "2WikiMultiHopQA":
            # Only original-source methods have a valid passage mapping here.
            original = method in {"bm25", "dense", "hipporag2", "catrag", "bm25_native"}
            texts = []
            for retrieved in row["retrieved"]:
                metadata = retrieved.get("metadata", {})
                if "original_source_text" in metadata:
                    texts.append(metadata["original_source_text"])
                elif original:
                    texts.append(retrieved["text"])
            if texts:
                gold = set(case["gold_passages"])
                assert gold
                item.update(gold_passages=len(gold), supporting_passages=len(gold & set(texts)),
                            all_support=gold.issubset(texts))
        records[key] = item
    assert set(records) == set(cases) and len(records) == COUNTS[task]
    assert math.isclose(100 * sum(r["score"] for r in records.values()) / len(records),
                        float(entry["score"]), abs_tol=1e-8), path
    return records


def analyze(root, output):
    entries = manifest(root)
    write_json(output / "manifest.json", [dict(setting=s, model=m, task=t, method=v,
               directory=e["directory"]) for (s, m, t, v), e in entries.items()])
    summary, evidence, categories = [], [], []
    concordant = defaultdict(list)
    conditions = predictions = comparisons = 0
    with (output / "question_scores.jsonl").open("w") as questions:
        for setting in ("one_shot", "cap_1", "cap_3", "cap_5"):
            full = "libra" if setting == "one_shot" else "full"
            for model in MODELS:
                for task in TASKS:
                    subset = {v: e for (s, m, t, v), e in entries.items()
                              if (s, m, t) == (setting, model, task)}
                    assert len(subset) == (11 if setting == "one_shot" else 5)
                    cases = {}
                    for row in read_lines(Path(subset[full]["directory"]) / "retrieval.jsonl"):
                        key = case_key(row)
                        assert key not in cases
                        cases[key] = row["case"]
                    assert len(cases) == COUNTS[task]
                    records = {method: audit_predictions(entry, cases, task, method)
                               for method, entry in subset.items()}
                    conditions += len(records)
                    predictions += len(records) * len(cases)
                    for key, case in cases.items():
                        unit = dict(setting=setting, model=model, task=task, group_id=key[0],
                                    case_id=key[1], question=case["question"],
                                    **reference_answer_fields(case),
                                    conditions={method: values[key] for method, values in records.items()})
                        questions.write(json.dumps(unit, ensure_ascii=False) + "\n")
                    for method, values in records.items():
                        if method == full:
                            continue
                        counts = Counter()
                        by_category = defaultdict(list)
                        evidence_counts = Counter()
                        for key, baseline in values.items():
                            ours = records[full][key]
                            a, b = ours["score"], baseline["score"]
                            direction = "higher" if a > b else "lower" if a < b else "equal"
                            counts[direction] += 1
                            comparisons += 1
                            by_category[cases[key].get("category")].append((a, b))
                            if setting == "one_shot" and direction != "equal":
                                concordant[task, method, key[0], key[1], direction].append(model)
                            if "all_support" in ours and "all_support" in baseline:
                                evidence_counts[baseline["all_support"], ours["all_support"], direction] += 1
                        summary.append(dict(setting=setting, model=model, task=task, comparator=method,
                                            questions=len(cases), **{k: counts[k] for k in COLORS}))
                        for (b, a, direction), n in evidence_counts.items():
                            evidence.append(dict(setting=setting, model=model, task=task, comparator=method,
                                                 baseline_all_support=b, amor_all_support=a,
                                                 score_direction=direction, questions=n))
                        if task == "LoCoMo":
                            for category, pairs in by_category.items():
                                categories.append(dict(setting=setting, model=model, comparator=method,
                                    category=category, questions=len(pairs),
                                    amor_score=100 * sum(a for a, _ in pairs) / len(pairs),
                                    baseline_score=100 * sum(b for _, b in pairs) / len(pairs)))
                    print(setting, model, task, len(records), "conditions audited", flush=True)
    assert conditions == 624
    write_csv(output / "paired_outcomes.csv", summary)
    write_csv(output / "support_transitions.csv", evidence)
    write_csv(output / "locomo_categories.csv", categories)
    write_json(output / "consistent_candidates.json", [dict(task=t, comparator=v, group_id=g,
        case_id=c, direction=d, models=ms) for (t, v, g, c, d), ms in concordant.items()
        if len(ms) == len(MODELS)])
    write_json(output / "audit.json", dict(complete=True, conditions=conditions,
        predictions=predictions, paired_comparisons=comparisons, tasks=TASKS, models=MODELS,
        native_scores_recomputed=True, full_question_sets=True,
        note="Descriptive reused test-as-dev results; no new split, tuning, or significance test."))


def plot(output):
    import textwrap
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "axes.labelweight": "bold"})
    rows = csv_rows(output / "paired_outcomes.csv")
    figdir = output / "figures"
    figdir.mkdir(exist_ok=True)
    for setting in ("one_shot", "cap_1", "cap_3", "cap_5"):
        methods = METHODS[:-1] if setting == "one_shot" else [
            "bm25_native", "bm25_same_context", "sources", "without_propagation"]
        fig, axes = plt.subplots(2, 3, figsize=(9.8, 4.5), sharex=True, sharey=True)
        fig.subplots_adjust(left=.19, right=.99, bottom=.13, top=.92, wspace=.16, hspace=.35)
        for ax, task in zip(axes.flat, TASKS):
            left = [0.] * len(methods)
            for direction in COLORS:
                widths = []
                for method in methods:
                    selected = [r for r in rows if (r["setting"], r["task"], r["comparator"]) ==
                                (setting, task, method)]
                    assert len(selected) == 4
                    widths.append(100 * sum(int(r[direction]) for r in selected) / (4 * COUNTS[task]))
                ax.barh(range(len(methods)), widths, left=left, color=COLORS[direction],
                        height=.67, edgecolor="white", linewidth=.4)
                for j, width in enumerate(widths):
                    if width >= 9:
                        ax.text(left[j] + width / 2, j, f"{width:.0f}", ha="center", va="center",
                                fontsize=7, color="black", fontweight="bold")
                left = [a + b for a, b in zip(left, widths)]
            ax.set_title(f"{SHORT[task]} (n = {4 * COUNTS[task]:,})", fontweight="bold", fontsize=9)
            ax.set_yticks(range(len(methods)), [textwrap.fill(LABELS[m], 24) for m in methods])
            ax.set_xlim(0, 100)
            ax.set_xticks([0, 50, 100])
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.tick_params(axis="y", length=0)
        axes[0, 0].invert_yaxis()
        fig.supxlabel("Question and LLM pairs (%)", fontsize=9, fontweight="bold", y=.02)
        for extension in ("pdf", "png", "svg"):
            fig.savefig(figdir / f"paired_outcomes_{setting}.{extension}", bbox_inches="tight", dpi=200)
        plt.close(fig)
    fig = plt.figure(figsize=(7.2, .32))
    legend_labels = {"higher": "Higher AMOR score", "equal": "Equal scores", "lower": "Lower AMOR score"}
    fig.legend(handles=[Patch(facecolor=c, label=legend_labels[d]) for d, c in COLORS.items()],
               ncol=3, loc="center", frameon=False, prop={"weight": "bold", "size": 9})
    for ext in ("pdf", "png", "svg"):
        fig.savefig(figdir / f"legend_paired_outcomes.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def export_cases(output, selection):
    requests = json.loads(selection.read_text())
    entries = json.loads((output / "manifest.json").read_text())
    manifest_map = {(r["setting"], r["model"], r["task"], r["method"]): Path(r["directory"])
                    for r in entries}
    directory = output / "cases"
    directory.mkdir(exist_ok=True)
    for request in requests:
        case = dict(request, conditions={})
        for method in request["methods"]:
            source = manifest_map[request["setting"], request["model"], request["task"], method]
            match = lambda r: (r.get("case_id", r.get("case", {}).get("case_id")) == request["case_id"]
                               and ("group_id" not in request or str(r["group_id"]) == request["group_id"]))
            inputs = next(r for r in read_lines(source / "retrieval.jsonl") if match(r))
            prediction_path, = source.glob("evaluations/*/predictions.jsonl")
            predictions = next(r for r in read_lines(prediction_path) if match(r))
            assert inputs["retrieved"] == predictions["retrieved"]
            case["conditions"][method] = dict(directory=str(source), input=inputs, prediction=predictions)
        write_json(directory / (request["name"] + ".json"), case)
        print("Exported", request["name"], flush=True)


def failure_mining(root, output, results=None):
    """Enumerate native scores and paired outcomes, without inferring failure causes."""
    entries = {key: value for key, value in manifest(root).items() if key[0] == "one_shot"}
    controls = {"without_propagation", "without_projection", "without_context_augmentation"}
    if results is not None:
        controls.remove("without_projection")
        entries = {key: value for key, value in entries.items() if key[3] != "without_projection"}
        for row in csv_rows(results / "qa_results.csv"):
            if row["setting"] != "one_shot":
                continue
            method = "libra" if row["variant"] == "full" else row["variant"]
            if method not in {"libra"} | controls:
                continue
            key = ("one_shot", row["model"].replace("/", "_"), row["task"], method)
            entries[key] = dict(row, directory=row["source"])
    write_json(output / "manifest.json", [dict(setting=s, model=m, task=t, method=v,
               directory=e["directory"]) for (s, m, t, v), e in entries.items()])
    summaries, categories, candidates, support = [], [], [], []
    predictions = 0
    with (output / "failure_questions.jsonl").open("w") as stream:
        for task in TASKS:
            source = Path(entries["one_shot", MODELS[0], task, "libra"]["directory"])
            cases = {case_key(row): row["case"] for row in read_lines(source / "retrieval.jsonl")}
            assert len(cases) == COUNTS[task]
            records = {}
            for model in MODELS:
                subset = {v: e for (s, m, t, v), e in entries.items() if (m, t) == (model, task)}
                assert set(subset) == set(METHODS) | controls
                records[model] = {v: audit_predictions(e, cases, task, v) for v, e in subset.items()}
                predictions += len(subset) * len(cases)
                for method, values in records[model].items():
                    mapped = [r for r in values.values() if "supporting_passages" in r]
                    if mapped:
                        assert len(mapped) == len(cases)
                        for state in ("none", "partial", "all"):
                            selected = [r for r in mapped if ("all" if r["all_support"] else
                                        "none" if r["supporting_passages"] == 0 else "partial") == state]
                            support.append(dict(task=task, model=model, method=method, support=state,
                                questions=len(selected), zero_score=sum(r["score"] == 0 for r in selected),
                                full_score=sum(r["score"] == 1 for r in selected),
                                mean_score=sum(r["score"] for r in selected) / len(selected) if selected else None))
                    groups = {None: list(cases)}
                    if task == "LoCoMo":
                        groups.update({category: [k for k, c in cases.items() if c["category"] == category]
                                       for category in sorted({c["category"] for c in cases.values()})})
                    for category, keys in groups.items():
                        counts = Counter()
                        for key in keys:
                            a, b = records[model]["libra"][key]["score"], values[key]["score"]
                            counts["zero" if b == 0 else "full" if b == 1 else "partial"] += 1
                            counts["amor_higher" if a > b else "amor_lower" if a < b else "equal"] += 1
                            counts["zero_to_full"] += b == 0 and a == 1
                            counts["full_to_zero"] += b == 1 and a == 0
                        row = dict(task=task, model=model, method=method, category=category,
                                   questions=len(keys), mean_score=sum(values[k]["score"] for k in keys) / len(keys),
                                   **{name: counts[name] for name in ("zero", "partial", "full", "amor_higher",
                                      "equal", "amor_lower", "zero_to_full", "full_to_zero")})
                        (summaries if category is None else categories).append(row)
            for position, (key, case) in enumerate(cases.items()):
                outcomes = {m: {v: rows[key] for v, rows in methods.items()} for m, methods in records.items()}
                item = dict(task=task, group_id=key[0], case_id=key[1], position=position,
                            question=case["question"], **reference_answer_fields(case), conditions=outcomes)
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
                for method in METHODS[:-1]:
                    baseline = [outcomes[m][method]["score"] for m in MODELS]
                    amor = [outcomes[m]["libra"]["score"] for m in MODELS]
                    candidates.append(dict(task=task, group_id=key[0], case_id=key[1], position=position,
                        method=method, question=case["question"], answers=case["answers"],
                        category=case.get("category"), baseline_scores=baseline, amor_scores=amor,
                        all_baseline_zero=all(b == 0 for b in baseline),
                        all_amor_zero=all(a == 0 for a in amor),
                        all_baseline_full=all(b == 1 for b in baseline),
                        all_amor_full=all(a == 1 for a in amor)))
            print(task, len(cases), "questions; all seven baselines and controls audited", flush=True)
    write_csv(output / "failure_outcomes.csv", summaries)
    write_csv(output / "failure_locomo_categories.csv", categories)
    write_csv(output / "failure_support_outcomes.csv", support)
    write_json(output / "failure_candidates.json", candidates)
    grouped = defaultdict(list)
    for row in candidates:
        grouped[row["task"], row["method"]].append(row)
    write_csv(output / "failure_candidate_summary.csv", [dict(task=task, method=method, questions=len(rows),
        zero_on_all_readers=sum(r["all_baseline_zero"] for r in rows),
        zero_to_full_on_all_readers=sum(r["all_baseline_zero"] and r["all_amor_full"] for r in rows),
        full_to_zero_on_all_readers=sum(r["all_baseline_full"] and r["all_amor_zero"] for r in rows))
        for (task, method), rows in grouped.items()])
    write_json(output / "failure_audit.json", dict(complete=True, tasks=TASKS, models=MODELS,
               external_baselines=METHODS[:-1], conditions=len(entries), predictions=predictions,
               native_scores_recomputed=True, full_question_sets=True,
               note="All questions retained. Zero/partial/full are native metric values, not semantic labels. "
                    "No failure causes assigned automatically. Reused development evaluation; no new split."))


def inspect_stores(specification, output):
    """Read exact stored entries for manual case inspection, not automatic labeling."""
    import pickle
    import sqlite3

    reports = []
    for request in json.loads(specification.read_text()):
        path = Path(request["path"])
        if request["format"] == "qdrant":
            with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
                points = [pickle.loads(blob) for blob, in connection.execute("SELECT point FROM points")]
            records = [dict(id=str(point.id), text=point.payload[request["field"]]) for point in points]
        elif request["format"] == "compiled":
            data = json.loads(path.read_text())
            records = [dict(id=key, **{field: row[field] for field in request["fields"]})
                       for key, row in data.items()]
        else:
            data = json.loads(path.read_text())
            records = [dict(id=row["idx"], **{field: row[field] for field in request["fields"]})
                       for row in data[request["collection"]]]
        matches = {term: [r for r in records if term.lower() in json.dumps(r, ensure_ascii=False).lower()]
                   for term in request["terms"]}
        reports.append(dict(request, stored_entries=len(records), matches=matches))
        print(request["name"], len(records), {t: len(rs) for t, rs in matches.items()}, flush=True)
    write_json(output / "failure_store_evidence.json", reports)


def review_cases(specification, output):
    notes = json.loads(specification.read_text())
    selected = {r["name"]: r for r in json.loads((output / "failure_selection.json").read_text())}
    scores = {(r["task"], r["group_id"], r["case_id"]): r
              for r in read_lines(output / "failure_questions.jsonl")}
    checked = []
    for note in notes:
        case = json.loads((output / "cases" / (note["case"] + ".json")).read_text())
        spec = selected[note["case"]]
        outcomes = scores[spec["task"], spec["group_id"], spec["case_id"]]
        quotes = []
        for quote in note.get("quotes", []):
            text = case["conditions"][quote["method"]]["input"]["retrieved"][quote["record"]]["text"]
            assert quote["text"] in text, (note["case"], quote)
            quotes.append(dict(quote, start=text.index(quote["text"])))
        context_checks = []
        for check in note.get("context_checks", []):
            records = case["conditions"][check["method"]]["input"]["retrieved"]
            present = any(check["text"] in r["text"] for r in records)
            assert present == check["present"], (note["case"], check)
            context_checks.append(dict(check, verified=True))
        if note.get("same_selected_sources"):
            methods = ("libra", "without_context_augmentation")
            centers = [[r["metadata"]["source_passage"] for r in case["conditions"][m]["input"]["retrieved"]
                        if "source_passage" in r.get("metadata", {})] for m in methods]
            assert centers[0] == centers[1] and len(centers[0]) == 5
        checked.append(dict(note, quotes=quotes, task=spec["task"], case_id=spec["case_id"],
                            context_checks=context_checks,
                            question=outcomes["question"], answers=outcomes["answers"],
                            answer_field_role=outcomes["answer_field_role"],
                            conditions=outcomes["conditions"], context_model=spec["model"]))
    assert set(selected) == {r["case"] for r in checked}
    write_json(output / "failure_review_evidence.json", checked)
    print(len(checked), "manually reviewed cases; exact quotes and native scores linked", flush=True)


def case_figures(output):
    import textwrap
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    annotations = json.loads((output / "case_annotations.json").read_text())
    provenance = []
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 5.7))
    fig.subplots_adjust(left=.015, right=.985, bottom=.035, top=.98, wspace=.09)
    for ax, note in zip(axes, annotations):
        case = json.loads((output / "cases" / (note["case"] + ".json")).read_text())
        amor = case["conditions"]["libra"]
        baseline = case["conditions"][note["baseline"]]
        without_augmentation = case["conditions"]["without_context_augmentation"]
        centers = lambda c: [r["metadata"]["original_source_text"] for r in c["input"]["retrieved"]
                             if "original_source_text" in r.get("metadata", {})]
        assert len(centers(amor)) == 5 and centers(amor) == centers(without_augmentation)
        question = amor["input"]["case"]["question"]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(.02, .98, note["title"], fontweight="bold", fontsize=10, va="top")
        ax.text(.02, .928, note["task"] + " | Qwen3.5-4B", fontsize=8, color="#525A65", va="top")
        ax.text(.02, .884, textwrap.fill(question, 48), fontsize=9, va="top", fontweight="bold")
        for condition, label, quotes, explanation, top, color in (
            (baseline, note["baseline_label"], note["baseline_quotes"], note["baseline_note"], .755, "#9C5952"),
            (amor, "AMOR", note["amor_quotes"], note["amor_note"], .405, "#426B99")):
            ax.add_patch(Rectangle((0, top-.32), 1, .32, facecolor="#F5F6F8", edgecolor="none"))
            ax.text(.025, top-.015, label, fontsize=9, fontweight="bold", color=color, va="top")
            y = top-.066
            for quote in quotes:
                matches = [(i, r["text"].index(quote)) for i, r in enumerate(condition["input"]["retrieved"])
                           if quote in r["text"]]
                assert matches, (note["case"], label, quote)
                provenance.append(dict(case=note["case"], condition=label, quote=quote,
                                       matches=matches, directory=condition["directory"]))
                wrapped = textwrap.fill('"' + quote + '"', 46)
                ax.text(.025, y, wrapped, fontsize=8, va="top")
                y -= .028 * (wrapped.count("\n") + 1) + .01
            ax.text(.025, y-.005, textwrap.fill(explanation, 47), fontsize=7.5, va="top", color="#505966")
            answer = condition["prediction"]["prediction"]
            metric, value = score(condition["input"]["case"], answer)
            ax.text(.025, top-.247, textwrap.fill("Answer: " + answer, 47), fontsize=8,
                    va="top", fontweight="bold", color=color)
            metric_label = "Substring EM" if metric == "substring_exact_match" else "F1"
            ax.text(.975, top-.307, f"{metric_label}: {value:.2f}", fontsize=7.5,
                    va="bottom", ha="right", color=color)
        ax.text(.02, .052, textwrap.fill(note["observation"], 45), fontsize=8, va="top", fontweight="bold")
    for extension in ("pdf", "png", "svg"):
        fig.savefig(output / "figures" / f"case_study.{extension}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    write_json(output / "quote_provenance.json", provenance)
    reviewed = {}
    for selection in ("review_selection.json", "consistent_review_selection.json"):
        for request in json.loads((output / selection).read_text()):
            if request["setting"] == "one_shot":
                reviewed[request["task"], request["case_id"]] = request["name"]
    scores = []
    for row in read_lines(output / "question_scores.jsonl"):
        key = row["task"], row["case_id"]
        if row["setting"] != "one_shot" or key not in reviewed:
            continue
        for method, values in row["conditions"].items():
            scores.append(dict(case=reviewed[key], model=row["model"], method=method,
                               score=values["score"],
                               supporting_passages=values.get("supporting_passages", ""),
                               gold_passages=values.get("gold_passages", "")))
    assert len(scores) == len(reviewed) * len(MODELS) * 11
    write_csv(output / "reviewed_case_scores.csv", scores)

    rows = csv_rows(output / "support_transitions.csv")
    per_model = []
    for model in MODELS:
        matrix = [[0, 0], [0, 0]]
        for row in rows:
            if (row["setting"], row["model"], row["comparator"]) == ("one_shot", model, "hipporag2"):
                matrix[row["baseline_all_support"] == "True"][row["amor_all_support"] == "True"] += int(row["questions"])
        per_model.append(matrix)
    assert all(matrix == per_model[0] for matrix in per_model) and sum(map(sum, matrix)) == 1000
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    colors = [["#E3E6EB", "#ADC3DE"], ["#E2B3AC", "#E3E6EB"]]
    for i in range(2):
        for j in range(2):
            ax.add_patch(Rectangle((j-.5, i-.5), 1, 1, facecolor=colors[i][j], edgecolor="white", linewidth=3))
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center", fontsize=17, fontweight="bold")
    ax.set_xlim(-.5, 1.5)
    ax.set_ylim(1.5, -.5)
    ax.set_xticks([0, 1], ["Incomplete", "Complete"])
    ax.set_yticks([0, 1], ["Incomplete", "Complete"])
    ax.set_xlabel("AMOR supporting passages", fontweight="bold")
    ax.set_ylabel("HippoRAG 2\nsupporting passages", fontweight="bold")
    ax.set_title("2Wiki: all annotated support in top 5", fontsize=10, fontweight="bold", pad=10)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(output / "figures" / f"support_coverage_2wiki.{extension}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/oscar/scratch/zliu328/agent-memory-outputs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--case-figures", action="store_true")
    parser.add_argument("--evidence-recovery", action="store_true")
    parser.add_argument("--results", type=Path, help="Use audited replacement AMOR results for evidence analysis")
    parser.add_argument("--failure-mining", action="store_true")
    parser.add_argument("--memory-failures", action="store_true")
    parser.add_argument("--context-controls", action="store_true")
    parser.add_argument("--context-components", action="store_true")
    parser.add_argument("--context-report", action="store_true")
    parser.add_argument("--memory-interventions", action="store_true")
    parser.add_argument("--intervention-report", action="store_true")
    parser.add_argument("--support-retention", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--inspect-stores", type=Path)
    parser.add_argument("--review", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.memory_interventions:
        memory_interventions(args.output, args.model)
    elif args.intervention_report:
        memory_intervention_report(args.results, args.output)
    elif args.context_report:
        if args.results is None:
            parser.error('--context-report requires --results')
        context_component_report(args.results, args.output)
    elif args.support_retention:
        if args.results is None:
            parser.error('--support-retention requires --results')
        support_retention(args.results, args.output)
    elif args.context_controls or args.context_components:
        if args.results is None:
            parser.error('--context-controls requires --results')
        context_controls(args.results, args.output, args.model, population=args.context_components)
    elif args.memory_failures:
        if args.results is None:
            parser.error('--memory-failures requires --results')
        memory_failure_analysis(args.root, args.output, args.results)
    elif args.inspect_stores:
        inspect_stores(args.inspect_stores, args.output)
    elif args.review:
        review_cases(args.review, args.output)
    elif args.failure_mining:
        failure_mining(args.root, args.output, args.results)
    elif args.evidence_recovery:
        evidence_recovery(args.root, args.output, args.results)
    elif args.case_figures:
        case_figures(args.output)
    elif args.selection:
        export_cases(args.output, args.selection)
    else:
        if not args.plot_only:
            analyze(args.root, args.output)
        plot(args.output)


def evidence_recovery(root, output, results=None):
    """Count annotated passage recovery at the existing top-five cutoff."""
    from dataset_loader.loader import load_hipporag2_dataset
    from utils.hipporag_metrics import gold_passage_recall_at_k

    dataset = Path(__file__).resolve().parents[1] / "baseline_algorithms/HippoRAG/reproduce/dataset"
    official = load_hipporag2_dataset(dataset, "2wikimultihopqa")
    references = {q.query_id: q for q in official}
    assert len(references) == len(official) == COUNTS["2WikiMultiHopQA"]
    entries = manifest(root)
    if results is not None:
        audit = json.loads((results / "audit.json").read_text())
        assert audit["complete"] and not audit["missing"]
        for row in csv_rows(results / "qa_results.csv"):
            if row["setting"] == "one_shot":
                method = "libra" if row["variant"] == "full" else row["variant"]
                entries["one_shot", row["model"].replace("/", "_"), row["task"], method] = dict(
                    row, directory=row["source"])
    methods = ("bm25", "dense", "hipporag2", "catrag", "without_propagation", "libra")
    selected, provenance = {}, []
    for method in methods:
        for model in MODELS:
            entry = entries["one_shot", model, "2WikiMultiHopQA", method]
            path = Path(entry["directory"]) / "retrieval.jsonl"
            selections = {}
            for row in read_lines(path):
                case = row["case"]
                key = case["case_id"]
                assert key not in selections
                assert case["question"] == references[key].question
                assert set(case["gold_passages"]) == set(references[key].gold_passages)
                if method in ("libra", "without_propagation"):
                    texts = [r["metadata"]["original_source_text"] for r in row["retrieved"]
                             if "source_passage" in r.get("metadata", {})]
                else:
                    texts = [r["text"] for r in row["retrieved"]]
                assert len(texts) == len(set(texts)) == 5, (method, key)
                assert set(texts).issubset(official[0].paragraphs)
                selections[key] = texts
            assert set(selections) == set(references)
            if method in selected:
                assert selected[method] == selections, (method, model, "reader selections differ")
            else:
                selected[method] = selections
            provenance.append(dict(method=method, model=model, source=str(path)))
            print("Verified", method, model, len(selections), flush=True)
    gold = {(key, text) for key, q in references.items() for text in q.gold_passages}
    found = {method: {(key, text) for key, texts in by_question.items()
                     for text in texts if text in references[key].gold_passages}
             for method, by_question in selected.items()}
    complete = {method: {key for key, texts in by_question.items()
                         if set(references[key].gold_passages).issubset(texts)}
                for method, by_question in selected.items()}
    any_support = {method: {key for key, _ in occurrences} for method, occurrences in found.items()}
    dense_missing = gold - found["dense"]
    rows, comparisons = [], []
    for method in methods:
        rows.append(dict(method=method, questions=len(references), gold_occurrences=len(gold),
            found=len(found[method]),
            questions_any_support=len(any_support[method]),
            questions_partial_support=len(any_support[method] - complete[method]),
            questions_no_support=len(references) - len(any_support[method]),
            recall_at_5=100 * sum(gold_passage_recall_at_k(references[k].gold_passages, texts, 5)
                                 for k, texts in selected[method].items()) / len(references),
            all_support_at_5=100 * len(complete[method]) / len(references),
            recovered_from_dense=len(found[method] & dense_missing),
            lost_from_dense=len(found["dense"] - found[method])))
        if method == "libra":
            continue
        gain, loss = found["libra"] - found[method], found[method] - found["libra"]
        comparisons.append(dict(comparator=method, missing=len(gold-found[method]),
            recovered=len(gain), lost=len(loss),
            questions_partial_to_complete=len((any_support[method] - complete[method]) & complete['libra']),
            questions_none_to_complete=len(complete['libra'] - any_support[method]),
            questions_only_amor_complete=len(complete["libra"] - complete[method]),
            questions_only_comparator_complete=len(complete[method] - complete["libra"]),
            questions_both_complete=len(complete[method] & complete["libra"]),
            questions_neither_complete=len(references)-len(complete[method] | complete["libra"])))
        assert len(found["libra"]) - len(found[method]) == len(gain) - len(loss)
    write_csv(output / "evidence_recovery.csv", rows)
    write_csv(output / "evidence_recovery_pairs.csv", comparisons)
    with (output / "evidence_recovery_questions.jsonl").open("w") as stream:
        for key, q in references.items():
            stream.write(json.dumps(dict(case_id=key, gold_passages=q.gold_passages,
                selected={m: data[key] for m, data in selected.items()}), ensure_ascii=False)+"\n")
    write_json(output / "evidence_recovery_audit.json", dict(complete=True,
        questions=len(references), corpus_passages=len(official[0].paragraphs),
        gold_occurrences=len(gold), gold_passages_per_question=dict(Counter(len(q.gold_passages) for q in official)),
        dense_missing=len(dense_missing), dense_found=len(found["dense"]),
        gold_source="Released supporting_facts titles mapped by the unchanged dataset loader",
        all_four_reader_selections_equal=True, reader_count_not_multiplied=True,
        augmentation_excluded=True, provenance=provenance))
    print(json.dumps(dict(results=rows, paired=comparisons), indent=2), flush=True)
    if results is not None:
        recommendation_analysis(root, output, entries, references, selected)


def recommendation_analysis(root, output, entries, references, selected):
    """Relate native support coverage to frozen similarities and answer F1."""
    import numpy as np
    import pandas as pd
    from experiments.analyze_geometry import GRAPH, QUERIES, ranks

    task = "2WikiMultiHopQA"
    original = list(read_lines(Path(entries["one_shot", MODELS[0], task, "libra"]["directory"]) / "retrieval.jsonl"))
    groups = {r["group_id"] for r in original}
    assert len(groups) == 1
    group = groups.pop()
    metadata = json.loads((GRAPH / task / "memory" / group / "graph.json").read_text())
    corpus = pd.read_parquet(Path(metadata["source_graph"]).parent / "chunk_embeddings/vdb_chunk.parquet")
    texts = corpus.content.tolist()
    positions = {text: i for i, text in enumerate(texts)}
    assert len(positions) == len(texts)
    vectors = np.stack(corpus.embedding).astype(np.float32)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    with np.load(QUERIES / task / group / "queries.npz", allow_pickle=False) as cache:
        questions = cache["queries"].tolist()
        query_vectors = cache["passage"].copy().reshape(len(questions), -1)
    query_positions = {q: i for i, q in enumerate(questions)}
    occurrences = []
    matching_dense_selections = 0
    for key, question in references.items():
        query = query_vectors[query_positions[question.question]]
        np.testing.assert_allclose(np.linalg.norm(query), 1, atol=1e-5)
        similarity = vectors @ query
        ranking = ranks(similarity)
        matching_dense_selections += set(texts[i] for i in np.argsort(ranking)[:5]) == set(selected["dense"][key])
        for text in dict.fromkeys(question.gold_passages):
            i = positions[text]
            occurrences.append(dict(case_id=key, source=corpus.iloc[i].hash_id,
                direct_similarity_rank=int(ranking[i]), cosine=float(similarity[i]),
                **{method: text in by_question[key] for method, by_question in selected.items()}))
    write_csv(output / "support_similarity.csv", occurrences)
    summaries = []
    for recovered in (True, False):
        subset = [r for r in occurrences if not r["dense"] and r["libra"] == recovered]
        summaries.append(dict(recovered_by_amor=recovered, occurrences=len(subset),
            median_direct_similarity_rank=float(np.median([r["direct_similarity_rank"] for r in subset])),
            median_cosine=float(np.median([r["cosine"] for r in subset]))))
    write_csv(output / "support_similarity_summary.csv", summaries)
    cases = {case_key(r): r["case"] for r in original}
    outcomes, transitions = [], []
    methods = list(selected) + ["without_context_augmentation"]
    for model in MODELS:
        predictions = {method: audit_predictions(entries["one_shot", model, task, method],
                        cases, task, method) for method in methods}
        for method, values in predictions.items():
            selections = selected["libra" if method == "without_context_augmentation" else method]
            for coverage in (False, True):
                subset = [r for (_, key), r in values.items()
                          if set(references[key].gold_passages).issubset(selections[key]) == coverage]
                outcomes.append(dict(model=model, method=method, all_support=coverage,
                    questions=len(subset), answer_f1=100 * sum(r["score"] for r in subset) / len(subset)))
        for comparator in ("dense", "hipporag2", "catrag", "without_propagation"):
            for before in (False, True):
                for after in (False, True):
                    keys = [k for k in cases if
                        set(references[k[1]].gold_passages).issubset(selected[comparator][k[1]]) == before and
                        set(references[k[1]].gold_passages).issubset(selected["libra"][k[1]]) == after]
                    for amor_variant in ("libra", "without_context_augmentation"):
                        deltas = [predictions[amor_variant][k]["score"] - predictions[comparator][k]["score"] for k in keys]
                        transitions.append(dict(model=model, comparator=comparator, amor_variant=amor_variant,
                            baseline_complete=before, amor_complete=after, questions=len(keys),
                            higher=sum(d > 0 for d in deltas), equal=sum(d == 0 for d in deltas),
                            lower=sum(d < 0 for d in deltas),
                            mean_f1_difference=100 * sum(deltas) / len(deltas) if deltas else None))
    write_csv(output / "answer_support.csv", outcomes)
    write_csv(output / "answer_support_transitions.csv", transitions)
    write_json(output / "recommendation_analysis_audit.json", dict(complete=True, questions=len(references),
        supporting_occurrences=len(occurrences), native_dense_top_five_matches=matching_dense_selections,
        similarity_source="Frozen AMOR query and passage embeddings; not native Dense baseline ranks",
        original_normalized_embeddings=True, scoring="Native answer F1 recomputed for each prediction",
        scope="Descriptive evidence selection and QA comparisons; no causal error attribution"))
    print(json.dumps(dict(similarity=summaries, answer_support=outcomes), indent=2), flush=True)
    reviewed = root / "paper_costs_20260925/analysis"
    examples = json.loads((reviewed / "context_tsne_audit.json").read_text())
    conversation = json.loads((reviewed / "conversation_context_audit.json").read_text())
    examples.append(dict(task="LoCoMo", case_id=conversation["case_id"]))
    case_rows = []
    for example in examples:
        task, cid = example["task"], example["case_id"]
        for model in MODELS:
            for method in METHODS + ["without_propagation", "without_context_augmentation"]:
                directory = Path(entries["one_shot", model, task, method]["directory"])
                context = next(r for r in read_lines(directory / "retrieval.jsonl") if r["case"]["case_id"] == cid)
                path, = directory.glob("evaluations/*/predictions.jsonl")
                prediction = next(r for r in read_lines(path) if r["case_id"] == cid)
                metric, value = score(context["case"], prediction["prediction"])
                assert math.isclose(value, prediction["metrics"][metric], abs_tol=1e-12)
                case_rows.append(dict(task=task, case_id=cid, model=model, method=method,
                    context=context, prediction=prediction["prediction"], score=value, source=str(directory)))
    write_json(output / "reviewed_cases.json", case_rows)


def memory_failure_analysis(root, output, results):
    """Inspect native LoCoMo categories, evidence IDs, and paired saved answers."""
    from dataset_loader.loader import load_locomo
    from experiments.analyze_geometry import GRAPH

    data_path = Path('/oscar/scratch/zliu328/agent-memory-data/locomo/locomo10.json')
    raw = json.loads(data_path.read_text())
    conversations = {c.sample_id: c for c in load_locomo(data_path)}
    entries = manifest(root)
    for row in csv_rows(results / 'qa_results.csv'):
        if row['setting'] == 'one_shot':
            method = 'libra' if row['variant'] == 'full' else row['variant']
            entries['one_shot', row['model'].replace('/', '_'), row['task'], method] = dict(row, directory=row['source'])
    references, source_maps, unresolved = {}, {}, []
    for sample in raw:
        conv = conversations[sample['sample_id']]
        group = 'locomo-' + conv.sample_id
        sessions = sorted((k for k in sample['conversation'] if k.startswith('session_') and k[8:].isdigit()),
                          key=lambda k: int(k[8:]))
        turns = [t for k in sessions for t in sample['conversation'][k]]
        assert len(turns) == len(conv.memory_items)
        ids = [t['dia_id'] for t in turns]
        assert len(set(ids)) == len(ids)
        by_text = {text: did for text, did in zip(conv.memory_items, ids)}
        assert len(by_text) == len(ids)
        meta = json.loads((GRAPH / 'LoCoMo/memory' / group / 'graph.json').read_text())
        contents = json.loads(Path(meta['compiled_source_file']).read_text())
        by_node = {}
        for node, item in contents.items():
            pos = item['source_position']
            assert item['original_source_text'] == conv.memory_items[pos]
            by_node[node] = ids[pos]
        source_maps[group] = (by_text, by_node)
        for i, qa in enumerate(sample['qa']):
            key = (group, f'{conv.sample_id}-{i}')
            evidence = list(qa.get('evidence', []))
            bad = [e for e in evidence if e not in ids]
            if bad:
                unresolved.append(dict(group_id=group, case_id=key[1], evidence=evidence, unresolved=bad))
            references[key] = dict(question=qa['question'], category=qa['category'],
                                   evidence=evidence, unresolved=bad,
                                   answer_role='distractor' if qa['category'] == 5 else 'reference',
                                   answer=str(qa.get('answer', qa.get('adversarial_answer', ''))))
    assert len(references) == COUNTS['LoCoMo']
    methods = METHODS + ['without_propagation', 'without_context_augmentation']
    contexts, predictions, coverage, summaries, provenance = {}, {}, {}, [], []
    original = {'bm25', 'dense', 'hipporag2', 'catrag'}
    amor = {'libra', 'without_propagation', 'without_context_augmentation'}
    for method in methods:
        directory = Path(entries['one_shot', MODELS[0], 'LoCoMo', method]['directory'])
        contexts[method] = {case_key(r): r for r in read_lines(directory / 'retrieval.jsonl')}
        assert set(contexts[method]) == set(references)
        for key, row in contexts[method].items():
            ref = references[key]
            assert row['case']['question'] == ref['question'] and row['case']['category'] == ref['category']
            if method not in original | amor:
                continue
            by_text, by_node = source_maps[key[0]]
            centers, expanded = [], set()
            for item in row['retrieved']:
                if method in original:
                    centers.append(by_text[item['text']])
                else:
                    md = item['metadata']
                    if 'source_passage' not in md:
                        assert md['context_representation'] == 'retrieved_triples'
                        continue
                    did = by_node[md['source_passage']]
                    assert by_text[md['original_source_text']] == did
                    centers.append(did)
                    if method != 'without_context_augmentation':
                        for node in md.get('window_sources', []):
                            expanded.add(by_node[node])
            expanded.update(centers)
            evidence = ref['evidence']
            coverage[method, key] = dict(selected=centers, expanded=sorted(expanded),
                selected_hits=[e for e in evidence if e in centers],
                expanded_hits=[e for e in evidence if e in expanded])
        cases = {k:r['case'] for k,r in contexts[method].items()}
        for model in MODELS:
            entry = entries['one_shot', model, 'LoCoMo', method]
            values = audit_predictions(entry, cases, 'LoCoMo', method)
            predictions[model, method] = values
            for category in range(1, 6):
                subset = [v for k,v in values.items() if references[k]['category'] == category]
                summaries.append(dict(model=model, method=method, category=category,
                    questions=len(subset), score=100*sum(v['score'] for v in subset)/len(subset)))
            provenance.append(dict(model=model, method=method, source=entry['directory']))
        print('Audited LoCoMo', method, flush=True)
    for key in references:
        assert coverage['libra', key]['selected'] == coverage['without_context_augmentation', key]['selected']
    write_csv(output / 'locomo_failure_categories.csv', summaries)
    coverage_rows = []
    for method in original | amor:
        for category in range(1, 6):
            keys = [k for k,r in references.items() if r['category'] == category and r['evidence']]
            coverage_rows.append(dict(method=method, category=category, questions=len(keys),
                unresolved_questions=sum(bool(references[k]['unresolved']) for k in keys),
                selected_recall=100*sum(len(coverage[method,k]['selected_hits'])/len(references[k]['evidence']) for k in keys)/len(keys),
                expanded_recall=100*sum(len(coverage[method,k]['expanded_hits'])/len(references[k]['evidence']) for k in keys)/len(keys)))
    write_csv(output / 'locomo_turn_coverage.csv', coverage_rows)
    rows = []
    for key, ref in references.items():
        conditions = {model: {method: predictions[model,method][key] for method in methods} for model in MODELS}
        rows.append(dict(group_id=key[0], case_id=key[1], **ref, conditions=conditions,
                         coverage={m:coverage[m,key] for m in methods if (m,key) in coverage}))
    with (output / 'locomo_failure_questions.jsonl').open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    candidates = []
    for row in rows:
        for comparator in ('without_propagation', 'without_context_augmentation'):
            deltas = [v['libra']['score'] - v[comparator]['score'] for v in row['conditions'].values()]
            if all(d > 0 for d in deltas) or all(d < 0 for d in deltas):
                candidates.append(dict(case_id=row['case_id'], question=row['question'], answer=row['answer'],
                    category=row['category'], comparator=comparator,
                    direction='higher' if deltas[0] > 0 else 'lower',
                    conditions={m:{k:v[k] for k in ('libra', comparator)} for m,v in row['conditions'].items()}))
    write_json(output / 'locomo_failure_candidates.json', candidates)
    selected_examples = ('conv-42-141', 'conv-26-172')
    examples = []
    for row in rows:
        if row['case_id'] not in selected_examples:
            continue
        key = (row['group_id'], row['case_id'])
        sample, = [s for s in raw if 'locomo-' + s['sample_id'] == row['group_id']]
        sessions = {e.split(':')[0][1:] for e in row['evidence']}
        examples.append(dict(row, contexts={m:contexts[m][key] for m in methods},
                             original_sessions={s:sample['conversation']['session_'+s] for s in sessions}))
    assert len(examples) == len(selected_examples)
    write_json(output / 'locomo_mechanism_examples.json', examples)
    write_json(output / 'locomo_failure_audit.json', dict(complete=True, questions=len(references),
        conditions=len(provenance), native_scores_recomputed=True, source=str(data_path),
        unresolved_evidence=unresolved, provenance=provenance,
        note='Native category partitions and literal released evidence IDs; no annotation repair or semantic source mapping. Evidence absent from source IDs remains unmatched. Category 5 evidence is not an answerability label. Expanded coverage counts original neighboring turns, not appended facts.'))
    wiki_question_types(output)


def support_retention(results, output):
    """Trace literal annotated turns through extraction and fact selection."""
    import igraph as ig
    from dataset_loader.loader import load_locomo
    from experiments.analyze_geometry import GRAPH

    data_path = Path('/oscar/scratch/zliu328/agent-memory-data/locomo/locomo10.json')
    raw = json.loads(data_path.read_text())
    conversations = {c.sample_id:c for c in load_locomo(data_path)}
    protocol = json.loads((results / 'protocol.json').read_text())
    questions = {(r['group_id'],r['case_id']):r for r in read_lines(output/'locomo_failure_questions.jsonl')}
    rows, unresolved = [], []
    for sample in raw:
        conv = conversations[sample['sample_id']]
        group = 'locomo-' + conv.sample_id
        sessions = sorted((k for k in sample['conversation'] if k.startswith('session_') and k[8:].isdigit()), key=lambda k:int(k[8:]))
        turns = [t for k in sessions for t in sample['conversation'][k]]
        assert len(turns) == len(conv.memory_items)
        meta = json.loads((GRAPH/'LoCoMo/memory'/group/'graph.json').read_text())
        contents = json.loads(Path(meta['compiled_source_file']).read_text())
        openie, = Path(meta['source_graph']).parent.parent.glob('openie_results_ner_*.json')
        docs = {d['idx']:d for d in json.loads(openie.read_text())['docs']}
        graph_path = Path(protocol['graph_source'])/'groups/LoCoMo'/group/(protocol['graph_variant']+'.pickle')
        graph = ig.Graph.Read_Pickle(str(graph_path))
        degree = dict(zip(graph.vs['name'],graph.degree()))
        stats = {}
        for key,content in contents.items():
            pos = content['source_position']
            assert content['original_source_text'] == conv.memory_items[pos]
            stats[turns[pos]['dia_id']] = dict(source=key,extracted=len(docs[key]['extracted_triples']),
                retained=len(content['retained_triples']),degree=degree[key])
        for i,qa in enumerate(sample['qa']):
            cid = f'{conv.sample_id}-{i}'
            ref = questions[group,cid]
            assert qa['question'] == ref['question']
            for did in dict.fromkeys(qa.get('evidence',[])):
                if did not in stats:
                    unresolved.append(dict(case_id=cid,evidence=did))
                    continue
                value = stats[did]
                rows.append(dict(case_id=cid,group=group,category=qa['category'],evidence=did,**value,
                    selected=did in ref['coverage']['libra']['selected'],
                    expanded=did in ref['coverage']['libra']['expanded']))
    write_csv(output/'locomo_support_retention.csv',rows)
    summary = []
    for category in range(1,6):
        subset = [r for r in rows if r['category']==category]
        selected_out = [r for r in subset if r['extracted']>0 and r['retained']==0]
        summary.append(dict(category=category,annotated_turn_occurrences=len(subset),
            questions=len({r['case_id'] for r in subset}),no_extracted_facts=sum(r['extracted']==0 for r in subset),
            all_extracted_facts_removed=len(selected_out),
            removed_and_isolated=sum(r['degree']==0 for r in selected_out),
            removed_but_selected=sum(r['selected'] for r in selected_out),
            removed_but_expanded=sum(r['expanded'] for r in selected_out)))
    write_csv(output/'locomo_support_retention_summary.csv',summary)
    write_json(output/'locomo_support_retention_audit.json',dict(complete=True,questions=len(questions),
        annotations=len(rows),unresolved=unresolved,new_inference=False,
        note='Literal annotated turn occurrences; an annotated turn is not itself a gold fact. Category 5 marks distractor-related evidence, not answerable support. No semantic fact labeling.'))


def memory_interventions(output, model=None):
    """Evaluate frozen diagnostic contexts with native prompts and scoring."""
    import random
    from experiments.runner import _read_retrieval_records, _answer_prompt, _official_generation, _score
    from optimization.ircot import Reader, MODELS as READERS, BATCH_SIZE, SEED

    target = output / 'memory_interventions'
    protocol = json.loads((target / 'prepared.json').read_text())
    assert protocol['complete'] and protocol['new_llm_calls'] == 0
    prepared = []
    for entry in protocol['entries']:
        rows = list(_read_retrieval_records(Path(entry['path'])))
        assert len(rows) == entry['questions']
        rng = random.Random(SEED)
        for row in rows:
            messages, answer_key = _answer_prompt(row.case, row.retrieved, rng)
            assert answer_key is None
            request = dict(phase='answer', messages=messages, generation=_official_generation(row.case))
            prepared.append((entry, row, request))
    assert len(prepared) == 207
    models = list(READERS) if model == 'all' else [model]
    assert all(m in READERS for m in models)
    for model in models:
        directory = target / model.replace('/', '_')
        directory.mkdir(exist_ok=True)
        path = directory / 'predictions.jsonl'
        completed = {(r['task'], r['condition'], r['case_id']): r for r in read_lines(path)} if path.exists() else {}
        pending = [p for p in prepared if (p[0]['task'], p[0]['condition'], p[1].case.case_id) not in completed]
        if pending:
            reader = Reader(model)
            try:
                write_json(directory / 'reader.json', reader.metadata)
                # Run the first request of each condition before its remaining batches.
                batches = []
                for entry in protocol['entries']:
                    group = [p for p in pending if p[0] == entry]
                    if group:
                        batches.append(group[:1])
                        batches.extend(group[i:i+BATCH_SIZE] for i in range(1, len(group), BATCH_SIZE))
                for batch in batches:
                    requests = [p[2] for p in batch]
                    assert not reader.request(phase='check_context', items=requests)['overflow']
                    generated = reader.request(phase='generate', items=requests)
                    for (entry, row, request), response in zip(batch, generated['responses'], strict=True):
                        result = dict(task=entry['task'], condition=entry['condition'], scope=entry['scope'],
                            case_id=row.case.case_id, prediction=response['answer'],
                            metrics=_score(row.case, response['answer'], row.retrieved, row.top_k),
                            request=request, usage=response['usage'], generation_settings=response['generation_settings'])
                        with path.open('a') as stream:
                            stream.write(json.dumps(result, ensure_ascii=False)+'\n')
                        completed[entry['task'], entry['condition'], row.case.case_id] = result
                    print(model, 'memory interventions', len(completed), '/', len(prepared), flush=True)
            finally:
                reader.close()
        assert len(completed) == len(prepared)
        for entry, row, request in prepared:
            result = completed[entry['task'], entry['condition'], row.case.case_id]
            assert result['request'] == request
            assert result['metrics'] == _score(row.case, result['prediction'], row.retrieved, row.top_k)
        write_json(directory / 'complete.json', dict(complete=True, requests=len(prepared),
            metrics_recomputed=True, model=model, revision=READERS[model], seed=SEED))


def memory_intervention_report(results, output):
    """Report paired population outcomes separately from selected local cases."""
    from optimization.ircot import MODELS as READERS
    target = output / 'memory_interventions'
    protocol = json.loads((target / 'prepared.json').read_text())
    entries = csv_rows(results / 'qa_results.csv')
    summary, local = [], []
    for model in READERS:
        directory = target / model.replace('/', '_')
        assert json.loads((directory / 'complete.json').read_text())['complete']
        predictions = list(read_lines(directory / 'predictions.jsonl'))
        for intervention in protocol['entries']:
            task, condition = intervention['task'], intervention['condition']
            rows = [r for r in predictions if r['task'] == task and r['condition'] == condition]
            assert len(rows) == intervention['questions']
            if intervention['scope'] == 'local_case':
                local.extend(dict(model=model, task=task, condition=condition, case_id=r['case_id'],
                    prediction=r['prediction'], score=r['metrics']['substring_exact_match']) for r in rows)
                continue
            entry, = [r for r in entries if r['setting'] == 'one_shot' and r['task'] == task
                      and r['variant'] == 'full' and r['model'] == model]
            path, = Path(entry['source']).glob('evaluations/*/predictions.jsonl')
            current = {r['case_id']: r for r in read_lines(path)}
            assert set(current) == {r['case_id'] for r in rows} and len(rows) == 100
            original_contexts = {r['case']['case_id']: [i['text'] for i in r['retrieved']]
                                 for r in read_lines(Path(entry['source']) / 'retrieval.jsonl')}
            control_contexts = {r['case']['case_id']: [i['text'] for i in r['retrieved']]
                                for r in read_lines(Path(intervention['path']))}
            unchanged = {cid for cid in original_contexts if original_contexts[cid] == control_contexts[cid]}
            differences = [current[r['case_id']]['metrics']['substring_exact_match']-
                           r['metrics']['substring_exact_match'] for r in rows]
            summary.append(dict(model=model, task=task, control=condition, questions=len(rows),
                current=sum(r['metrics']['substring_exact_match'] for r in current.values()),
                control_score=sum(r['metrics']['substring_exact_match'] for r in rows),
                current_higher=sum(d>0 for d in differences), equal=sum(d==0 for d in differences),
                current_lower=sum(d<0 for d in differences),
                unchanged_contexts=len(unchanged),
                unchanged_score_disagreements=sum(d!=0 and r['case_id'] in unchanged for r,d in zip(rows,differences)),
                mean_input_tokens=sum(r['usage']['input_tokens'] for r in rows)/len(rows),
                mean_output_tokens=sum(r['usage']['output_tokens'] for r in rows)/len(rows)))
    write_csv(target / 'population_results.csv', summary)
    write_csv(target / 'local_results.csv', local)
    print(json.dumps(dict(population=summary, local_cases=local), indent=2), flush=True)


def context_controls(results, output, model=None, population=False):
    """Replay reviewed memory failures with explicit source/fact interventions."""
    from dataclasses import replace
    import random
    from types import SimpleNamespace
    from baseline.base import RetrievedItem
    from dataset_loader.loader import load_locomo
    from experiments.runner import (_read_retrieval_records, _answer_prompt, _official_generation,
                                    _score, _locomo_category_5_answer)
    from experiments.ablate_memory import original_text_only
    from optimization.retriever.query_fact_context import QueryFactContext
    from optimization.ircot import GRAPH, Reader, MODELS as READERS, BATCH_SIZE, SEED

    target = output / ('context_components' if population else 'context_controls')
    target.mkdir(exist_ok=True)
    entry, = [r for r in csv_rows(results / 'qa_results.csv') if r['setting'] == 'one_shot'
              and r['task'] == 'LoCoMo' and r['variant'] == 'full' and r['model'] == 'Qwen/Qwen3.5-4B']
    rows = list(_read_retrieval_records(Path(entry['source']) / 'retrieval.jsonl'))
    assert len(rows) == 1986
    case_ids = ({row.case.case_id for row in rows} if population
                else {'conv-42-141', 'conv-26-172'})
    randomizer = random.Random(SEED)
    prepared, audit = [], []
    compiled = {}
    for row in rows:
        state = randomizer.getstate()
        original_messages, original_answer_key = _answer_prompt(row.case, row.retrieved, randomizer)
        if row.case.case_id not in case_ids:
            continue
        if row.group_id not in compiled:
            meta = json.loads((GRAPH / 'LoCoMo/memory' / row.group_id / 'graph.json').read_text())
            compiled[row.group_id] = meta, json.loads(Path(meta['compiled_source_file']).read_text())
        meta, contents = compiled[row.group_id]
        centers = [r for r in row.retrieved if 'source_passage' in r.metadata]
        facts = [r for r in row.retrieved if r.metadata.get('context_representation') == 'retrieved_triples']
        assert len(centers) == 5 and len(centers) + len(facts) == len(row.retrieved)
        center_keys = {r.metadata['source_passage'] for r in centers}
        all_windows = set(k for r in centers for k in r.metadata.get('window_sources', []))
        neighbors = list(dict.fromkeys(k for r in centers for k in r.metadata.get('window_sources', []) if k not in center_keys))

        def render(window, include_facts):
            changed, sources = dict(contents), []
            for item in centers:
                key = item.metadata['source_passage']
                keys = [k for k in contents[key].get('window_sources', []) if k in window]
                changed[key] = dict(contents[key], window_sources=keys)
                updated = replace(item, metadata=dict(item.metadata, window_sources=keys))
                sources.append(QueryFactContext._source(SimpleNamespace(contents=changed), updated, key))
            return tuple(sources + (facts if include_facts else []))

        conditions = [('full', render(all_windows, True), None),
                      ('centers_only', render(set(), False), None),
                      ('neighbors_only', render(all_windows, False), None),
                      ('facts_only', render(set(), True), None)]
        assert [r.text for r in conditions[0][1]] == [r.text for r in row.retrieved]
        assert [r.text for r in conditions[1][1]] == [r.text for r in original_text_only(row).retrieved]
        if population:
            conditions = conditions[2:]
        else:
            conditions.extend(('remove_neighbor:' + k, render(all_windows - {k}, True), k) for k in neighbors)
        restored = None
        if not population and row.case.case_id == 'conv-42-141':
            data_path = Path('/oscar/scratch/zliu328/agent-memory-data/locomo/locomo10.json')
            sample, = [s for s in json.loads(data_path.read_text()) if s['sample_id'] == 'conv-42']
            sessions = sorted((k for k in sample['conversation'] if k.startswith('session_') and k[8:].isdigit()), key=lambda k: int(k[8:]))
            turns = [t for k in sessions for t in sample['conversation'][k]]
            pos, = [i for i,t in enumerate(turns) if t['dia_id'] == 'D17:1']
            key, = [k for k,c in contents.items() if c['source_position'] == pos]
            native, = [c for c in load_locomo(data_path) if c.sample_id == 'conv-42']
            assert contents[key]['original_source_text'] == native.memory_items[pos]
            openie, = Path(meta['source_graph']).parent.parent.glob('openie_results_ner_*.json')
            doc, = [d for d in json.loads(openie.read_text())['docs'] if d['idx'] == key]
            triple = ['Nate', 'won', 'fourth video game tournament']
            assert triple in doc['extracted_triples'] and triple not in contents[key]['retained_triples']
            restored = dict(source=key, original_triple=triple, original_openie=str(openie))
            conditions.append(('restore_extracted_fact', render(set(), False) +
                               (RetrievedItem('(' + ', '.join(triple) + ')', None, {'source_passages': [key]}),), None))
            conditions.append(('restore_original_turn', render({key}, False), None))
        for name, items, removed in conditions:
            if name == 'full':
                assert [r.text for r in items] == [r.text for r in row.retrieved]
            if name == 'centers_only':
                assert [r.text for r in items] == [r.text for r in original_text_only(row).retrieved]
            rng = random.Random()
            rng.setstate(state)
            messages, answer_key = _answer_prompt(row.case, items, rng)
            assert answer_key == original_answer_key
            if name == 'full':
                assert messages == original_messages
            request = dict(phase='answer', messages=messages, generation=_official_generation(row.case))
            prepared.append((row, name, removed, items, request, answer_key))
        audit.append(dict(case_id=row.case.case_id, question=row.case.question, category=row.case.category,
            conditions=len(conditions), neighbor_texts={} if population else {k:contents[k]['original_source_text'] for k in neighbors},
            centers=[r.metadata['source_passage'] for r in centers], restoration=restored))
    assert {r['case_id'] for r in audit} == case_ids
    protocol = dict(cases=audit, conditions=len(prepared), seed=SEED, native_prompts_and_options=True,
        selection='All 1986 LoCoMo questions' if population else 'Two previously reviewed mechanism cases; no population estimate',
        intervention='Neighbors only and facts only, preserving the five original selections and frozen fact choices' if population else '2x2 neighbor/fact removal, removal of each extra neighboring turn, and diagnostic restoration of an extracted historical fact or original turn',
        note='Restoration uses inspected source evidence, not a deployable selection rule; no method tuning')
    if model is None:
        write_json(target / 'protocol.json', protocol)
        print('Prepared context interventions', len(prepared), flush=True)
        return
    assert model in READERS and json.loads((target / 'protocol.json').read_text()) == protocol
    directory = target / model.replace('/', '_')
    directory.mkdir(exist_ok=True)
    path = directory / 'predictions.jsonl'
    completed = {(r['case_id'],r['condition']):r for r in read_lines(path)} if path.exists() else {}
    pending = [p for p in prepared if (p[0].case.case_id,p[1]) not in completed]
    reader = Reader(model)
    try:
        write_json(directory / 'reader.json', reader.metadata)
        for start in range(0,len(pending),BATCH_SIZE):
            batch = pending[start:start+BATCH_SIZE]
            requests = [p[4] for p in batch]
            assert not reader.request(phase='check_context',items=requests)['overflow']
            generated = reader.request(phase='generate',items=requests)
            for (row,name,removed,items,request,answer_key), response in zip(batch,generated['responses'],strict=True):
                answer = response['answer']
                if answer_key is not None:
                    answer = _locomo_category_5_answer(answer,answer_key)
                result = dict(case_id=row.case.case_id,condition=name,removed_neighbor=removed,
                    raw_answer=response['answer'],prediction=answer,metrics=_score(row.case,answer,items,row.top_k),
                    request=request,usage=response['usage'])
                with path.open('a') as stream:
                    stream.write(json.dumps(result,ensure_ascii=False)+'\n')
                completed[row.case.case_id,name] = result
            print(model, 'context interventions',len(completed),'/',len(prepared),flush=True)
    finally:
        reader.close()
    assert len(completed) == len(prepared)
    for row,name,_,items,request,_ in prepared:
        result = completed[row.case.case_id,name]
        assert result['request'] == request and result['metrics'] == _score(row.case,result['prediction'],items,row.top_k)
    write_json(directory/'complete.json',dict(complete=True,conditions=len(prepared),cases=len(case_ids),
        metrics_recomputed=True,model=model,revision=READERS[model],seed=SEED))


def context_component_report(results, output):
    """Compare the four context conditions using the unchanged LoCoMo evaluator."""
    from optimization.ircot import MODELS as READERS

    entries = csv_rows(results / 'qa_results.csv')
    summary, pairs, sources = [], [], []
    target = output / 'context_components'
    for model in READERS:
        directory = target / model.replace('/', '_')
        complete = json.loads((directory / 'complete.json').read_text())
        assert complete['complete'] and complete['conditions'] == 3972
        conditions, cases = {}, None
        for variant, name in (('full', 'full'), ('without_context_augmentation', 'centers_only')):
            entry, = [r for r in entries if r['setting'] == 'one_shot' and r['task'] == 'LoCoMo'
                      and r['variant'] == variant and r['model'] == model]
            source = Path(entry['source'])
            contexts = list(read_lines(source / 'retrieval.jsonl'))
            native_cases = {r['case']['case_id']: r['case'] for r in contexts}
            assert len(native_cases) == len(contexts) == 1986
            if cases is None:
                cases = native_cases
            else:
                assert cases == native_cases
            path, = source.glob('evaluations/*/predictions.jsonl')
            rows = list(read_lines(path))
            assert len(rows) == len({r['case_id'] for r in rows}) == 1986
            values = {}
            for row in rows:
                metric, value = score(cases[row['case_id']], row['prediction'])
                assert math.isclose(value, row['metrics'][metric], abs_tol=1e-12)
                values[row['case_id']] = value
            assert set(values) == set(cases)
            conditions[name] = values
            sources.append(dict(model=model, condition=name, source=str(path)))
        new = list(read_lines(directory / 'predictions.jsonl'))
        assert len(new) == len({(r['case_id'],r['condition']) for r in new}) == 3972
        for name in ('neighbors_only', 'facts_only'):
            values = {}
            for row in new:
                if row['condition'] != name:
                    continue
                metric, value = score(cases[row['case_id']], row['prediction'])
                assert math.isclose(value, row['metrics'][metric], abs_tol=1e-12)
                values[row['case_id']] = value
            assert set(values) == set(cases)
            conditions[name] = values
            sources.append(dict(model=model, condition=name, source=str(directory / 'predictions.jsonl')))
        for category in ('ALL', 1, 2, 3, 4, 5):
            ids = [cid for cid,c in cases.items() if category == 'ALL' or c['category'] == category]
            for name, values in conditions.items():
                summary.append(dict(model=model, condition=name, category=category, questions=len(ids),
                    score=100*sum(values[cid] for cid in ids)/len(ids)))
            for name in ('centers_only', 'neighbors_only', 'facts_only'):
                differences = [conditions['full'][cid]-conditions[name][cid] for cid in ids]
                pairs.append(dict(model=model, comparator=name, category=category, questions=len(ids),
                    full_higher=sum(d>0 for d in differences), equal=sum(d==0 for d in differences),
                    full_lower=sum(d<0 for d in differences)))
    write_csv(output / 'context_components.csv', summary)
    write_csv(output / 'context_component_pairs.csv', pairs)
    write_json(output / 'context_component_audit.json', dict(complete=True, questions_per_model=1986,
        models=len(READERS), conditions=4, new_predictions=15888, native_scores_recomputed=True,
        sources=sources, note='All original categories retained; category 5 is abstention accuracy. Full and centers-only reuse verified main outputs; the two intermediate conditions use frozen selections, facts and native prompts.'))


def wiki_question_types(output):
    """Use released 2Wiki question types, without classifying question wording."""
    path = Path(__file__).resolve().parents[1] / 'baseline_algorithms/HippoRAG/reproduce/dataset/2wikimultihopqa.json'
    official = {r['_id']: r for r in json.loads(path.read_text())}
    questions = list(read_lines(output / 'evidence_recovery_questions.jsonl'))
    assert len(questions) == len(official) == 1000
    grouped = defaultdict(list)
    for row in questions:
        grouped[official[row['case_id']]['type']].append(row)
    summary = []
    for kind, rows in grouped.items():
        for method in rows[0]['selected']:
            hits = [len(set(r['gold_passages']) & set(r['selected'][method])) for r in rows]
            gold = [len(set(r['gold_passages'])) for r in rows]
            summary.append(dict(question_type=kind, method=method, questions=len(rows),
                supporting_passages=','.join(map(str, sorted(set(gold)))),
                recall_at_5=100*sum(h/g for h,g in zip(hits,gold))/len(rows),
                all_support_at_5=100*sum(h==g for h,g in zip(hits,gold))/len(rows)))
    write_csv(output / 'wiki_question_types.csv', summary)


if __name__ == "__main__":
    main()

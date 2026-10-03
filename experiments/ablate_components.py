"""Remove projection or propagation from frozen AMOR, retaining its QA protocol."""

import argparse
import csv
from collections import Counter
from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import random
import shutil
from types import SimpleNamespace
from time import perf_counter
import zipfile

from experiments.runner import _read_retrieval_records, _retrieval_record, _answer_prompt, _official_generation
from optimization.ircot import BASE, GRAPH, MODELS, Reader, evaluate_task, write_json
from optimization.report_results import TASK_METRICS, audited_score

ROOT = BASE / "optimization_component_ablation_seed42_20260927"
REFERENCE = BASE / "optimization_method_analysis_seed42_20260921/graph"
QUERY_CACHE = BASE / "optimization_query_embeddings_seed42_20260919"
VARIANTS = ("without_projection", "without_propagation")


def save_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(_retrieval_record(row), ensure_ascii=False) + "\n")
    temporary.replace(path)


def archive():
    ROOT.mkdir(parents=True, exist_ok=True)
    target = ROOT / f"code_{os.environ['SLURM_JOB_ID']}.zip"
    with zipfile.ZipFile(target, "x", zipfile.ZIP_DEFLATED) as output:
        for name in ("experiments/ablate_components.py", "experiments/ablate_components.sbatch",
                     "optimization/ircot.py", "experiments/runner.py",
                     "optimization/retriever/hipporag.py", "optimization/retriever/hybrid_graph.py",
                     "optimization/retriever/query_fact_context.py"):
            output.write(name, name)


def membership_graph(original, contents, entity_keys, normalize):
    """Return exactly the residual binary source links used by construction."""
    from optimization.graph_construction.source_consolidation import statement_weights

    selected = [(key, tuple(normalize(list(triple)))) for key, content in contents.items()
                for triple in content["retained_triples"]]
    weights = statement_weights(original, selected, entity_keys)
    result = original.copy()
    result.es["weight"] = weights.tolist()
    result.delete_edges([edge.index for edge in result.es
                         if edge["passage_source"] is None or edge["weight"] == 0])
    assert all(weight == 1 for weight in result.es["weight"])
    assert result.vs["name"] == original.vs["name"]
    return result, selected


class PropagationControl:
    """Keep recognition and seed construction; optionally bypass only PPR."""

    def __init__(self, hippo):
        self.hippo = hippo
        self.dense = hippo.dense_passage_retrieval
        self.ppr = hippo.run_ppr
        self.reset = self.dense_result = None
        self.disabled = False
        hippo.dense_passage_retrieval = self.capture_dense
        hippo.run_ppr = self.propagate

    def capture_dense(self, query):
        self.dense_result = self.dense(query)
        return self.dense_result

    def propagate(self, reset_prob, damping=None):
        self.reset = reset_prob.copy()
        if self.disabled:
            if self.dense_result is None:
                raise RuntimeError("PPR bypass requires this query's dense scores")
            return self.dense_result
        return self.ppr(reset_prob, damping=damping)

    def begin(self, disabled=False):
        self.disabled = disabled
        self.reset = self.dense_result = None

    def close(self):
        self.hippo.dense_passage_retrieval = self.dense
        self.hippo.run_ppr = self.ppr


def connection_controls(full, membership):
    """Remove either projected source weights or additional graph connections."""
    assert full.vs["name"] == membership.vs["name"]
    source_edges = set(membership.get_edgelist())
    assert source_edges.issubset(full.get_edgelist())
    assert all(weight == 1 for weight in membership.es["weight"])
    source_weighted = full.copy()
    source_weighted.delete_edges([edge.index for edge in full.es if edge.tuple not in source_edges])
    binary_sources = full.copy()
    binary_sources.es["weight"] = [1.0 if edge.tuple in source_edges else edge["weight"] for edge in full.es]
    assert set(source_weighted.get_edgelist()) == source_edges
    assert binary_sources.get_edgelist() == full.get_edgelist()
    return dict(without_projected_connections=source_weighted, without_source_reweighting=binary_sources)


def prepare(task, *, edge_controls=False):
    import igraph as ig
    import numpy as np
    from optimization.retriever.hipporag import CacheMissGuard, load_optimized_memory, load_query_embeddings
    from optimization.retriever.query_fact_context import QueryFactContext
    from optimization.graph_construction.statement_incidence import statement_incidence_graph, project_statement_graph
    from optimization.graph_construction.source_consolidation import statement_weights
    from optimization.run_graph import retrieval_config
    from utils.models import release_accelerator_memory

    expected = list(_read_retrieval_records(REFERENCE / "inputs/graph_retained_index_retained" / task / "retrieval.jsonl"))
    assert len(expected) == TASK_METRICS[task][0]
    assert len({(r.group_id, r.case.case_id) for r in expected}) == len(expected)
    config, _ = retrieval_config(SimpleNamespace(task=task.replace("_", " "), output_root=ROOT,
        generator_base_url="http://127.0.0.1:1/v1",
        path="/oscar/scratch/zliu328/agent-memory-data/locomo/locomo10.json",
        data_root=str(Path.cwd() / "baseline_algorithms/HippoRAG/reproduce/dataset")))
    for group in dict.fromkeys(r.group_id for r in expected):
        directory = ROOT / "groups" / task / group
        directory.mkdir(parents=True, exist_ok=True)
        original_rows = [r for r in expected if r.group_id == group]
        if (directory / "complete.json").exists():
            for variant in VARIANTS:
                rows = list(_read_retrieval_records(directory / variant / "retrieval.jsonl"))
                assert [(r.group_id, r.case) for r in rows] == [(r.group_id, r.case) for r in original_rows]
            continue
        runtime = ROOT / "runtime" / task / group
        cached, = (GRAPH / task / "runtime" / group / "llm_cache").glob("*.sqlite")
        destination = runtime / "llm_cache" / cached.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            if cached.with_name(cached.name + "-wal").exists():
                raise ValueError("Recognition cache has an active WAL")
            shutil.copy2(cached, destination)
        memory = load_optimized_memory(config, GRAPH / task / "memory" / group, runtime)
        control = selector = None
        try:
            hippo = memory._memory
            guard = CacheMissGuard(memory._generator)
            load_query_embeddings(hippo, QUERY_CACHE / task / group / "queries.npz")
            def reject_encoding(*args, **kwargs):
                raise RuntimeError("Missing frozen query embedding")
            hippo.embedding_model.batch_encode = reject_encoding
            from hipporag.utils.misc_utils import text_processing
            metadata = json.loads((GRAPH / task / "memory" / group / "graph.json").read_text())
            source = ig.Graph.Read_Pickle(metadata["source_graph"])
            entities = hippo.entity_embedding_store.get_all_id_to_rows()
            entity_keys = {row["content"]: key for key, row in entities.items()}
            binary, selected = membership_graph(source, memory.contents, entity_keys, text_processing)
            openie, = Path(metadata["source_graph"]).parent.parent.glob("openie_results_ner_*.json")
            docs = json.loads(openie.read_text())["docs"]
            source.es["weight"] = statement_weights(source, selected, entity_keys).tolist()
            incidence = statement_incidence_graph(source, docs, selected, entity_keys, text_processing, refined=True)
            reconstructed = project_statement_graph(incidence, source.vcount())
            full = hippo.graph
            assert full.vs["name"] == binary.vs["name"] == reconstructed.vs["name"]
            assert (full.get_adjacency_sparse(attribute="weight") != reconstructed.get_adjacency_sparse(attribute="weight")).nnz == 0
            binary.write_pickle(str(directory / "without_projection.pickle"))
            controls = connection_controls(full, binary) if edge_controls else {
                "without_projection": binary, "without_propagation": full}
            assert set(controls) == set(VARIANTS)
            if edge_controls:
                for variant, graph in controls.items():
                    graph.write_pickle(str(directory / (variant + ".pickle")))
            selector = QueryFactContext(hippo, memory.contents)
            control = PropagationControl(hippo)
            outputs = {variant: [] for variant in VARIANTS}
            calls = 0
            for row in original_rows:
                hippo.graph = full
                control.begin()
                full_items = memory.retrieve(row.case.question, 5)
                rendered = selector.render(row.case.question, full_items)
                guard.check()
                assert [i.text for i in rendered] == [i.text for i in row.retrieved], (task, group, row.case.case_id)
                full_reset = control.reset
                calls += full_reset is not None
                for variant in VARIANTS:
                    hippo.graph = controls[variant]
                    control.begin(disabled=variant == "without_propagation")
                    start = perf_counter()
                    items = memory.retrieve(row.case.question, 5)
                    guard.check()
                    if full_reset is None:
                        assert control.reset is None
                    else:
                        np.testing.assert_array_equal(full_reset, control.reset)
                    outputs[variant].append(replace(row, retrieved=selector.render(row.case.question, items),
                                                    retrieval_seconds=perf_counter() - start))
            for variant, rows in outputs.items():
                save_rows(directory / variant / "retrieval.jsonl", rows)
            write_json(directory / "complete.json", dict(complete=True, questions=len(original_rows),
                full_context_exact=True, reset_vectors_exact=True, ppr_queries=int(calls),
                full_edges=full.ecount(), membership_edges=binary.ecount(),
                new_llm_calls=guard.misses, new_embeddings=0))
            print("Prepared", task, group, len(original_rows), flush=True)
        finally:
            if control is not None:
                control.close()
            memory.close()
            memory = control = selector = None
            gc.collect()
            release_accelerator_memory()
    for variant in VARIANTS:
        rows = []
        for group in dict.fromkeys(r.group_id for r in expected):
            rows.extend(_read_retrieval_records(ROOT / "groups" / task / group / variant / "retrieval.jsonl"))
        assert [(r.group_id, r.case) for r in rows] == [(r.group_id, r.case) for r in expected]
        save_rows(ROOT / "inputs" / variant / task / "retrieval.jsonl", rows)
    write_json(ROOT / "inputs" / (task + ".json"), dict(complete=True, questions=len(expected), variants=VARIANTS))


def qa(model):
    for task, (count, _) in TASK_METRICS.items():
        assert json.loads((ROOT / "inputs" / (task + ".json")).read_text())["questions"] == count
    slug = model.replace("/", "_")
    directory = ROOT / slug
    directory.mkdir(parents=True, exist_ok=True)
    reader = Reader(model)
    try:
        previous = json.loads((BASE / "optimization_ircot_fact_context_seed42_20260920/main" / slug / "reader.json").read_text())
        assert {k: v for k, v in previous.items() if k != "gpu"} == {k: v for k, v in reader.metadata.items() if k != "gpu"}
        write_json(directory / f"reader_{os.environ['SLURM_JOB_ID']}.json", reader.metadata)
        for pilot in (True, False):
            for variant in VARIANTS:
                for task, (count, _) in TASK_METRICS.items():
                    target = directory / ("pilot" if pilot else "main") / variant / task
                    target.mkdir(parents=True, exist_ok=True)
                    source = ROOT / "inputs" / variant / task / "retrieval.jsonl"
                    rows = list(_read_retrieval_records(source))
                    if pilot:
                        groups = Counter()
                        selected = []
                        for row in rows:
                            if groups[row.group_id] < 2:
                                selected.append(row)
                            groups[row.group_id] += 1
                        rows = selected
                    else:
                        assert json.loads((directory / "pilot" / variant / task / "qa_complete.json").read_text())["complete"]
                        assert len(rows) == count
                    if (target / "qa_complete.json").exists():
                        complete = json.loads((target / "qa_complete.json").read_text())
                        assert complete["complete"] and complete["pilot"] == pilot and complete["questions"] == len(rows)
                        if not pilot:
                            assert audited_score(target / "evaluations" / slug, task,
                                {(r.group_id, r.case.case_id) for r in rows}) == complete["score"]
                        continue
                    if pilot:
                        save_rows(target / "retrieval.jsonl", rows)
                    elif not (target / "retrieval.jsonl").exists():
                        (target / "retrieval.jsonl").symlink_to(source)
                    rng = random.Random(42)
                    requests = [dict(phase="answer", messages=_answer_prompt(r.case, r.retrieved, rng)[0],
                                     generation=_official_generation(r.case)) for r in rows]
                    lengths = []
                    for start in range(0, len(requests), 32):
                        check = reader.request(phase="check_context", items=requests[start:start + 32])
                        assert not check["overflow"], str(target)
                        lengths.extend(check["input_tokens"])
                    write_json(target / "context_length.json", dict(questions=len(rows), input_tokens=lengths, overflow=[]))
                    evaluate_task(target, task, reader, model, pilot)
                    print("QA complete", model, pilot, variant, task, flush=True)
        write_json(directory / "complete.json", dict(complete=True, tasks=list(TASK_METRICS), variants=VARIANTS))
    finally:
        reader.close()


def report():
    from experiments.runner import _score
    from utils.hipporag_metrics import gold_passage_recall_at_k

    rows, retrieval = [], []
    for task, (count, metric) in TASK_METRICS.items():
        reference_rows = list(_read_retrieval_records(REFERENCE / "inputs/graph_retained_index_retained" / task / "retrieval.jsonl"))
        expected = {(r.group_id, r.case.case_id) for r in reference_rows}
        assert len(reference_rows) == len(expected) == count
        for variant in ("full", *VARIANTS):
            inputs = reference_rows if variant == "full" else list(_read_retrieval_records(ROOT / "inputs" / variant / task / "retrieval.jsonl"))
            assert [(r.group_id, r.case) for r in inputs] == [(r.group_id, r.case) for r in reference_rows]
            if task == "2WikiMultiHopQA":
                scores = []
                complete = 0
                for row in inputs:
                    sources = [i.metadata["original_source_text"] for i in row.retrieved
                               if i.metadata.get("context_representation") == "original_source_and_frozen_window"]
                    assert len(sources) == 5 and row.case.gold_passages
                    scores.append(gold_passage_recall_at_k(row.case.gold_passages, sources, 5))
                    complete += set(row.case.gold_passages).issubset(sources)
                retrieval.append(dict(variant=variant, questions=count, recall_at_5=100 * sum(scores) / count,
                                      all_support_percent=100 * complete / count, all_support_count=complete))
            for model in MODELS:
                slug = model.replace("/", "_")
                directory = (REFERENCE / slug / "main/graph_retained_index_retained" / task if variant == "full"
                             else ROOT / slug / "main" / variant / task)
                marker = json.loads((directory / "qa_complete.json").read_text())
                assert marker["complete"] and not marker["pilot"] and marker["questions"] == count
                output = directory / "evaluations" / slug
                score = audited_score(output, task, expected)
                assert score == marker["score"]
                with (output / "predictions.jsonl").open() as stream:
                    for row in inputs:
                        prediction = json.loads(next(stream))
                        assert (prediction["group_id"], prediction["case_id"]) == (row.group_id, row.case.case_id)
                        assert [i["text"] for i in prediction["retrieved"]] == [i.text for i in row.retrieved]
                        assert prediction["metrics"] == _score(row.case, prediction["prediction"], row.retrieved, row.top_k)
                    assert next(stream, None) is None
                usage = [json.loads(line) for line in (output / "qa_usage.jsonl").open()]
                assert len(usage) == count
                assert [r["case_id"] for r in usage] == [r.case.case_id for r in inputs]
                tokens = sum(r["input_tokens"] + r["output_tokens"] for r in usage)
                assert tokens == marker["input_tokens"] + marker["output_tokens"]
                rows.append(dict(model=model, task=task, variant=variant, questions=count, metric=metric,
                                 score=100 * score, qa_tokens=tokens, source=str(directory)))
    for filename, values in (("qa_results.csv", rows), ("retrieval_results.csv", retrieval)):
        with (ROOT / filename).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    scores = {(r["model"], r["task"], r["variant"]): r["score"] for r in rows}
    labels = {"full": "AMOR", "without_projection": "w/o hypergraph projection",
              "without_propagation": "w/o graph propagation"}
    table = [r"\begin{table*}[t]", r"\centering\small",
        r"\caption{One-shot component ablations (\%). Metrics follow the experimental setup. Each variant retains the same context construction procedure. Results use seed 42 and the development evaluation data.}",
        r"\label{tab:component-ablation}", r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r"Method & SH-Doc & MH-Doc & FC-SH & FC-MH & LoCoMo & 2Wiki \\"]
    comparisons = {}
    for model in MODELS:
        table.extend([r"\midrule", r"\multicolumn{7}{c}{\textbf{" + model.split("/")[-1] + r"}} \\"])
        for variant, label in labels.items():
            values = [scores[model, task, variant] for task in TASK_METRICS]
            table.append(label + " & " + " & ".join(f"{value:.2f}" for value in values) + r" \\")
    for variant in VARIANTS:
        differences = [scores[model, task, "full"] - scores[model, task, variant]
                       for model in MODELS for task in TASK_METRICS]
        comparisons[variant] = dict(full_wins=sum(d > 0 for d in differences),
                                   ties=sum(d == 0 for d in differences),
                                   full_losses=sum(d < 0 for d in differences))
    table.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (ROOT / "ablation_results.tex").write_text("\n".join(table) + "\n")
    write_json(ROOT / "complete.json", dict(complete=True, new_conditions=48, reference_conditions=24,
        tasks=list(TASK_METRICS), models=MODELS, variants=VARIANTS, qa=rows, retrieval=retrieval,
        comparisons=comparisons, native_scores_recomputed=True, context_text_checked=True,
        test_as_dev=True, seed=42))
    print(json.dumps(dict(complete=True, conditions=len(rows), retrieval=retrieval)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "qa", "report"))
    parser.add_argument("--task", choices=TASK_METRICS)
    parser.add_argument("--model", choices=MODELS)
    args = parser.parse_args()
    archive()
    protocol = dict(variants=VARIANTS, tasks=TASK_METRICS, models=MODELS, seed=42,
        without_projection="A=C: binary retained entity/source membership; no projected edges",
        without_propagation="Replace PPR output with existing dense passage ranking; keep recognition and seed calculation",
        fixed="Extracted facts, retained candidates, embeddings, recognition, BM25/RRF, source count, context procedure, QA prompts and scoring",
        reference=str(REFERENCE), full_context_reproduction_required=True,
        new_llm_calls=0, new_embeddings=0, test_as_dev=True)
    protocol = json.loads(json.dumps(protocol))
    path = ROOT / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == protocol
    else:
        write_json(path, protocol)
    if args.phase == "prepare":
        for task in ([args.task] if args.task else TASK_METRICS):
            prepare(task)
    elif args.phase == "qa":
        if not args.model:
            parser.error("qa requires --model")
        qa(args.model)
    else:
        report()


if __name__ == "__main__":
    main()

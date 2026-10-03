"""Frozen recommendation experiments: retrieval depth and seed/graph factorial.

Preparation is CPU-only and rejects new LLM/embedding requests. Generation uses
the existing native reader implementation. Every condition is a diagnostic,
not a replacement retrieval algorithm. Existing experiment artifacts are read-only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory


def source_edge_weights(statements, entity_keys):
    """Exact C + C*(HDH^T) operator on entity/source pairs, without graph IO."""
    sources = defaultdict(set)
    for source, triple in statements:
        if len(triple) != 3:
            raise ValueError("Expected an extracted subject/relation/object triple")
        sources[tuple(triple)].add(source)
    contributions = defaultdict(float)
    for (subject, relation, obj), records in sources.items():
        entities = {entity_keys[subject], entity_keys[obj]}
        if entities & records:
            raise ValueError("Entity and source identities overlap")
        members = entities | records
        weight = 1.0 / (len(members) - 1)
        for entity in entities:
            for source in records:
                contributions[tuple(sorted((entity, source)))] += weight
    return {edge: 1.0 + value for edge, value in contributions.items()}


def support_swap_plan(base, recommended, gold):
    """Choose paired source replacements by frozen rank, never by QA outcome.

    The control donor is UNANNOTATED, not assumed irrelevant or redundant.
    Source count and replacement position match; token length need not match.
    """
    base, recommended, gold = list(base), list(recommended), set(gold)
    if len(base) != len(set(base)) or len(recommended) != len(set(recommended)):
        raise ValueError("Duplicate ranked source IDs")
    missing = [key for key in recommended if key in gold and key not in base]
    sham = [key for key in recommended if key not in gold and key not in base]
    victims = [i for i, key in enumerate(base) if key not in gold]
    if not missing or not sham or not victims:
        return None
    return dict(position=victims[-1], removed=base[victims[-1]], support=missing[0],
                unannotated=sham[0], rule="last non-gold victim; first eligible donor in AMOR rank")


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_jsonl(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def freeze_design(output, design):
    output.mkdir(parents=True, exist_ok=True)
    path = output / "design.json"
    design = json.loads(json.dumps(design))
    if path.exists():
        assert json.loads(path.read_text()) == design
    else:
        dump(path, design)
    code = output / "recommendation_findings.py"
    current = Path(__file__).read_bytes()
    if code.exists():
        assert code.read_bytes() == current, "Code changed: use a new output directory"
    else:
        code.write_bytes(current)


def entries_for(results, task):
    from experiments.mine_cases import csv_rows
    entries = csv_rows(results / "qa_results.csv")
    return {row["variant"]: Path(row["source"]) for row in entries
            if row["task"] == task and row["setting"] == "one_shot"
            and row["model"] == "Qwen/Qwen3.5-4B"}


def load_frozen(task, group, root, runtime, results):
    import igraph as ig
    from experiments.ablate_memory import config_for, copy_cache
    from experiments.analyze_geometry import QUERIES
    from optimization.retriever.hipporag import load_optimized_memory, load_query_embeddings, CacheMissGuard
    copy_cache(root / task / "runtime" / group, runtime)
    memory = load_optimized_memory(config_for(task), root / task / "memory" / group, runtime)
    protocol = json.loads((results / "protocol.json").read_text())
    path = Path(protocol["graph_source"]) / "groups" / task / group / (protocol["graph_variant"] + ".pickle")
    graph = ig.Graph.Read_Pickle(str(path))
    assert graph.vs["name"] == memory._memory.graph.vs["name"]
    memory._memory.graph = graph
    guard = CacheMissGuard(memory._generator)
    load_query_embeddings(memory._memory, QUERIES / task / group / "queries.npz")
    def reject(*args, **kwargs):
        raise RuntimeError("Frozen experiment must not request new embeddings")
    memory._memory.embedding_model.batch_encode = reject
    return memory, guard


def prepare_depth(results, output, task):
    from experiments.analyze_geometry import GRAPH
    from experiments.ablate_components import PropagationControl, save_rows
    from experiments.ablate_memory import original_text_only
    from experiments.runner import _read_retrieval_records
    from optimization.retriever.query_fact_context import QueryFactContext
    design = dict(experiment="retrieval_depth_and_source_replacement", task=task,
        k=[5, 10, 15], methods=["amor", "without_recommendation"], seed=42,
        results=str(results), new_embeddings=False, new_recognition=False,
        context="Central raw sources for all depths; augmented context additionally at k=5 and k=15",
        swap="2Wiki only; oracle support vs unannotated donor at same position and source count; not equal tokens",
        gold_usage="Only metrics and explicitly labelled oracle interventions; never normal retrieval",
        limitations="Previously inspected evaluation data; no held-out discovery claim")
    freeze_design(output, design)
    sources = entries_for(results, task)
    refs = list(_read_retrieval_records(sources["full"] / "retrieval.jsonl"))
    baseline = {r.case.case_id: r for r in _read_retrieval_records(sources["without_propagation"] / "retrieval.jsonl")}
    groups = defaultdict(list)
    for row in refs:
        groups[row.group_id].append(row)
    outputs, traces = defaultdict(list), []
    for group, rows in groups.items():
        with TemporaryDirectory(prefix="amor-depth-") as temporary:
            memory, guard = load_frozen(task, group, GRAPH, Path(temporary), results)
            control = PropagationControl(memory._memory)
            try:
                selector = QueryFactContext(memory._memory, memory.contents)
                for index, row in enumerate(rows):
                    central = {}
                    for method, disabled in (("amor", False), ("without_recommendation", True)):
                        for k in design["k"]:
                            memory.base.rank_window = k
                            control.begin(disabled=disabled)
                            centers = memory.retrieve(row.case.question, k)
                            rendered = replace(row, retrieved=selector.render(row.case.question, centers), top_k=k)
                            guard.check()
                            if k == 5:
                                expected = row if method == "amor" else baseline[row.case.case_id]
                                assert [i.text for i in rendered.retrieved] == [i.text for i in expected.retrieved], row.case.case_id
                                central[method] = original_text_only(rendered)
                            plain = original_text_only(rendered)
                            outputs[f"{method}_k{k}_central"].append(plain)
                            if k in (5, 15):
                                outputs[f"{method}_k{k}_augmented"].append(rendered)
                            traces.append(dict(case_id=row.case.case_id, group_id=group, method=method, k=k,
                                gold=list(row.case.gold_passages or []),
                                sources=[i.metadata["original_source_text"] for i in centers]))
                    if row.case.gold_passages:
                        base, amor = central["without_recommendation"], central["amor"]
                        texts = lambda r: [i.metadata["original_source_text"] for i in r.retrieved]
                        plan = support_swap_plan(texts(base), texts(amor), row.case.gold_passages)
                        if plan is not None:
                            donors = {i.metadata["original_source_text"]: i for i in amor.retrieved}
                            outputs["swap_original"].append(base)
                            for kind in ("support", "unannotated"):
                                items = list(base.retrieved)
                                items[plan["position"]] = donors[plan[kind]]
                                outputs["swap_" + kind].append(replace(base, retrieved=items))
                            traces.append(dict(case_id=row.case.case_id, group_id=group, swap=plan))
                    if index % 100 == 0:
                        print("depth", task, group, index, "/", len(rows), flush=True)
            finally:
                control.close()
                memory.close()
    prepared = []
    for condition, rows in outputs.items():
        path = output / "inputs" / condition / "retrieval.jsonl"
        save_rows(path, rows)
        prepared.append(dict(task=task, condition=condition, questions=len(rows), path=str(path)))
    dump(output / "traces.json", traces)
    dump(output / "prepared.json", dict(complete=True, entries=prepared, exact_k5_replays=len(refs)*2,
        new_embeddings=0, new_recognition=0))


def prepare_consistency(results, output, task):
    import igraph as ig
    import numpy as np
    from baseline.base import RetrievedItem
    from experiments.analyze_geometry import GRAPH, BASE
    from experiments.ablate_components import save_rows
    from experiments.runner import _read_retrieval_records
    from optimization.retriever.compiled_sources import compiled_item
    from optimization.retriever.hybrid_graph import fuse_rankings
    from optimization.retriever.query_fact_context import QueryFactContext
    all_root = BASE / "optimization_support_ablation_seed42_20260917/graph_all_index_all"
    design = dict(experiment="recognition_pool_by_graph_fact_retention", task=task, results=str(results),
        seeds=["current", "all"], graphs=["current", "all"], k=5,
        fixed="Exact restart vector within each graph contrast; raw corpus, BM25, damping, current-fact augmentation",
        graph="All means all extracted facts including current and earlier facts, not a pre-update snapshot",
        seed="All-fact candidate recognition includes its native normalization, not an object-only substitution",
        postfilter="Current retained appended facts only; raw sources may still expose old assertions",
        limitation="This factorial does not by itself establish efficacy over oracle raw-assertion postfiltering",
        new_embeddings=False, new_recognition=False)
    freeze_design(output, design)
    sources = entries_for(results, task)
    refs = list(_read_retrieval_records(sources["full"] / "retrieval.jsonl"))
    assert len(refs) == 100 and {r.group_id for r in refs} == {"mab-0"}
    outputs, traces = defaultdict(list), []
    group = "mab-0"
    with TemporaryDirectory(prefix="amor-consistency-") as temporary:
        memories, guards = {}, {}
        try:
            for name, root in (("current", GRAPH), ("all", all_root)):
                memories[name], guards[name] = load_frozen(task, group, root, Path(temporary)/name, results)
            current, all_memory = memories["current"], memories["all"]
            # Loading the project's baseline installs its pinned HippoRAG import path.
            from hipporag.utils.misc_utils import text_processing
            assert current.contents == all_memory.contents
            hippo = current._memory
            names = hippo.graph.vs["name"]
            assert names == all_memory._memory.graph.vs["name"]
            entities = {v["content"]: k for k, v in hippo.entity_embedding_store.get_all_id_to_rows().items()}
            metadata = json.loads((GRAPH/task/"memory"/group/"graph.json").read_text())
            openie, = Path(metadata["source_graph"]).parent.parent.glob("openie_results_ner_*.json")
            docs = json.loads(openie.read_text())["docs"]
            selected = [(key, tuple(text_processing(list(triple)))) for key, item in current.contents.items()
                        for triple in item["retained_triples"]]
            all_statements = [(doc["idx"], tuple(text_processing(triple))) for doc in docs for triple in doc["extracted_triples"]]
            expected = {tuple(sorted((names[e.source], names[e.target]))): e["weight"] for e in hippo.graph.es}
            reconstructed = source_edge_weights(selected, entities)
            assert expected.keys() == reconstructed.keys()
            np.testing.assert_allclose(list(expected.values()), [reconstructed[e] for e in expected], rtol=1e-12, atol=1e-12)
            all_weights = source_edge_weights(all_statements, entities)
            positions = {name: i for i, name in enumerate(names)}
            all_graph = ig.Graph(n=len(names), edges=[(positions[a], positions[b]) for a,b in all_weights], directed=False)
            all_graph.vs["name"] = names
            all_graph.es["weight"] = list(all_weights.values())
            graphs = dict(current=hippo.graph, all=all_graph)
            retained = {triple for key, triple in selected}
            capture, native_ppr = {}, {}
            for label, memory in memories.items():
                engine = memory._memory
                original_ppr, original_recognition = engine.run_ppr, engine.rerank_facts
                native_ppr[label] = original_ppr
                def record_ppr(reset_prob, damping=None, _label=label, _native=original_ppr):
                    capture[_label]["reset"] = reset_prob.copy()
                    capture[_label]["damping"] = damping
                    return _native(reset_prob, damping)
                def record_facts(*args, _label=label, _native=original_recognition, **kwargs):
                    answer = _native(*args, **kwargs)
                    capture[_label]["facts"] = [tuple(text_processing(list(f))) for f in answer[1]]
                    return answer
                engine.run_ppr, engine.rerank_facts = record_ppr, record_facts
            selector = QueryFactContext(hippo, current.contents)
            source_rows = hippo.chunk_embedding_store.get_all_id_to_rows()
            for index, row in enumerate(refs):
                query = row.case.question
                hippo.graph = graphs["current"]
                capture.clear()
                actual = {}
                for label, memory in memories.items():
                    capture[label] = {}
                    actual[label] = memory.retrieve(query, 5)
                    guards[label].check()
                rendered = selector.render(query, actual["current"])
                assert [i.text for i in rendered] == [i.text for i in row.retrieved], row.case.case_id
                lexical = current.base.lexical.retrieve(query, 5)
                trace = dict(case_id=row.case.case_id, query=query, seeds={}, retrieved={})
                for seed, values in capture.items():
                    trace["seeds"][seed] = dict(facts=values.get("facts", []),
                        excluded_from_current=[f for f in values.get("facts", []) if f not in retained],
                        has_reset="reset" in values)
                    if "reset" in values:
                        trace["seeds"][seed]["reset_sha256"] = hashlib.sha256(values["reset"].tobytes()).hexdigest()
                    for graph_name, graph in graphs.items():
                        condition = f"seeds_{seed}_graph_{graph_name}"
                        hippo.graph = graph
                        if "reset" in values:
                            reset = values["reset"].copy()
                            order, scores = native_ppr["current"](reset, values["damping"])
                            np.testing.assert_array_equal(reset, values["reset"])
                            keys = [hippo.passage_node_keys[i] for i in order[:5]]
                            graph_items = [RetrievedItem(source_rows[key]["content"], float(score), {"source_passage":key})
                                           for key, score in zip(keys, scores[:5], strict=True)]
                            fused = fuse_rankings(graph_items, lexical, 5)
                            centers = [compiled_item(item, item.metadata["source_passage"], current.contents[item.metadata["source_passage"]])
                                       if "source_passage" in item.metadata else
                                       compiled_item(item, current.text_to_key[item.text], current.contents[current.text_to_key[item.text]])
                                       for item in fused]
                        else:
                            centers = actual[seed]
                        context = selector.render(query, centers)
                        if graph_name == "current":
                            replay = selector.render(query, actual[seed])
                            assert [i.text for i in context] == [i.text for i in replay]
                        outputs[condition].append(replace(row, retrieved=context))
                        trace["retrieved"][condition] = [i.metadata["source_passage"] for i in centers]
                traces.append(trace)
                if index % 10 == 0:
                    print("consistency", task, index, "/", len(refs), flush=True)
            dump(output/"graph_audit.json", dict(current_reconstruction_exact=True,
                current_edges=graphs["current"].ecount(), all_edges=graphs["all"].ecount(),
                current_fact_occurrences=len(selected), all_fact_occurrences=len(all_statements)))
        finally:
            for memory in memories.values():
                memory.close()
    prepared = []
    for condition, rows in outputs.items():
        path = output/"inputs"/condition/"retrieval.jsonl"
        save_rows(path, rows)
        prepared.append(dict(task=task, condition=condition, questions=len(rows), path=str(path)))
    dump(output/"traces.json", traces)
    dump(output/"prepared.json", dict(complete=True, entries=prepared, new_embeddings=0, new_recognition=0,
        exact_current_context_replays=len(refs)))


def qa(output, model):
    from experiments.runner import _read_retrieval_records, _answer_prompt, _official_generation, _score
    from optimization.ircot import Reader, MODELS, SEED, BATCH_SIZE
    protocol = json.loads((output/"prepared.json").read_text())
    assert protocol["complete"]
    directory = output/"qa"/model.replace("/", "_")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/"predictions.jsonl"
    completed = {(r["condition"],r["case_id"]):r for r in read_jsonl(path)} if path.exists() else {}
    prepared = []
    for entry in protocol["entries"]:
        rng = random.Random(SEED)
        rows = list(_read_retrieval_records(Path(entry["path"])))
        assert len(rows) == entry["questions"]
        for row in rows:
            messages, answer_key = _answer_prompt(row.case, row.retrieved, rng)
            request = dict(phase="answer", messages=messages, generation=_official_generation(row.case))
            key = (entry["condition"], row.case.case_id)
            if key in completed:
                assert completed[key]["request"] == request
            else:
                prepared.append((entry, row, request, answer_key))
    if prepared:
        reader = Reader(model)
        try:
            dump(directory/"reader.json", reader.metadata)
            for start in range(0, len(prepared), BATCH_SIZE):
                batch = prepared[start:start+BATCH_SIZE]
                requests = [item[2] for item in batch]
                assert not reader.request(phase="check_context", items=requests)["overflow"]
                response = reader.request(phase="generate", items=requests)
                for (entry,row,request,answer_key),answer in zip(batch,response["responses"],strict=True):
                    # This phase evaluates 2Wiki and FC only. LoCoMo adversarial
                    # multiple-choice scoring requires the native answer-key path.
                    assert answer_key is None
                    record = dict(condition=entry["condition"], case_id=row.case.case_id, group_id=row.group_id,
                        task=entry["task"], prediction=answer["answer"], request=request,
                        metrics=_score(row.case,answer["answer"],row.retrieved,row.top_k),
                        usage=answer["usage"], generation_settings=answer["generation_settings"])
                    with path.open("a") as stream:
                        stream.write(json.dumps(record,ensure_ascii=False)+"\n")
                    completed[entry["condition"],row.case.case_id] = record
                print(model,len(completed),"completed",flush=True)
        finally:
            reader.close()
    assert len(completed) == sum(e["questions"] for e in protocol["entries"])
    dump(directory/"complete.json",dict(complete=True, model=model, revision=MODELS[model],
        requests=len(completed), seed=SEED))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=["depth","consistency","qa"])
    parser.add_argument("--results",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--task")
    parser.add_argument("--model")
    args=parser.parse_args()
    if args.phase == "depth":
        prepare_depth(args.results,args.output,args.task)
    elif args.phase == "consistency":
        prepare_consistency(args.results,args.output,args.task)
    else:
        qa(args.output,args.model)


if __name__ == "__main__":
    main()

"""Compare published CatRAG key-fact weighting with frozen AMOR associations.

This is an operator comparison, not a reproduction of the full CatRAG pipeline.
All conditions share reciprocal directed arcs, recognition, personalization,
source corpus, damping, and rank fusion. No QA, extraction, or embedding calls.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import igraph as ig
import numpy as np

from baseline.base import RetrievedItem
from experiments.analyze_geometry import GRAPH, QUERIES
from experiments.mine_cases import read_lines, csv_rows, write_json, write_csv
from experiments.report_paper_costs import COUNTS, TASKS
from experiments.runner import _read_retrieval_records
from optimization.retriever.hipporag import CacheMissGuard, load_optimized_memory, load_query_embeddings
from optimization.retriever.hybrid_graph import fuse_rankings
from optimization.run_graph import retrieval_config


def key_fact_weights(graph, base_weights, fact_sources, accepted, entities, variant):
    """CatRAG §3.4: released operator or published Eq. 3, forward arcs only.

    Released CatRAG.py:1825--1895 multiplies by 2.5 and preserves each
    entity's total passage-edge weight. The paper instead specifies 1+beta,
    beta=2.5, without that rescaling. Both are reported, with no tuning.
    """
    weights = base_weights.copy()
    targets = defaultdict(set)
    for triple in accepted:
        for phrase in {triple[0], triple[2]}:
            if phrase in entities:
                targets[entities[phrase]].update(fact_sources[triple])
    changes = 0
    for entity, sources in targets.items():
        edges = graph.incident(entity, mode="out")
        # This experiment's shared graph contains only entity/source links.
        assert all(graph.vs[graph.es[e].target]["name"].startswith("chunk-") for e in edges)
        matching = [e for e in edges if graph.vs[graph.es[e].target]["name"] in sources]
        if not matching:
            continue
        before = float(weights[edges].sum())
        weights[matching] *= 2.5 if variant == "released" else 3.5
        if variant == "released":
            weights[edges] *= before / weights[edges].sum()
            assert np.isclose(weights[edges].sum(), before)
        changes += len(matching)
    return weights, changes


def verify_released_operator(graph, weights, fact_sources, accepted, entities):
    """Pilot equality against the unmodified block in the released code."""
    from hipporag.utils.misc_utils import compute_mdhash_id
    source = Path("baseline_algorithms/CatRAG/src/catrag/CatRAG.py").read_text()
    start = source.index("        modified_edges = {}", source.index("def graph_search_with_fact_entities"))
    end = source.index("        # End of adjust the seed entity->passage edge weight", start)
    block = "\n".join(line[8:] for line in source[start:end].splitlines())
    mapping = defaultdict(lambda: defaultdict(set))
    for triple, sources in fact_sources.items():
        fact_id = compute_mdhash_id(str(triple), prefix="fact-")
        for phrase in {triple[0], triple[2]}:
            for key in sources:
                mapping[compute_mdhash_id(phrase, prefix="entity-")][key].add(fact_id)
    copied = graph.copy()
    copied.es["weight"] = weights.tolist()
    native = SimpleNamespace(graph=copied, ent_node_to_chunk_ids=mapping,
                             node_name_to_vertex_idx={v["name"]: v.index for v in copied.vs})
    scope = dict(self=native, top_k_facts=accepted, compute_mdhash_id=compute_mdhash_id,
                 phrases_and_ids={(p, entities.get(p)) for f in accepted for p in (f[0], f[2])})
    exec(compile(block, "CatRAG_released_key_fact_block", "exec"), scope)
    actual, _ = key_fact_weights(graph, weights, fact_sources, accepted, entities, "released")
    np.testing.assert_allclose(actual, copied.es["weight"], rtol=1e-13, atol=1e-13)


def passage_order(graph, weights, reset, damping, passage_indices):
    values = np.asarray(graph.personalized_pagerank(damping=damping, directed=True,
                        weights=weights.tolist(), reset=reset, implementation="prpack"))
    scores = values[passage_indices]
    order = np.argsort(scores)[::-1]
    return order, scores[order], values


def run(results, output):
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((results / "protocol.json").read_text())
    design = dict(tasks=TASKS, questions=COUNTS, conditions=[
        "unit", "amor", "unit_released", "amor_released", "unit_paper", "amor_paper"],
        fixed="Recognition, personalization, symmetric directed topology, damping, BM25 and top-five",
        released_operator="CatRAG.py 1825-1895: factor 2.5, entity passage mass preserved, forward arcs",
        paper_operator="ACL 2026 Findings 290, Eq.3 and Table7: factor 1+2.5, forward arcs",
        comparison="Published operator transferred to a shared topology, not full CatRAG",
        new_llm_calls=False, new_embeddings=False, qa=False, results=str(results))
    design_path = output / "design.json"
    if design_path.exists():
        assert json.loads(design_path.read_text()) == design
    else:
        write_json(design_path, design)
    entries = csv_rows(results / "qa_results.csv")
    summary, all_audits = [], []
    for task in TASKS:
        entry, = [r for r in entries if r["task"] == task and r["setting"] == "one_shot"
                  and r["variant"] == "full" and r["model"] == "Qwen/Qwen3.5-4B"]
        references = list(_read_retrieval_records(Path(entry["source"]) / "retrieval.jsonl"))
        assert len(references) == COUNTS[task]
        groups = defaultdict(list)
        for row in references:
            groups[row.group_id].append(row)
        target = output / (task + ".jsonl")
        completed = {r["case_id"]: r for r in read_lines(target)} if target.exists() else {}
        assert set(completed).issubset({r.case.case_id for r in references})
        for group, rows in groups.items():
            if all(r.case.case_id in completed for r in rows):
                continue
            with TemporaryDirectory(prefix="fact-weighting-") as temporary:
                runtime = Path(temporary)
                task_argument = task.replace("_", " ")
                print("Loading", task_argument, group, "from", __file__, flush=True)
                config, _ = retrieval_config(SimpleNamespace(task=task_argument, output_root=runtime,
                    generator_base_url="http://127.0.0.1:1/v1",
                    path="/oscar/scratch/zliu328/agent-memory-data/locomo/locomo10.json",
                    data_root=str(Path.cwd() / "baseline_algorithms/HippoRAG/reproduce/dataset")))
                cache, = (GRAPH / task / "runtime" / group / "llm_cache").glob("*.sqlite")
                assert not cache.with_name(cache.name + "-wal").exists()
                (runtime / "llm_cache").mkdir()
                shutil.copy2(cache, runtime / "llm_cache" / cache.name)
                memory = load_optimized_memory(config, GRAPH / task / "memory" / group, runtime)
                try:
                    from hipporag.utils.misc_utils import text_processing
                    hippo, hybrid = memory._memory, memory.base
                    graph_path = Path(protocol["graph_source"]) / "groups" / task / group / (protocol["graph_variant"] + ".pickle")
                    graph = ig.Graph.Read_Pickle(str(graph_path))
                    assert not graph.is_directed() and not any(graph.is_loop())
                    hippo.graph = graph
                    directed = graph.as_directed(mode="mutual")
                    names = graph.vs["name"]
                    positions = {key: index for index, key in enumerate(names)}
                    entities = {r["content"]: positions[k] for k, r in
                                hippo.entity_embedding_store.get_all_id_to_rows().items()}
                    fact_sources = defaultdict(set)
                    for key, content in memory.contents.items():
                        for triple in content["retained_triples"]:
                            fact_sources[tuple(text_processing(list(triple)))].add(key)
                    sources = hippo.chunk_embedding_store.get_all_id_to_rows()
                    text_keys = {r["content"]: k for k, r in sources.items()}
                    guard = CacheMissGuard(memory._generator)
                    load_query_embeddings(hippo, QUERIES / task / group / "queries.npz")
                    def reject(*args, **kwargs):
                        raise RuntimeError("Fact weighting comparison requires frozen embeddings")
                    hippo.embedding_model.batch_encode = reject
                    captured = {}
                    native_ppr, native_recognition = hippo.run_ppr, hippo.rerank_facts
                    def capture_ppr(reset_prob, damping=None):
                        captured.update(reset=reset_prob.copy(), damping=0.5 if damping is None else damping)
                        return native_ppr(reset_prob, damping)
                    def capture_recognition(*args, **kwargs):
                        value = native_recognition(*args, **kwargs)
                        captured["facts"] = [tuple(text_processing(list(f))) for f in value[1]]
                        assert all(len(f) == 3 and f in fact_sources for f in captured["facts"])
                        return value
                    hippo.run_ppr, hippo.rerank_facts = capture_ppr, capture_recognition
                    base = dict(unit=np.ones(directed.ecount()), amor=np.asarray(directed.es["weight"]))
                    piloted = False
                    for reference in rows:
                        cid, query = reference.case.case_id, reference.case.question
                        expected = [r.metadata["source_passage"] for r in reference.retrieved if "source_passage" in r.metadata]
                        if cid in completed:
                            assert completed[cid]["selected"]["amor"] == expected
                            continue
                        captured.clear()
                        actual = memory.retrieve(query, 5)
                        guard.check()
                        assert [i.metadata["source_passage"] for i in actual] == expected, cid
                        lexical = hybrid.lexical.retrieve(query, 5)
                        selected, boosted = {}, {}
                        gold = [text_keys[t] for t in (reference.case.gold_passages or ())]
                        for condition in design["conditions"]:
                            if "reset" not in captured:
                                selected[condition], boosted[condition] = expected, 0
                                continue
                            kind, *operator = condition.split("_")
                            weights = base[kind]
                            changes = 0
                            if operator:
                                weights, changes = key_fact_weights(directed, weights, fact_sources,
                                                                    captured["facts"], entities, operator[0])
                                assert changes > 0, (cid, "recognized facts must have source edges")
                            order, scores, values = passage_order(directed, weights, captured["reset"],
                                                                 captured["damping"], hippo.passage_node_idxs)
                            if condition == "amor":
                                native_values = graph.personalized_pagerank(damping=captured["damping"],
                                    directed=False, weights="weight", reset=captured["reset"], implementation="prpack")
                                np.testing.assert_allclose(values, native_values, rtol=1e-10, atol=1e-13)
                            items = [RetrievedItem(sources[hippo.passage_node_keys[i]]["content"], float(s),
                                     {"source_passage": hippo.passage_node_keys[i]}) for i, s in zip(order[:5], scores[:5])]
                            fused = fuse_rankings(items, lexical, 5, hybrid.rank_constant)
                            selected[condition] = [text_keys[i.text] for i in fused]
                            boosted[condition] = changes
                        assert selected["amor"] == expected, (cid, "reciprocal directed baseline changed")
                        if not piloted and "reset" in captured:
                            for weights in base.values():
                                verify_released_operator(directed, weights, fact_sources, captured["facts"], entities)
                            piloted = True
                            print(task, group, "native operator and symmetric PPR pilot passed", flush=True)
                        row = dict(task=task, group=group, case_id=cid, question=query,
                                   category=reference.case.category, gold=gold, selected=selected,
                                   recognized_facts=captured.get("facts", []), matched_edges=boosted,
                                   dense_fallback="reset" not in captured)
                        with target.open("a") as stream:
                            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                        completed[cid] = row
                        if len(completed) % 100 == 0:
                            print(task, len(completed), "/", COUNTS[task], flush=True)
                    guard.check()
                finally:
                    memory.close()
        assert len(completed) == COUNTS[task]
        values = list(completed.values())
        for condition in design["conditions"]:
            row = dict(task=task, condition=condition, questions=len(values),
                       changed_source_sets=sum(set(r["selected"][condition]) != set(r["selected"]["amor"]) for r in values),
                       recall_at_5="", complete_at_5="")
            if task == "2WikiMultiHopQA":
                assert all(r["gold"] for r in values)
                row.update(recall_at_5=100*sum(len(set(r["gold"]) & set(r["selected"][condition]))/len(set(r["gold"])) for r in values)/len(values),
                           complete_at_5=100*sum(set(r["gold"]).issubset(r["selected"][condition]) for r in values)/len(values))
            summary.append(row)
        all_audits.append(dict(task=task, questions=len(values), fallback=sum(r["dense_fallback"] for r in values)))
        write_csv(output / "scores.csv", summary)
        write_json(output / "progress.json", all_audits)
    write_json(output / "complete.json", dict(complete=True, questions=sum(COUNTS.values()),
        conditions=design["conditions"], tasks=all_audits, exact_amor_replay=True,
        released_operator_pilot_verified=True, new_llm_calls=0, new_embeddings=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.results, args.output)

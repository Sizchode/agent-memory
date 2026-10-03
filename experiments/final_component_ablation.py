"""Four single-component interventions on final AMOR; frozen native one-shot QA."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory

import numpy as np

from experiments.encoder_robustness import DEFAULT_RESULTS, RecognitionAudit, digest, json_save, source_rows
from experiments.final_parameter_sensitivity import TASKS, READERS

VARIANTS = ("without_bm25_fusion", "uniform_weights", "without_recognizer", "raw_relations")
REFERENCE = Path("/oscar/scratch/zliu328/agent-memory-outputs/final_parameter_sensitivity_20261002")


class KeepCandidateFacts:
    """Identity at the native recognizer seam, after native candidate retrieval."""

    def __init__(self):
        self.calls = []

    def __call__(self, query, candidate_facts, candidate_fact_indices, len_after_rerank):
        if len(candidate_facts) != len(candidate_fact_indices):
            raise ValueError("Fact IDs and facts disagree")
        if len(candidate_facts) > len_after_rerank:
            raise ValueError("Native candidate count exceeds declared limit")
        self.calls.append(dict(query=query, indices=list(candidate_fact_indices), facts=list(candidate_facts)))
        return list(candidate_fact_indices), list(candidate_facts), {}


@contextmanager
def retrieval_intervention(memory, variant):
    """Change one runtime component and restore native objects even on failure."""
    engine = memory._memory
    original_graph, original_filter, original_base = engine.graph, engine.rerank_filter, memory.base
    identity = None
    try:
        if variant == "uniform_weights":
            graph = original_graph.copy()
            weights = np.asarray(graph.es["weight"])
            if not np.isfinite(weights).all() or not (weights > 0).all():
                raise ValueError("Expected positive finite final-graph edges")
            graph.es["weight"] = [1.0] * graph.ecount()
            engine.graph = graph
        elif variant == "without_bm25_fusion":
            from optimization.retriever.hybrid_graph import HybridGraphMemory
            if not isinstance(original_base, HybridGraphMemory):
                raise TypeError("BM25 intervention requires the native fusion wrapper")
            memory.base = original_base.base
        elif variant == "without_recognizer":
            identity = KeepCandidateFacts()
            engine.rerank_filter = identity
        else:
            raise ValueError(f"Not a runtime intervention: {variant}")
        yield identity
    finally:
        engine.graph, engine.rerank_filter, memory.base = original_graph, original_filter, original_base


def check_reset(reference, actual):
    if reference is None or actual is None:
        if reference is not actual:
            raise ValueError("Propagation eligibility changed in a fixed-initialization control")
    else:
        np.testing.assert_array_equal(reference, actual)


def initialize(args):
    source = json.loads((args.results / "protocol.json").read_text())
    if source["graph_variant"] != "without_projected_connections":
        raise ValueError("Not the final AMOR graph")
    json_save(args.output / "protocol.json", dict(
        experiment="final_amor_four_component_ablation", variants=VARIANTS, tasks=TASKS, readers=READERS,
        source_results=str(args.results), source_protocol_sha256=digest(args.results / "protocol.json"),
        reference_qa=str(args.reference), source_k=5, fusion_window=5, seed=42,
        alpha=0.5, passage_node_weight=0.05, graph_variant=source["graph_variant"],
        interventions=dict(without_bm25_fusion="Remove BM25 branch and RRF, keep native graph/dense ranking",
            uniform_weights="All existing final-graph edges become one; same facts, seeds and connectivity",
            without_recognizer="Native top-five fact candidates pass unchanged to native seed construction",
            raw_relations="Disable relation canonicalization during retention; rebuild final membership graph, fact candidates and augmentation from resulting retained facts"),
        fixed="Original passages, source order, extraction, embeddings, query sets, other algorithms and generation protocol",
        raw_relations_note="Retention-induced changes to candidates, graph, recognition and appended facts are intended downstream effects",
        reference_policy="Reuse freshly generated sensitivity reference only after exact context, generation and hash verification",
        reporting="All six tasks, all four readers, all four variants; native metrics; no condition selection",
        recognition="Frozen historical native cache replay with parse-failure audit; no response repair or new recognition",
        qa_requests=54176, test_as_dev=True,
        code_sha256={str(p): digest(p) for p in (Path(__file__), Path("experiments/encoder_robustness.py"),
            Path("experiments/recommendation_findings.py"), Path("experiments/final_parameter_sensitivity.py"))}))


def load_raw(task, group, runtime):
    """Verify the raw-relation construction, then retain only weighted membership edges."""
    import igraph as ig
    from experiments.ablate_memory import RAW, config_for, copy_cache, check_raw_graph
    from experiments.ablate_components import membership_graph, connection_controls
    from experiments.analyze_geometry import QUERIES, GRAPH
    from optimization.retriever.hipporag import load_optimized_memory, load_query_embeddings, CacheMissGuard

    copy_cache(RAW / task / "runtime" / group, runtime)
    memory = load_optimized_memory(config_for(task), RAW / task / "memory" / group, runtime)
    try:
        from hipporag.utils.misc_utils import text_processing
        audit = check_raw_graph(memory, task, group)
        raw_meta = json.loads((RAW / task / "memory" / group / "graph.json").read_text())
        canonical_meta = json.loads((GRAPH / task / "memory" / group / "graph.json").read_text())
        canonical_contents = json.loads(Path(canonical_meta["compiled_source_file"]).read_text())
        for key, content in memory.contents.items():
            for field in ("original_source_text", "source_position", "timestamp", "window_sources"):
                if content.get(field) != canonical_contents[key].get(field):
                    raise ValueError(f"Raw-relation control changed original source metadata: {field}")
        engine = memory._memory
        original = ig.Graph.Read_Pickle(raw_meta["source_graph"])
        entities = {row["content"]: key for key, row in engine.entity_embedding_store.get_all_id_to_rows().items()}
        membership, selected = membership_graph(original, memory.contents, entities, text_processing)
        final_graph = connection_controls(engine.graph, membership)["without_projected_connections"]
        from experiments.recommendation_findings import source_edge_weights
        expected = source_edge_weights(selected, entities)
        actual = {tuple(sorted((final_graph.vs[e.source]["name"], final_graph.vs[e.target]["name"]))): e["weight"]
            for e in final_graph.es}
        if expected.keys() != actual.keys():
            raise ValueError("Raw-relation final graph membership differs from retained triples")
        np.testing.assert_allclose([actual[k] for k in expected], list(expected.values()), rtol=1e-12, atol=1e-12)
        engine.graph = final_graph
        guard = CacheMissGuard(memory._generator)
        load_query_embeddings(engine, QUERIES / task / group / "queries.npz")
        def reject(*args, **kwargs):
            raise RuntimeError("Frozen raw-relation intervention cannot request new embeddings")
        engine.embedding_model.batch_encode = reject
        audit.update(final_membership_equation_verified=True, original_source_metadata_unchanged=True,
            raw_metadata_sha256=digest(RAW / task / "memory" / group / "graph.json"),
            raw_contents_sha256=digest(Path(raw_meta["compiled_source_file"])),
            edges=final_graph.ecount(), facts=len(engine.fact_node_keys))
        return memory, guard, audit
    except BaseException:
        memory.close()
        raise


def prepare(args):
    from experiments.analyze_geometry import GRAPH
    from experiments.recommendation_findings import load_frozen
    from experiments.ablate_components import PropagationControl, save_rows
    from experiments.runner import _read_retrieval_records
    from optimization.retriever.query_fact_context import QueryFactContext
    from optimization.report_results import TASK_METRICS
    from utils.models import release_accelerator_memory

    initialize(args)
    expected, sources = source_rows(args.results, args.task)
    if len(expected) != TASK_METRICS[args.task][0]:
        raise ValueError("Incomplete task input")
    for group in dict.fromkeys(row.group_id for row in expected):
        rows = [row for row in expected if row.group_id == group]
        directory = args.output / "groups" / args.task / group
        directory.mkdir(parents=True, exist_ok=True)
        marker = directory / "complete.json"
        if marker.exists():
            saved = json.loads(marker.read_text())
            for variant in ("reference", *VARIANTS):
                if digest(directory / variant / "retrieval.jsonl") != saved["retrieval_sha256"][variant]:
                    raise ValueError("Completed group artifact changed")
            continue
        outputs = {variant: [] for variant in ("reference", *VARIANTS)}
        trace, construction = [], None
        for mode in ("canonical", "raw"):
            runtime = args.runtime_root / mode / args.task / group
            if mode == "canonical":
                memory, guard = load_frozen(args.task, group, GRAPH, runtime, args.results)
            else:
                memory, guard, construction = load_raw(args.task, group, runtime)
            engine = memory._memory
            control = PropagationControl(engine)
            audit = RecognitionAudit(engine.rerank_filter, directory / (mode + "_recognition.jsonl"), policy="historical_replay")
            try:
                if (engine.global_config.damping, engine.global_config.passage_node_weight,
                    engine.global_config.linking_top_k, memory.base.rank_window) != (0.5, 0.05, 5, 5):
                    raise ValueError("Native retrieval defaults differ")
                selector = QueryFactContext(engine, memory.contents)
                edges, weights, facts = engine.graph.get_edgelist(), list(engine.graph.es["weight"]), list(engine.fact_node_keys)
                engine.graph.write_pickle(str(directory / (mode + ".pickle")))
                for index, row in enumerate(rows):
                    control.begin()
                    centers = memory.retrieve(row.case.question, 5)
                    rendered = selector.render(row.case.question, centers)
                    guard.check()
                    audit.check()
                    if mode == "canonical":
                        if [i.text for i in rendered] != [i.text for i in row.retrieved]:
                            raise ValueError(f"Final AMOR replay differs: {row.case.case_id}")
                        outputs["reference"].append(replace(row, retrieved=rendered, top_k=5))
                        reset = None if control.reset is None else control.reset.copy()
                        for variant in VARIANTS[:-1]:
                            with retrieval_intervention(memory, variant) as identity:
                                control.begin()
                                changed_centers = memory.retrieve(row.case.question, 5)
                                changed = selector.render(row.case.question, changed_centers)
                                guard.check()
                                audit.check()
                                if variant != "without_recognizer":
                                    check_reset(reset, control.reset)
                                else:
                                    if len(identity.calls) != 1:
                                        raise ValueError("Native candidate recognizer seam was not reached exactly once")
                                    trace.append(dict(case_id=row.case.case_id, **identity.calls[0]))
                                outputs[variant].append(replace(row, retrieved=changed, top_k=5))
                    else:
                        outputs["raw_relations"].append(replace(row, retrieved=rendered, top_k=5))
                    if index % 50 == 0:
                        print("prepared", args.task, group, mode, index, len(rows), flush=True)
                if engine.graph.get_edgelist() != edges or list(engine.graph.es["weight"]) != weights or list(engine.fact_node_keys) != facts:
                    raise ValueError("Intervention leaked into the baseline memory")
                json_save(directory / (mode + "_audit.json"), dict(responses=audit.responses, cache_misses=guard.misses,
                    graph_sha256=digest(directory / (mode + ".pickle"))))
            finally:
                audit.close()
                control.close()
                memory.close()
                memory = engine = selector = control = None
                release_accelerator_memory()
        hashes = {}
        for variant, selected in outputs.items():
            path = directory / variant / "retrieval.jsonl"
            save_rows(path, selected)
            hashes[variant] = digest(path)
        json_save(directory / "recognizer_bypass.json", trace)
        json_save(marker, dict(complete=True, questions=len(rows), raw_construction=construction,
            exact_reference_replay=True, fixed_initialization_controls=True, retrieval_sha256=hashes,
            source_retrieval_sha256=digest(sources["full"] / "retrieval.jsonl")))
    merged_hashes = {}
    for variant in ("reference", *VARIANTS):
        collected = []
        for group in dict.fromkeys(row.group_id for row in expected):
            collected.extend(_read_retrieval_records(args.output / "groups" / args.task / group / variant / "retrieval.jsonl"))
        by_key = {(r.group_id, r.case.case_id): r for r in collected}
        keys = [(r.group_id, r.case.case_id) for r in expected]
        if len(by_key) != len(collected) or set(by_key) != set(keys):
            raise ValueError("Missing or duplicate question identities")
        path = args.output / "inputs" / variant / args.task / "retrieval.jsonl"
        save_rows(path, [by_key[key] for key in keys])
        merged_hashes[variant] = digest(path)
    json_save(args.output / "inputs" / (args.task + ".json"), dict(complete=True, questions=len(expected), retrieval_sha256=merged_hashes))


def qa(args):
    from optimization.ircot import Reader, evaluate_task, BATCH_SIZE
    from optimization.report_results import TASK_METRICS
    from experiments.runner import _read_retrieval_records, _answer_prompt, _official_generation
    from experiments.ablate_components import save_rows

    initialize(args)
    for task in TASKS:
        marker = json.loads((args.output / "inputs" / (task + ".json")).read_text())
        if not marker["complete"] or marker["questions"] != TASK_METRICS[task][0]:
            raise ValueError("Incomplete preparation")
        for variant in VARIANTS:
            if digest(args.output / "inputs" / variant / task / "retrieval.jsonl") != marker["retrieval_sha256"][variant]:
                raise ValueError("Prepared input hash differs")
    reader = Reader(args.model)
    try:
        for variant in VARIANTS:
            for task in TASKS:
                directory = args.output / ("qa_smoke" if args.smoke else "qa") / args.model.replace("/", "_") / variant / task
                if (directory / "verified.json").exists():
                    continue
                records = list(_read_retrieval_records(args.output / "inputs" / variant / task / "retrieval.jsonl"))
                if args.smoke:
                    records = [next(row for row in records if row.case.category == c) for c in dict.fromkeys(r.case.category for r in records)]
                directory.mkdir(parents=True, exist_ok=True)
                save_rows(directory / "retrieval.jsonl", records)
                json_save(directory / "reader.json", reader.metadata)
                rng = random.Random(42)
                requests = [dict(phase="answer", messages=_answer_prompt(r.case, r.retrieved, rng)[0],
                    generation=_official_generation(r.case)) for r in records]
                for start in range(0, len(requests), BATCH_SIZE):
                    if reader.request(phase="check_context", items=requests[start:start+BATCH_SIZE])["overflow"]:
                        raise ValueError(f"Context overflow: {variant}/{task}; refusing truncation")
                evaluate_task(directory, task, reader, args.model, args.smoke)
                completed = json.loads((directory / "qa_complete.json").read_text())
                json_save(directory / "verified.json", dict(complete=True, model=args.model, task=task, setting=variant,
                    pilot=args.smoke, questions=len(records), score=completed["score"],
                    retrieval_sha256=digest(directory / "retrieval.jsonl"),
                    predictions_sha256=digest(directory / "evaluations" / args.model.replace("/", "_") / "predictions.jsonl")))
                print("QA complete", args.model, variant, task, completed["score"], flush=True)
    finally:
        reader.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "qa"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--model", choices=READERS)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare" and args.task is None:
        parser.error("prepare requires --task")
    if args.phase == "qa" and args.model is None:
        parser.error("qa requires --model")
    if args.phase == "prepare":
        with TemporaryDirectory(prefix="amor-component-runtime-") as temporary:
            args.runtime_root = Path(temporary)
            prepare(args)
    else:
        qa(args)


if __name__ == "__main__":
    main()

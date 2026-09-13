"""Build frozen graph variants, replay fixed retrieval, and run complete QA."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from time import perf_counter
import types

from baseline.base import RetrievedItem
from experiments.run_anchormem import TASKS
from experiments.run_gap_query_memory import MeasuredFinalAnswer
from experiments.runner import RetrievedCase, _retrieval_record, evaluate_retrieval
from main import _config_from_args, _load_groups, _seed_everything, build_parser
from optimization.graph_construction.context_weights import VARIANTS, construct_weights
from optimization.graph_construction.source_consolidation import VARIANTS as CONSOLIDATION_VARIANTS, SCHEMA_VARIANTS, CANONICAL_VARIANTS
from optimization.graph_construction.fact_index import VARIANTS as INDEX_VARIANTS
from optimization.graph_construction.weight_grid import CONFIGS as WEIGHT_CONFIGS
from optimization.graph_construction.provenance_weights import VARIANTS as PROVENANCE_VARIANTS
from optimization.graph_construction.adaptive_synonyms import VARIANTS as ADAPTIVE_VARIANTS
from optimization.graph_construction.compiled_sources import VARIANTS as FACT_CONTEXT_VARIANTS, SOURCE_VARIANTS
from optimization.graph_construction.source_window import VARIANTS as WINDOW_VARIANTS
from optimization.retriever.hybrid_graph import VARIANTS as HYBRID_VARIANTS
from optimization.retriever.compiled_sources import PACKING_VARIANTS

COMPILED_VARIANTS = (*FACT_CONTEXT_VARIANTS, *SOURCE_VARIANTS, *WINDOW_VARIANTS)


BASE = Path("/oscar/scratch/zliu328/agent-memory-outputs")
SOURCE = BASE / "final_qwen3_30b_seed42_clean_20260910"
MODELS = ("Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-2B")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def retrieve(args):
    import numpy as np
    from scipy import sparse
    from optimization.retriever.hipporag import CacheMissGuard, load_memory

    slug = args.task.replace(" ", "_")
    source = args.source_root / "hipporag2" / slug
    control = args.output_root / "original" / slug
    common = build_parser().parse_args([
        "--task", args.task, "--baseline", "hipporag2", "--output-dir", str(control),
        "--chunk-size", "512", "--path", args.path, "--data-root", args.data_root,
        "--generator-base-url", args.generator_base_url])
    groups = list(_load_groups(common))
    official = json.loads((Path(__file__).resolve().parents[1] / "experiments/configs/hipporag2.json").read_text())
    config = _config_from_args(common).hipporag_config(official)
    variants = ("original", *args.variants)
    references = [json.loads(line) for line in (source / "retrieval.jsonl").open()]
    reference = {(row["group_id"], row["case"]["case_id"]): row for row in references}
    expected = {(g.group_id, c.case_id) for g in groups for c in g.cases}
    if len(reference) != len(references) or set(reference) != expected:
        raise ValueError("Source and official loader do not cover the same complete task")
    with ExitStack() as stack:
        streams = {}
        for variant in variants:
            directory = args.output_root / variant / slug
            directory.mkdir(parents=True, exist_ok=False)
            write_json(directory / "settings.json", dict(
                variant=variant, task=args.task, seed=42, full_questions=len(expected),
                source_memory=str(source), source_efficiency=str(source / "efficiency.jsonl"),
                fixed_retrieval="HippoRAG recognition, reset, PPR, top5 original passages",
                graph_inputs="source graph, entity provenance, source passage embeddings only",
                test_set_used_as_development_set=True, extra_construction_llm_calls=0,
                reused_extraction_cost_not_zero=True, config=config))
            streams[variant] = stack.enter_context((directory / "retrieval.jsonl").open("x"))
        total, original_matches = 0, 0
        for group in groups:
            runtime = control / "memory" / group.group_id
            runtime.mkdir(parents=True)
            start = perf_counter()
            memory = load_memory(config, source / "hipporag_indices" / group.group_id, runtime)
            guard = CacheMissGuard(memory._generator) if not args.allow_generator_calls else None
            try:
                hippo = memory._memory
                if set(hippo.chunk_embedding_store.get_all_texts()) != set(group.memory_items):
                    raise ValueError("Frozen index and original source passages differ")
                loading_seconds = perf_counter() - start
                original_graph = hippo.graph
                start = perf_counter()
                weights = construct_weights(original_graph, hippo.ent_node_to_chunk_ids,
                                            hippo.passage_node_keys, hippo.passage_embeddings)
                construction_seconds = perf_counter() - start
                write_json(runtime / "construction.json", dict(
                    source_graph=str(Path(hippo.working_dir) / "graph.pickle"),
                    vertices=original_graph.vcount(), edges=original_graph.ecount(),
                    loading_seconds=loading_seconds, construction_seconds=construction_seconds,
                    incremental_generator_calls=0, variants=list(args.variants)))
                for variant in args.variants:
                    destination = args.output_root / variant / slug / "memory" / group.group_id
                    destination.mkdir(parents=True)
                    np.save(destination / "edge_weights.npy", weights[variant])
                    write_json(destination / "graph.json", dict(
                        source_graph=str(Path(hippo.working_dir) / "graph.pickle"),
                        edge_order="unchanged source graph edge order", variant=variant,
                        zero_weight_edges=int(np.count_nonzero(weights[variant] == 0)),
                        nonzero_edges=int(np.count_nonzero(weights[variant])), frozen=True))

                original_ppr = hippo.run_ppr
                capture = {}

                def capture_ppr(self, reset_prob, damping=0.5):
                    capture["reset"] = reset_prob.copy()
                    capture["damping"] = damping
                    return original_ppr(reset_prob, damping=damping)

                hippo.run_ppr = types.MethodType(capture_ppr, hippo)
                candidate_graph = original_graph.copy()
                resets, query_records = [], []
                with (runtime / "retrieval_timing.jsonl").open("x") as timing:
                    for case in group.cases:
                        capture.clear()
                        hippo.graph = original_graph
                        start = perf_counter()
                        solution = hippo.retrieve([case.question], num_to_retrieve=5)[0]
                        if guard:
                            guard.check()
                        seconds = perf_counter() - start
                        old = reference[(group.group_id, case.case_id)]
                        original_matches += solution.docs == [r["text"] for r in old["retrieved"]]
                        if "reset" in capture:
                            resets.append(sparse.csr_matrix(capture["reset"].reshape(1, -1)))
                        else:
                            resets.append(sparse.csr_matrix((1, original_graph.vcount())))
                        query_records.append(dict(case_id=case.case_id, question=case.question,
                                                  graph_seeds=solution.graph_seeds,
                                                  dense_fallback="reset" not in capture))
                        for variant in variants:
                            search_start = perf_counter()
                            if variant == "original" or "reset" not in capture:
                                texts, scores = solution.docs, solution.doc_scores
                            else:
                                candidate_graph.es["weight"] = weights[variant].tolist()
                                hippo.graph = candidate_graph
                                ids, scores = original_ppr(capture["reset"], damping=capture["damping"])
                                texts = [hippo.chunk_embedding_store.get_row(hippo.passage_node_keys[i])["content"]
                                         for i in ids[:5]]
                                scores = scores[:5]
                            items = tuple(RetrievedItem(text, float(score)) for text, score in zip(texts, scores, strict=True))
                            row = _retrieval_record(RetrievedCase(group.group_id, case, items, 5,
                                                                 seconds if variant == "original" else perf_counter() - search_start))
                            row["retrieval_protocol"] = variant
                            row["shared_recognition_and_embedding"] = variant != "original"
                            streams[variant].write(json.dumps(row, ensure_ascii=False) + "\n")
                            streams[variant].flush()
                        timing.write(json.dumps(dict(case_id=case.case_id, shared_retrieval_seconds=seconds,
                                                      candidate_ppr_seconds=perf_counter() - start - seconds)) + "\n")
                        timing.flush()
                        total += 1
                        if total % 25 == 0:
                            print(json.dumps(dict(task=args.task, retrieved=total, reference_matches=original_matches)), flush=True)
                hippo.graph = original_graph
                sparse.save_npz(runtime / "query_resets.npz", sparse.vstack(resets, format="csr"))
                write_json(runtime / "queries.json", query_records)
                write_json(runtime / "retrieval_cost.json", memory.efficiency_metrics())
            finally:
                memory.close()
        if total != len(expected):
            raise ValueError("Incomplete task")
        for variant in variants:
            write_json(args.output_root / variant / slug / "retrieval_complete.json", dict(
                questions=total, original_reference_matches=original_matches,
                original_reference_questions=len(expected)))
        print(json.dumps(dict(task=args.task, complete=total, original_matches=original_matches)), flush=True)


def evaluate(args):
    from utils.models import HuggingFaceChatModel
    model = HuggingFaceChatModel(args.evaluation_backbone, max_tokens=2048,
                                dtype="bfloat16", device_map="cuda", seed=42)
    model._ensure_loaded()
    for variant in args.variants:
        directory = args.output_root / variant / args.task.replace(" ", "_")
        complete = json.loads((directory / "retrieval_complete.json").read_text())
        output = directory / "evaluations" / args.evaluation_backbone.replace("/", "_")
        output.mkdir(parents=True, exist_ok=False)
        _seed_everything(42)
        with (output / "qa_usage.jsonl").open("x") as stream:
            summary = evaluate_retrieval(directory / "retrieval.jsonl",
                answer_model=MeasuredFinalAnswer(model, stream), output_dir=output, seed=42)
        with (output / "predictions.jsonl").open() as stream:
            count = sum(1 for _ in stream)
        if count != complete["questions"]:
            raise ValueError("Incomplete evaluation")
        print(json.dumps(dict(task=args.task, model=args.evaluation_backbone, variant=variant,
                              questions=count, summary=summary)), flush=True)


def verify(args):
    from optimization.retriever.hipporag import CacheMissGuard, load_optimized_memory

    common = build_parser().parse_args([
        "--task", args.task, "--baseline", "hipporag2", "--output-dir", str(args.output_root),
        "--generator-base-url", args.generator_base_url])
    official = json.loads((Path(__file__).resolve().parents[1] / "experiments/configs/hipporag2.json").read_text())
    config = _config_from_args(common).hipporag_config(official)
    for variant in args.variants:
        directory = args.output_root / variant / args.task.replace(" ", "_")
        complete = json.loads((directory / "retrieval_complete.json").read_text())
        rows = [json.loads(line) for line in (directory / "retrieval.jsonl").open()]
        if len(rows) != complete["questions"]:
            raise ValueError("Incomplete retrieval artifact")
        matches = 0
        for group_id in dict.fromkeys(row["group_id"] for row in rows):
            memory = load_optimized_memory(config, directory / "memory" / group_id,
                                           directory / "verification" / group_id)
            guard = CacheMissGuard(memory._generator) if not args.allow_generator_calls else None
            try:
                for row in rows:
                    if row["group_id"] != group_id:
                        continue
                    items = memory.retrieve(row["case"]["question"], 5)
                    if guard:
                        guard.check()
                    if [item.text for item in items] != [item["text"] for item in row["retrieved"]]:
                        raise ValueError(f"Saved graph retrieval mismatch: {group_id}, {row['case']['case_id']}")
                    matches += 1
            finally:
                memory.close()
        write_json(directory / "loaded_graph_verified.json", dict(questions=matches, matched=matches,
            uses_saved_query_resets=False, interface="optimization.retriever.hipporag.load_optimized_memory"))
        print(json.dumps(dict(task=args.task, variant=variant, loaded_graph_matches=matches)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("retrieve", "evaluate", "verify"), required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=SOURCE)
    parser.add_argument("--variants", nargs="+", choices=("original", *VARIANTS, *CONSOLIDATION_VARIANTS, *SCHEMA_VARIANTS, *INDEX_VARIANTS, *CANONICAL_VARIANTS, *WEIGHT_CONFIGS, *PROVENANCE_VARIANTS, *ADAPTIVE_VARIANTS, *COMPILED_VARIANTS, *HYBRID_VARIANTS, *PACKING_VARIANTS), default=list(VARIANTS))
    parser.add_argument("--evaluation-backbone", choices=MODELS, default=MODELS[0])
    parser.add_argument("--generator-base-url", default="http://127.0.0.1:9/v1")
    parser.add_argument("--allow-generator-calls", action="store_true")
    parser.add_argument("--path", default="/oscar/scratch/zliu328/agent-memory-data/locomo/locomo10.json")
    parser.add_argument("--data-root", default=str(Path(__file__).resolve().parents[1] / "baseline_algorithms/HippoRAG/reproduce/dataset"))
    args = parser.parse_args()
    if len(set(args.variants)) != len(args.variants) or (args.phase == "retrieve" and "original" in args.variants):
        parser.error("Do not repeat variants; retrieval always includes the original control")
    if args.phase == "retrieve" and any(v in (*CONSOLIDATION_VARIANTS, *SCHEMA_VARIANTS, *INDEX_VARIANTS, *CANONICAL_VARIANTS, *WEIGHT_CONFIGS, *PROVENANCE_VARIANTS, *ADAPTIVE_VARIANTS, *COMPILED_VARIANTS, *HYBRID_VARIANTS, *PACKING_VARIANTS) for v in args.variants):
        parser.error("Use optimization.replay_graph to construct consolidation variants")
    if args.phase == "verify" and "original" in args.variants:
        parser.error("Verification loads optimized weight artifacts, not the original control")
    _seed_everything(42)
    {"retrieve": retrieve, "evaluate": evaluate, "verify": verify}[args.phase](args)


if __name__ == "__main__":
    main()

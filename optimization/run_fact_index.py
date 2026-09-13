"""Freeze graph and fact-index pruning, then run unchanged HippoRAG retrieval."""

import argparse
from contextlib import ExitStack
import gc
import json
import pickle
from pathlib import Path
import subprocess
from time import perf_counter
import types

import numpy as np

from baseline.base import RetrievedItem
from baseline.graph_usage import GraphUsageRecorder
from experiments.run_anchormem import TASKS
from experiments.runner import RetrievedCase, _retrieval_record
from main import _config_from_args, _seed_everything, build_parser
from optimization.graph_construction.fact_index import VARIANTS, apply_fact_index, construct_fact_index
from optimization.graph_construction.source_consolidation import latest_relation_weights
from optimization.retriever.hipporag import GenerationFailureGuard, load_memory
from optimization.run_graph import SOURCE, write_json


def load_source_groups(common):
    """Run the existing loader in the exact source-baseline dependency environment."""
    code = """import argparse, json, pickle, sys
from main import _load_groups
groups = list(_load_groups(argparse.Namespace(**json.loads(sys.argv[1]))))
pickle.dump(groups, sys.stdout.buffer)
"""
    result = subprocess.run([
        "/oscar/scratch/zliu328/agent-memory-envs/hipporag/bin/python", "-c", code,
        json.dumps(vars(common))], stdout=subprocess.PIPE, check=True)
    return pickle.loads(result.stdout)


def completed_group_prefix(rows, groups):
    actual = [(r["group_id"], r["case"]["case_id"], r["case"]["question"]) for r in rows]
    expected = [(g.group_id, c.case_id, c.question) for g in groups for c in g.cases]
    if actual != expected[:len(actual)]:
        raise ValueError("Existing retrieval is not the original task prefix")
    completed, offset = set(), 0
    for group in groups:
        stop = offset + len(group.cases)
        if offset < len(rows) < stop:
            raise ValueError("Resume requires complete source groups, not a partial group")
        if stop <= len(rows):
            completed.add(group.group_id)
        offset = stop
    return completed


def retrieve_task(args, task):
    slug = task.replace(" ", "_")
    common = build_parser().parse_args([
        "--task", task, "--baseline", "hipporag2", "--output-dir", str(args.output_root),
        "--chunk-size", "512", "--path", args.path, "--data-root", args.data_root,
        "--generator-base-url", args.generator_base_url])
    groups = load_source_groups(common)
    expected = {(g.group_id, c.case_id) for g in groups for c in g.cases}
    official = json.loads((Path(__file__).resolve().parents[1] / "experiments/configs/hipporag2.json").read_text())
    config = _config_from_args(common).hipporag_config(official)
    source = SOURCE / "hipporag2" / slug
    reference_rows = [json.loads(line) for line in (source / "retrieval.jsonl").open()]
    reference = {(r["group_id"], r["case"]["case_id"]): r for r in reference_rows}
    if set(reference) != expected or len(reference_rows) != len(expected):
        raise ValueError("Source reference does not cover the complete task")
    with ExitStack() as stack:
        streams = {}
        completed_groups = None
        for variant in VARIANTS:
            directory = args.output_root / variant / slug
            directory.mkdir(parents=True, exist_ok=args.resume_completed_groups)
            retrieval_path = directory / "retrieval.jsonl"
            existing = [json.loads(line) for line in retrieval_path.open()] if retrieval_path.exists() else []
            completed = completed_group_prefix(existing, groups)
            if completed_groups is not None and completed != completed_groups:
                raise ValueError("Variant retrieval files have different completed source groups")
            completed_groups = completed
            streams[variant] = stack.enter_context(retrieval_path.open("a" if args.resume_completed_groups else "x"))
            settings = dict(
                task=task, variant=variant, config=config, seed=42, top_k=5,
                test_set_used_as_development_set=True, full_questions=len(expected),
                construction_reads_questions=False, original_passages_preserved=True,
                schema_root=str(args.schema_root) if variant == "index_schema" else None,
                source_order_is_event_time=False, fact_wording_and_embeddings="original retained rows",
                retrieval="Original HippoRAG recognition and PPR; new recognition when candidates change",
                source_efficiency=str(source / "efficiency.jsonl"))
            settings_path = directory / "settings.json"
            if settings_path.exists():
                saved_settings = json.loads(settings_path.read_text())
                saved_settings["config"]["llm_base_url"] = config["llm_base_url"]
                if saved_settings != settings:
                    raise ValueError("Existing task settings differ")
            if not settings_path.exists():
                write_json(settings_path, settings)
        total, original_matches = 0, 0
        for group in groups:
            if group.group_id in completed_groups:
                total += len(group.cases)
                original_matches += len(group.cases)
                continue
            runtime = args.output_root / "runtime" / slug / group.group_id
            runtime.mkdir(parents=True, exist_ok=args.resume_completed_groups)
            memory = load_memory(config, source / "hipporag_indices" / group.group_id, runtime)
            try:
                memory._generator.openai_client.models.list()
                usage_stream = stack.enter_context((runtime / "generation_usage.jsonl").open("a" if args.resume_completed_groups else "x"))
                client = memory._generator.openai_client
                client.chat.completions.create = GraphUsageRecorder(usage_stream).wrap(
                    client.chat.completions.create, method="fact_index", stage="recognition", group_id=group.group_id)
                guard = GenerationFailureGuard(memory._generator)
                from hipporag.utils.misc_utils import text_processing
                hippo = memory._memory
                graph = hippo.graph
                text_to_key = {row["content"]: key for key, row in hippo.chunk_embedding_store.get_all_id_to_rows().items()}
                if set(text_to_key) != set(group.memory_items):
                    raise ValueError("Original passage store differs from loader")
                ordered = list(dict.fromkeys(text_to_key[text] for text in group.memory_items))
                entities = {row["content"]: key for key, row in hippo.entity_embedding_store.get_all_id_to_rows().items()}
                source_graph = Path(hippo.working_dir) / "graph.pickle"
                documents = json.loads(next(source_graph.parent.parent.glob("openie_results_ner_*.json")).read_text())["docs"]
                schema_dir = args.schema_root / slug / group.group_id
                schema = json.loads((schema_dir / "schema.json").read_text())
                if json.loads((schema_dir / "complete.json").read_text())["relations"] != len(schema):
                    raise ValueError("Incomplete source schema")
                fact_rows = hippo.fact_embedding_store.get_all_id_to_rows()
                original = (list(hippo.fact_node_keys), hippo.fact_embeddings,
                            hippo.ent_node_to_chunk_ids, hippo.proc_triples_to_docs, graph.es["weight"][:])
                variants = {}
                for variant in VARIANTS:
                    selected_schema = schema if variant == "index_schema" else None
                    start = perf_counter()
                    weights, stats = latest_relation_weights(graph, documents, ordered, entities,
                                                              text_processing, selected_schema)
                    weight = weights["schema_latest" if selected_schema is not None else "latest_relation"]
                    index = construct_fact_index(documents, ordered, entities, fact_rows, text_processing, selected_schema)
                    apply_fact_index(hippo, index)
                    variants[variant] = (weight, (hippo.fact_node_keys, hippo.fact_embeddings,
                                                 hippo.ent_node_to_chunk_ids, hippo.proc_triples_to_docs))
                    destination = args.output_root / variant / slug / "memory" / group.group_id
                    destination.mkdir(parents=True)
                    np.save(destination / "edge_weights.npy", weight)
                    write_json(destination / "fact_index.json", index)
                    write_json(destination / "graph.json", dict(
                        source_graph=str(source_graph), variant=variant, frozen=True,
                        edge_order="unchanged source graph edge order", construction_seconds=perf_counter() - start,
                        retained_facts=index["retained_fact_count"], source_facts=index["original_fact_count"],
                        statistics=stats))
                # Both construction artifacts are frozen before retrieval sees any questions.
                rerank = hippo.rerank_facts

                def checked_rerank(self, query, scores):
                    result = rerank(query, scores)
                    if "error" in result[2]:
                        raise RuntimeError(result[2]["error"])
                    return result

                hippo.rerank_facts = types.MethodType(checked_rerank, hippo)
                with (runtime / "retrieval_timing.jsonl").open("x") as timing:
                    for case in group.cases:
                        hippo.fact_node_keys, hippo.fact_embeddings = original[:2]
                        hippo.ent_node_to_chunk_ids, hippo.proc_triples_to_docs = original[2:4]
                        graph.es["weight"] = original[4]
                        control = hippo.retrieve([case.question], num_to_retrieve=5)[0]
                        guard.check()
                        if control.docs != [r["text"] for r in reference[(group.group_id, case.case_id)]["retrieved"]]:
                            raise ValueError("Original retrieval changed in the new runtime; refusing a confounded comparison")
                        original_matches += 1
                        for variant in VARIANTS:
                            weight, state = variants[variant]
                            graph.es["weight"] = weight.tolist()
                            start = perf_counter()
                            hippo.fact_node_keys, hippo.fact_embeddings = state[:2]
                            hippo.ent_node_to_chunk_ids, hippo.proc_triples_to_docs = state[2:]
                            solution = hippo.retrieve([case.question], num_to_retrieve=5)[0]
                            guard.check()
                            elapsed = perf_counter() - start
                            items = tuple(RetrievedItem(text, float(score))
                                          for text, score in zip(solution.docs, solution.doc_scores, strict=True))
                            row = _retrieval_record(RetrievedCase(group.group_id, case, items, 5, elapsed))
                            row.update(retrieval_protocol=variant, shared_recognition_and_embedding=False)
                            streams[variant].write(json.dumps(row, ensure_ascii=False) + "\n")
                            streams[variant].flush()
                            timing.write(json.dumps(dict(case_id=case.case_id, variant=variant, seconds=elapsed)) + "\n")
                            timing.flush()
                        total += 1
                        if total % 25 == 0:
                            print(json.dumps(dict(task=task, questions=total, original_matches=original_matches)), flush=True)
                write_json(runtime / "retrieval_cost.json", memory.efficiency_metrics())
            finally:
                memory.close()
                memory = hippo = rerank = checked_rerank = None
                gc.collect()
                import torch
                torch.cuda.empty_cache()
        if total != len(expected):
            raise ValueError("Incomplete retrieval")
        for variant in VARIANTS:
            write_json(args.output_root / variant / slug / "retrieval_complete.json",
                       dict(questions=total, original_reference_matches=original_matches))
        print(json.dumps(dict(task=task, complete=total, original_matches=original_matches)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--schema-root", type=Path, required=True)
    parser.add_argument("--generator-base-url", required=True)
    parser.add_argument("--resume-completed-groups", "--resume-empty-task", dest="resume_completed_groups", action="store_true")
    parser.add_argument("--path", default="/oscar/scratch/zliu328/agent-memory-data/locomo/locomo10.json")
    parser.add_argument("--data-root", default=str(Path(__file__).resolve().parents[1] / "baseline_algorithms/HippoRAG/reproduce/dataset"))
    args = parser.parse_args()
    _seed_everything(42)
    for task in args.tasks:
        retrieve_task(args, task)


if __name__ == "__main__":
    main()

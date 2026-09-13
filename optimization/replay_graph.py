"""Construct source-consolidated graphs and replay saved HippoRAG query seeds."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from time import perf_counter

import igraph as ig
import numpy as np
import pandas as pd
from scipy import sparse

from experiments.run_anchormem import TASKS
from main import _load_groups, build_parser
from optimization.graph_construction.source_consolidation import VARIANTS, SCHEMA_VARIANTS, CANONICAL_VARIANTS, latest_relation_weights, retained_statements
from optimization.graph_construction.weight_grid import CONFIGS as WEIGHT_CONFIGS, calibrated_weights
from optimization.graph_construction.provenance_weights import VARIANTS as PROVENANCE_VARIANTS, provenance_weights
from optimization.graph_construction.adaptive_synonyms import VARIANTS as ADAPTIVE_VARIANTS, adaptive_weights
from optimization.run_graph import SOURCE, write_json


def replay(args):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline_algorithms/HippoRAG/src"))
    from hipporag.utils.misc_utils import text_processing

    slug = args.task.replace(" ", "_")
    control = args.seed_root / "original" / slug
    complete = json.loads((control / "retrieval_complete.json").read_text())
    if complete["original_reference_matches"] != complete["questions"]:
        raise ValueError("Exact original retrieval replay required before using cached seeds")
    common = build_parser().parse_args([
        "--task", args.task, "--baseline", "bm25", "--output-dir", str(args.output_root),
        "--chunk-size", "512", "--path", args.path, "--data-root", args.data_root])
    groups = list(_load_groups(common))
    rows = [json.loads(line) for line in (control / "retrieval.jsonl").open()]
    reference = {(r["group_id"], r["case"]["case_id"]): r for r in rows}
    expected = {(g.group_id, c.case_id) for g in groups for c in g.cases}
    if set(reference) != expected or len(rows) != len(expected):
        raise ValueError("Full source task coverage mismatch")
    settings = json.loads((control / "settings.json").read_text())
    # The source replay uses the official default unless explicitly configured.
    from hipporag.utils.config_utils import BaseConfig
    damping = BaseConfig(**settings["config"]).damping
    variants = tuple(WEIGHT_CONFIGS) if args.weight_grid else CANONICAL_VARIANTS if args.consolidate_multiple else SCHEMA_VARIANTS if args.schema_root else VARIANTS
    if args.provenance_weights:
        variants = PROVENANCE_VARIANTS
    if args.adaptive_synonyms:
        variants = ADAPTIVE_VARIANTS
    with ExitStack() as stack:
        streams = {}
        for variant in variants:
            directory = args.output_root / variant / slug
            directory.mkdir(parents=True, exist_ok=False)
            streams[variant] = stack.enter_context((directory / "retrieval.jsonl").open("x"))
            write_json(directory / "settings.json", dict(
                task=args.task, variant=variant, seed=42, top_k=5, damping=damping,
                source_memory=str(SOURCE / "hipporag2" / slug), source_query_seeds=str(control),
                construction_reads_queries=False, source_order_is_event_time=False,
                test_set_used_as_development_set=True, additional_generator_calls_this_replay=0,
                source_schema=str(args.schema_root) if args.schema_root else None,
                consolidate_multiple_relations=args.consolidate_multiple,
                weight_calibration=WEIGHT_CONFIGS.get(variant) if args.weight_grid else None,
                passage_entity_retained_fraction=args.provenance_weights,
                entity_assertion_compatible_synonyms=args.adaptive_synonyms,
                historical_construction_and_recognition_costs_excluded_from_incremental_time=True))
        total = 0
        for group in groups:
            original = control / "memory" / group.group_id
            source_graph = Path(json.loads((original / "construction.json").read_text())["source_graph"])
            graph = ig.Graph.Read_Pickle(str(source_graph))
            index_dir = source_graph.parent
            passage_df = pd.read_parquet(index_dir / "chunk_embeddings/vdb_chunk.parquet")
            entity_df = pd.read_parquet(index_dir / "entity_embeddings/vdb_entity.parquet")
            passage_keys = passage_df["hash_id"].tolist()
            passage_text = dict(zip(passage_keys, passage_df["content"], strict=True))
            text_to_key = {text: key for key, text in passage_text.items()}
            # Source order comes from the existing loader, not the shuffled OpenIE export.
            ordered_keys = list(dict.fromkeys(text_to_key[text] for text in group.memory_items))
            documents = json.loads(next(index_dir.parent.glob("openie_results_ner_*.json")).read_text())["docs"]
            entity_keys = dict(zip(entity_df["content"], entity_df["hash_id"], strict=True))
            schema = None
            if args.schema_root:
                schema_dir = args.schema_root / slug / group.group_id
                schema_complete = json.loads((schema_dir / "complete.json").read_text())
                schema = json.loads((schema_dir / "schema.json").read_text())
                if schema_complete["relations"] != len(schema):
                    raise ValueError("Incomplete relation schema")
            start = perf_counter()
            weights, stats = latest_relation_weights(graph, documents, ordered_keys, entity_keys, text_processing,
                                                      schema=schema, consolidate_multiple=args.consolidate_multiple)
            if args.weight_grid:
                weights = calibrated_weights(graph, weights["canonical_latest"])
            if args.provenance_weights:
                retained, _ = retained_statements(documents, ordered_keys, text_processing, schema,
                                                   consolidate_multiple=True)
                weights = provenance_weights(graph, weights["canonical_latest"], documents, retained,
                                             entity_keys, text_processing)
            if args.adaptive_synonyms:
                retained, _ = retained_statements(documents, ordered_keys, text_processing, schema,
                                                   consolidate_multiple=True)
                weights = adaptive_weights(graph, weights["canonical_latest"], documents, retained,
                                           entity_keys, text_processing, schema)
            construction_seconds = perf_counter() - start
            for variant in variants:
                directory = args.output_root / variant / slug / "memory" / group.group_id
                directory.mkdir(parents=True)
                np.save(directory / "edge_weights.npy", weights[variant])
                write_json(directory / "construction.json", dict(stats, source_graph=str(source_graph),
                    variant=variant, construction_seconds_shared=construction_seconds,
                    edge_order="unchanged source graph edge order", frozen=True))
            # Queries are read only after the two graph variants have been frozen.
            resets = sparse.load_npz(original / "query_resets.npz")
            queries = json.loads((original / "queries.json").read_text())
            if len(queries) != len(group.cases) or resets.shape != (len(queries), graph.vcount()):
                raise ValueError("Saved seed matrix shape mismatch")
            vertex = {key: i for i, key in enumerate(graph.vs["name"])}
            passage_vertices = [vertex[key] for key in passage_keys]
            original_weights = graph.es["weight"][:]
            for i, (case, query) in enumerate(zip(group.cases, queries, strict=True)):
                if case.case_id != query["case_id"] or case.question != query["question"]:
                    raise ValueError("Saved seed case order mismatch")
                reset = resets.getrow(i).toarray().ravel()
                reset = np.where(np.isnan(reset) | (reset < 0), 0, reset)
                if not query["dense_fallback"]:
                    graph.es["weight"] = original_weights
                    original_ppr = graph.personalized_pagerank(vertices=range(graph.vcount()),
                        damping=damping, directed=False, weights="weight", reset=reset, implementation="prpack")
                    original_order = np.argsort(np.asarray(original_ppr)[passage_vertices])[::-1][:5]
                    original_texts = [passage_text[passage_keys[j]] for j in original_order]
                    if original_texts != [r["text"] for r in reference[(group.group_id, case.case_id)]["retrieved"]]:
                        raise ValueError("CPU PPR replay does not reproduce original retrieval")
                for variant in variants:
                    row = dict(reference[(group.group_id, case.case_id)])
                    start = perf_counter()
                    if not query["dense_fallback"]:
                        graph.es["weight"] = weights[variant].tolist()
                        ppr = graph.personalized_pagerank(vertices=range(graph.vcount()),
                            damping=damping, directed=False, weights="weight", reset=reset, implementation="prpack")
                        scores = np.asarray(ppr)[passage_vertices]
                        order = np.argsort(scores)[::-1][:5]
                        row["retrieved"] = [dict(text=passage_text[passage_keys[j]], score=float(scores[j]), metadata={})
                                            for j in order]
                    row.update(retrieval_seconds=perf_counter() - start, retrieval_protocol=variant,
                               shared_recognition_and_embedding=True)
                    streams[variant].write(json.dumps(row, ensure_ascii=False) + "\n")
                    streams[variant].flush()
                total += 1
            print(json.dumps(dict(task=args.task, group=group.group_id, complete=total, construction=stats)), flush=True)
        if total != len(expected):
            raise ValueError("Incomplete replay")
        for variant in variants:
            write_json(args.output_root / variant / slug / "retrieval_complete.json", dict(
                questions=total, original_cpu_replay_matches=total))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--schema-root", type=Path)
    parser.add_argument("--consolidate-multiple", action="store_true")
    parser.add_argument("--weight-grid", action="store_true")
    parser.add_argument("--provenance-weights", action="store_true")
    parser.add_argument("--adaptive-synonyms", action="store_true")
    parser.add_argument("--path", default="/oscar/scratch/zliu328/agent-memory-data/locomo/locomo10.json")
    parser.add_argument("--data-root", default=str(Path(__file__).resolve().parents[1] / "baseline_algorithms/HippoRAG/reproduce/dataset"))
    args = parser.parse_args()
    if args.weight_grid and not (args.consolidate_multiple and args.schema_root):
        parser.error("The weight grid requires canonical consolidation and a source schema")
    if args.provenance_weights and (args.weight_grid or not (args.consolidate_multiple and args.schema_root)):
        parser.error("Provenance weighting requires canonical consolidation, a schema, and no weight grid")
    if args.adaptive_synonyms and (args.provenance_weights or args.weight_grid or not (args.consolidate_multiple and args.schema_root)):
        parser.error("Adaptive synonyms require canonical consolidation, a schema, and no other weight family")
    replay(args)


if __name__ == "__main__":
    main()

"""Materialize the main graph and its fixed-readout original-graph control."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import igraph as ig
import numpy as np
import pandas as pd

from baseline.official import _prepend
from main import _load_groups, build_parser
from optimization.graph_construction.compiled_sources import compile_sources
from optimization.graph_construction.source_consolidation import latest_relation_weights, retained_statements
from optimization.graph_construction.source_window import attach_source_windows
from optimization.graph_construction.statement_incidence import statement_incidence_graph, project_statement_graph
from optimization.retriever.hybrid_graph import RANK_CONSTANT, RANK_WINDOW
from optimization.run_graph import BASE, SOURCE, TASKS, write_json


def build(args):
    _prepend(Path(__file__).resolve().parents[1] / "baseline_algorithms/HippoRAG/src")
    from hipporag.utils.misc_utils import text_processing

    slug = args.task.replace(" ", "_")
    common = build_parser().parse_args([
        "--task", args.task, "--baseline", "bm25", "--output-dir", str(args.output_root),
        "--chunk-size", "512", "--path", args.path, "--data-root", args.data_root])
    groups = list(_load_groups(common))
    incidence = args.construction != "projected"
    variants = ((f"{args.construction}_rrf_window", f"{args.construction}_refined_rrf_window") if incidence else
                ("canonical_latest_rrf_window", "original_graph_rrf_window"))
    if args.retained_fact_index:
        variants = (f"{args.construction}_retained_index_rrf_window",)
    for variant in variants:
        directory = args.output_root / variant / slug
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / "settings.json", dict(task=args.task, variant=variant, seed=42, top_k=5,
            source_memory=str(args.source_root / slug), source_schema=str(args.schema_root / slug),
            construction_reads_questions=False, source_order_is_event_time=False,
            construction=args.construction,
            fact_candidates="retained support" if args.retained_fact_index else "original complete index",
            support_selection=("all extracted support" if variant in
                (f"{args.construction}_rrf_window", "original_graph_rrf_window") else
                "latest source per canonical subject/relation; no role filtering"),
            test_set_used_as_development_set=True, additional_generator_calls=0,
            inherited_extraction_and_schema_costs_excluded=True))
    total = 0
    for group in groups:
        start = perf_counter()
        index_root = args.source_root / slug / "hipporag_indices" / group.group_id
        source_graph, = index_root.glob("*/graph.pickle")
        openie, = index_root.glob("openie_results_ner_*.json")
        graph = ig.Graph.Read_Pickle(str(source_graph))
        passages = pd.read_parquet(source_graph.parent / "chunk_embeddings/vdb_chunk.parquet")
        entities = pd.read_parquet(source_graph.parent / "entity_embeddings/vdb_entity.parquet")
        text_to_key = dict(zip(passages["content"], passages["hash_id"], strict=True))
        if set(text_to_key) != set(group.memory_items):
            raise ValueError("Source store and original loader differ")
        ordered = list(dict.fromkeys(text_to_key[text] for text in group.memory_items))
        timestamps = {}
        if group.memory_timestamps:
            for text, timestamp in zip(group.memory_items, group.memory_timestamps, strict=True):
                timestamps.setdefault(text_to_key[text], timestamp)
        documents = json.loads(openie.read_text())["docs"]
        schema_dir = args.schema_root / slug / group.group_id
        schema = json.loads((schema_dir / "schema.json").read_text())
        if json.loads((schema_dir / "complete.json").read_text())["relations"] != len(schema):
            raise ValueError("Incomplete relation schema")
        entity_keys = dict(zip(entities["content"], entities["hash_id"], strict=True))
        weights, stats = latest_relation_weights(graph, documents, ordered, entity_keys, text_processing, schema)
        retained, _ = retained_statements(documents, ordered, text_processing, schema)
        contents = attach_source_windows(compile_sources(documents, ordered, retained, timestamps, text_processing),
                                         ordered, 3)
        if incidence:
            frozen = json.loads((args.readout_root / "compiled_sources" / slug / group.group_id /
                                 "contents.json").read_text())
            if contents != frozen:
                raise ValueError("Source readout differs from the frozen refinement comparison")
            contents = frozen
        shared = args.output_root / "compiled_sources" / slug / group.group_id
        shared.mkdir(parents=True)
        write_json(shared / "contents.json", contents)
        for variant in variants:
            directory = args.output_root / variant / slug / "memory" / group.group_id
            directory.mkdir(parents=True)
            refined = args.retained_fact_index or variant in ("canonical_latest_rrf_window", f"{args.construction}_refined_rrf_window")
            selected_weights = weights if refined else np.asarray(graph.es["weight"], dtype=np.float64)
            graph_file = None
            if incidence:
                base_graph = graph.copy()
                base_graph.es["weight"] = selected_weights.tolist()
                selected = retained if refined else [(doc["idx"], tuple(text_processing(list(triple))))
                           for doc in documents for triple in doc["extracted_triples"]]
                transformed = statement_incidence_graph(base_graph, documents, selected, entity_keys,
                    text_processing, refined=refined)
                transformed = project_statement_graph(transformed, graph.vcount())
                graph_file = directory / "graph.pickle"
                transformed.write_pickle(str(graph_file))
                selected_weights = np.asarray(transformed.es["weight"], dtype=np.float64)
            np.save(directory / "edge_weights.npy", selected_weights)
            write_json(directory / "lexical_source_keys.json", ordered)
            write_json(directory / "graph.json", dict(source_graph=str(source_graph), frozen=True, variant=variant,
                rank_fusion=dict(rank_constant=RANK_CONSTANT, rank_window=RANK_WINDOW),
                compiled_source_file=str(shared / "contents.json"), edge_order="unchanged source graph edge order"))
            if graph_file is not None:
                metadata_path = directory / "graph.json"
                metadata = json.loads(metadata_path.read_text())
                write_json(metadata_path, dict(metadata, constructed_graph_file=str(graph_file),
                    edge_order="constructed graph edge order", original_vertices=graph.vcount(),
                    vertices=transformed.vcount(), edges=transformed.ecount(),
                    new_node_reset=("zero; original entity and passage seeds unchanged" if
                        transformed.vcount() > graph.vcount() else "no additional vertices")))
            if args.retained_fact_index:
                facts = pd.read_parquet(source_graph.parent / "fact_embeddings/vdb_fact.parquet")
                selected_facts = {str(triple) for _, triple in retained}
                if not selected_facts.issubset(set(facts["content"])):
                    raise ValueError("Retained statements are missing from the original fact index")
                keys = facts.loc[facts["content"].isin(selected_facts), "hash_id"].tolist()
                index_file = directory / "retained_fact_keys.json"
                write_json(index_file, keys)
                metadata = json.loads((directory / "graph.json").read_text())
                write_json(directory / "graph.json", dict(metadata, retained_fact_keys_file=str(index_file)))
        write_json(shared / "construction.json", dict(stats, source_schema=str(schema_dir),
            seconds=perf_counter() - start, frozen=True))
        total += len(ordered)
        print(json.dumps(dict(task=args.task, group=group.group_id, sources=len(ordered))), flush=True)
    for variant in variants:
        write_json(args.output_root / variant / slug / "build_complete.json",
                   dict(groups=[group.group_id for group in groups], sources=total, construction_reads_questions=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--construction", choices=("projected", "statement_projection_loop_free"),
                        default="statement_projection_loop_free")
    parser.add_argument("--retained-fact-index", action="store_true",
                        help="Restrict existing recognition candidates to the refined graph's supported facts")
    parser.add_argument("--readout-root", type=Path,
                        default=BASE / "optimization_canonical_latest_cleanup_seed42_20260914")
    parser.add_argument("--source-root", type=Path, default=SOURCE / "hipporag2")
    parser.add_argument("--schema-root", type=Path, default=BASE / "optimization_canonical_schema_seed42_20260913")
    parser.add_argument("--path", default="/oscar/scratch/zliu328/agent-memory-data/locomo/locomo10.json")
    parser.add_argument("--data-root", default=str(Path(__file__).resolve().parents[1] / "baseline_algorithms/HippoRAG/reproduce/dataset"))
    args = parser.parse_args()
    if args.retained_fact_index and args.construction != "statement_projection_loop_free":
        parser.error("--retained-fact-index requires the refined loop-free graph")
    build(args)


if __name__ == "__main__":
    main()

"""Materialize the main graph and its fixed-readout original-graph control."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import igraph as ig
import numpy as np
import pandas as pd

from baseline.official import _prepend
from experiments.run_anchormem import TASKS
from main import _load_groups, build_parser
from optimization.graph_construction.compiled_sources import compile_sources
from optimization.graph_construction.source_consolidation import latest_relation_weights, retained_statements
from optimization.graph_construction.source_window import attach_source_windows
from optimization.retriever.hybrid_graph import RANK_CONSTANT, RANK_WINDOW
from optimization.run_graph import BASE, SOURCE, write_json


def build(args):
    _prepend(Path(__file__).resolve().parents[1] / "baseline_algorithms/HippoRAG/src")
    from hipporag.utils.misc_utils import text_processing

    slug = args.task.replace(" ", "_")
    common = build_parser().parse_args([
        "--task", args.task, "--baseline", "bm25", "--output-dir", str(args.output_root),
        "--chunk-size", "512", "--path", args.path, "--data-root", args.data_root])
    groups = list(_load_groups(common))
    variants = ("canonical_latest_rrf_window", "original_graph_rrf_window")
    for variant in variants:
        directory = args.output_root / variant / slug
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / "settings.json", dict(task=args.task, variant=variant, seed=42, top_k=5,
            source_memory=str(args.source_root / slug), source_schema=str(args.schema_root / slug),
            construction_reads_questions=False, source_order_is_event_time=False,
            support_selection="latest source per canonical subject/relation; no role filtering",
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
        shared = args.output_root / "compiled_sources" / slug / group.group_id
        shared.mkdir(parents=True)
        write_json(shared / "contents.json", contents)
        for variant in variants:
            directory = args.output_root / variant / slug / "memory" / group.group_id
            directory.mkdir(parents=True)
            np.save(directory / "edge_weights.npy", weights if variant.startswith("canonical") else
                    np.asarray(graph.es["weight"], dtype=np.float64))
            write_json(directory / "lexical_source_keys.json", ordered)
            write_json(directory / "graph.json", dict(source_graph=str(source_graph), frozen=True, variant=variant,
                rank_fusion=dict(rank_constant=RANK_CONSTANT, rank_window=RANK_WINDOW),
                compiled_source_file=str(shared / "contents.json"), edge_order="unchanged source graph edge order"))
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
    parser.add_argument("--source-root", type=Path, default=SOURCE / "hipporag2")
    parser.add_argument("--schema-root", type=Path, default=BASE / "optimization_canonical_schema_seed42_20260913")
    parser.add_argument("--path", default="/oscar/scratch/zliu328/agent-memory-data/locomo/locomo10.json")
    parser.add_argument("--data-root", default=str(Path(__file__).resolve().parents[1] / "baseline_algorithms/HippoRAG/reproduce/dataset"))
    build(parser.parse_args())


if __name__ == "__main__":
    main()

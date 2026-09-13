"""Compile frozen graph-node context and compare original versus consolidated retrieval."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter

import igraph as ig
import numpy as np
import pandas as pd

from baseline.base import RetrievedItem
from experiments.run_anchormem import TASKS
from main import _load_groups, build_parser
from optimization.graph_construction.compiled_sources import VARIANTS, SOURCE_VARIANTS, compile_sources
from optimization.graph_construction.source_consolidation import retained_statements
from optimization.graph_construction.source_window import VARIANTS as WINDOW_VARIANTS, attach_source_windows
from optimization.retriever.compiled_sources import compiled_item
from optimization.run_graph import BASE, write_json


def compile_task(args):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline_algorithms/HippoRAG/src"))
    from hipporag.utils.misc_utils import text_processing

    slug = args.task.replace(" ", "_")
    common = build_parser().parse_args([
        "--task", args.task, "--baseline", "bm25", "--output-dir", str(args.output_root),
        "--chunk-size", "512", "--path", args.path, "--data-root", args.data_root])
    groups = list(_load_groups(common))
    source_dirs = dict(original_graph_compiled=args.seed_root / "original" / slug,
                       canonical_graph_compiled=args.graph_root / "canonical_latest" / slug)
    variants = SOURCE_VARIANTS if args.with_source else VARIANTS
    representation = "original_source_with_retained_triples" if args.with_source else "retained_source_triples"
    if args.with_source:
        source_dirs = dict(canonical_graph_with_source=args.graph_root / "canonical_latest" / slug,
                           adaptive_graph_with_source=args.adaptive_root / "adaptive_syn020" / slug)
    if args.source_window:
        variants = WINDOW_VARIANTS
        representation = "original_source_with_retained_triples_and_window"
        source_dirs = dict(zip(variants, source_dirs.values(), strict=True))
    source_completeness = {variant: json.loads((directory / "retrieval_complete.json").read_text())
                           for variant, directory in source_dirs.items()}
    count = sum(len(group.cases) for group in groups)
    if any(complete["questions"] != count for complete in source_completeness.values()):
        raise ValueError("Graph retrieval sources must cover the full task")
    for variant, complete in source_completeness.items():
        verification = "original_reference_matches" if variant == "original_graph_compiled" else "original_cpu_replay_matches"
        if complete.get(verification) != count:
            raise ValueError(f"Graph retrieval source was not fully verified: {variant}")
    for variant in variants:
        directory = args.output_root / variant / slug
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / "settings.json", dict(
            task=args.task, variant=variant, seed=42, top_k=5, full_questions=count,
            test_set_used_as_development_set=True, context_representation=representation,
            original_text_in_reader_context=args.with_source,
            source_window_size=args.source_window, source_window_boundary="contiguous equal non-null timestamp",
            fixed_original_text_context=False, graph_source=str(source_dirs[variant]),
            source_order_is_event_time=False, construction_reads_questions=False,
            additional_generator_calls=0, inherited_extraction_cost_not_zero=True,
            note="Graph and readout control; unchanged source-node ranking, QA prompt, decoding and answer scoring. "
                 "Text-based gold-passage recall on compiled strings is not source-level retrieval recall."))
    all_contents = {}
    for group in groups:
        start = perf_counter()
        control = args.seed_root / "original" / slug / "memory" / group.group_id
        source_graph = Path(json.loads((control / "construction.json").read_text())["source_graph"])
        passages = pd.read_parquet(source_graph.parent / "chunk_embeddings/vdb_chunk.parquet")
        text_to_key = dict(zip(passages["content"], passages["hash_id"], strict=True))
        if set(text_to_key) != set(group.memory_items):
            raise ValueError("Source store and original loader differ")
        ordered = list(dict.fromkeys(text_to_key[text] for text in group.memory_items))
        timestamps = {}
        if group.memory_timestamps:
            for text, timestamp in zip(group.memory_items, group.memory_timestamps, strict=True):
                timestamps.setdefault(text_to_key[text], timestamp)
        documents = json.loads(next(source_graph.parent.parent.glob("openie_results_ner_*.json")).read_text())["docs"]
        schema_dir = args.schema_root / slug / group.group_id
        schema = json.loads((schema_dir / "schema.json").read_text())
        if json.loads((schema_dir / "complete.json").read_text())["relations"] != len(schema):
            raise ValueError("Incomplete relation schema")
        retained, stats = retained_statements(documents, ordered, text_processing, schema, consolidate_multiple=True)
        contents = compile_sources(documents, ordered, retained, timestamps, text_processing,
                                   with_source=args.with_source)
        if args.source_window:
            contents = attach_source_windows(contents, ordered, args.source_window)
        all_contents[group.group_id] = (text_to_key, contents)
        shared = args.output_root / "compiled_sources" / slug / group.group_id
        shared.mkdir(parents=True)
        write_json(shared / "contents.json", contents)
        write_json(shared / "construction.json", dict(stats, seconds=perf_counter() - start,
            source_schema=str(schema_dir), frozen=True, contexts=len(contents),
            facts=sum(item["facts"] for item in contents.values()),
            source_characters=sum(len(item["original_source_text"]) for item in contents.values()),
            compiled_characters=sum(len(item["text"]) for item in contents.values())))
        graph = ig.Graph.Read_Pickle(str(source_graph))
        for variant in variants:
            destination = args.output_root / variant / slug / "memory" / group.group_id
            destination.mkdir(parents=True)
            weights = (np.asarray(graph.es["weight"], dtype=np.float64) if variant == "original_graph_compiled"
                       else np.load(source_dirs[variant] / "memory" / group.group_id / "edge_weights.npy"))
            np.save(destination / "edge_weights.npy", weights)
            write_json(destination / "graph.json", dict(source_graph=str(source_graph), frozen=True,
                variant=variant, compiled_source_file=str(shared / "contents.json"),
                edge_order="unchanged source graph edge order"))
    # Source content for every group is frozen before reading any saved queries.
    expected = {(group.group_id, case.case_id) for group in groups for case in group.cases}
    for variant, source_dir in source_dirs.items():
        directory = args.output_root / variant / slug
        seen = set()
        with (directory / "retrieval.jsonl").open("x") as stream:
            for line in (source_dir / "retrieval.jsonl").open():
                row = json.loads(line)
                key = (row["group_id"], row["case"]["case_id"])
                if key not in expected or key in seen:
                    raise ValueError("Unexpected or repeated retrieval case")
                text_to_key, contents = all_contents[row["group_id"]]
                converted = []
                for item in row["retrieved"]:
                    source_key = text_to_key[item["text"]]
                    converted.append(asdict(compiled_item(RetrievedItem(**item), source_key, contents[source_key])))
                row.update(retrieved=converted, retrieval_protocol=variant,
                           context_representation=representation, source_ranking_unchanged=True)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                seen.add(key)
        if seen != expected:
            raise ValueError("Incomplete compiled-context retrieval")
        write_json(directory / "retrieval_complete.json", dict(questions=len(seen), source_rankings_preserved=True))
    print(json.dumps(dict(task=args.task, questions=count, variants=variants)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed-root", type=Path, default=BASE / "optimization_context_graph_seed42_20260913")
    parser.add_argument("--graph-root", type=Path, default=BASE / "optimization_canonical_latest_seed42_20260913")
    parser.add_argument("--schema-root", type=Path, default=BASE / "optimization_canonical_schema_seed42_20260913")
    parser.add_argument("--adaptive-root", type=Path, default=BASE / "optimization_adaptive_synonyms_seed42_20260913")
    parser.add_argument("--with-source", action="store_true", help="Retain original text alongside frozen graph facts")
    parser.add_argument("--source-window", type=int, default=0,
                        help="Add this many neighboring sources on either side within a timestamp group")
    parser.add_argument("--path", default="/oscar/scratch/zliu328/agent-memory-data/locomo/locomo10.json")
    parser.add_argument("--data-root", default=str(Path(__file__).resolve().parents[1] / "baseline_algorithms/HippoRAG/reproduce/dataset"))
    args = parser.parse_args()
    if args.source_window < 0 or (args.source_window and not args.with_source):
        parser.error("--source-window must be nonnegative and requires --with-source")
    compile_task(args)


if __name__ == "__main__":
    main()

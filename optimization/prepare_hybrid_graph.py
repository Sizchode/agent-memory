"""Cross original/consolidated graphs with fixed lexical fusion and source readouts."""

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import igraph as ig
import numpy as np
import pandas as pd

from baseline.base import RetrievedItem
from baseline.bm25 import BM25Baseline
from experiments.run_anchormem import TASKS
from main import _load_groups, build_parser
from optimization.retriever.compiled_sources import compiled_item
from optimization.retriever.hybrid_graph import VARIANTS, RANK_CONSTANT, RANK_WINDOW, fuse_rankings
from optimization.run_graph import BASE, write_json


def prepare(args):
    slug = args.task.replace(" ", "_")
    common = build_parser().parse_args([
        "--task", args.task, "--baseline", "bm25", "--output-dir", str(args.output_root),
        "--chunk-size", "512", "--path", args.path, "--data-root", args.data_root])
    groups = list(_load_groups(common))
    expected = {(group.group_id, case.case_id) for group in groups for case in group.cases}
    sources = dict(original=args.seed_root / "original" / slug,
                   canonical=args.canonical_root / "canonical_latest" / slug)
    for name, directory in sources.items():
        complete = json.loads((directory / "retrieval_complete.json").read_text())
        checked = "original_reference_matches" if name == "original" else "original_cpu_replay_matches"
        if complete["questions"] != len(expected) or complete.get(checked) != len(expected):
            raise ValueError("Source graph retrieval must be fully verified")
    indices, representations = {}, {}
    fusion = dict(rank_constant=RANK_CONSTANT, rank_window=RANK_WINDOW)
    for variant in VARIANTS:
        directory = args.output_root / variant / slug
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / "settings.json", dict(
            task=args.task, variant=variant, seed=42, top_k=5, full_questions=len(expected),
            rank_fusion=fusion, bm25=dict(k1=1.5, b=0.75, source="baseline.bm25.BM25Baseline"),
            graph_rankings=str(sources["canonical" if variant.startswith("canonical") else "original"]),
            test_set_used_as_development_set=True, construction_reads_questions=False,
            retrieval_changed=True, pure_graph_ablation=False, additional_query_llm_calls=0,
            historical_construction_and_recognition_costs_excluded_from_incremental_time=True,
            readout="source_with_retained_facts_and_timestamp_window" if variant.endswith("window") else "original_text",
            note="Crossed control: graph choice, reciprocal-rank fusion, and readout are separate factors. "
                 "RRF uses each ranker's top five, not the original paper's full rankings."))
    for group in groups:
        start = perf_counter()
        source_metadata = json.loads((sources["original"] / "memory" / group.group_id / "construction.json").read_text())
        source_graph = Path(source_metadata["source_graph"])
        passages = pd.read_parquet(source_graph.parent / "chunk_embeddings/vdb_chunk.parquet")
        text_to_key = dict(zip(passages["content"], passages["hash_id"], strict=True))
        if set(text_to_key) != set(group.memory_items):
            raise ValueError("Original source store differs from loader")
        ordered = list(dict.fromkeys(text_to_key[text] for text in group.memory_items))
        lexical = BM25Baseline()
        lexical.build([text for text in dict.fromkeys(group.memory_items)])
        indices[group.group_id] = lexical
        window_metadata = json.loads((args.window_root / "canonical_graph_source_window" / slug /
                                      "memory" / group.group_id / "graph.json").read_text())
        contents_path = Path(window_metadata["compiled_source_file"])
        contents = json.loads(contents_path.read_text())
        if set(contents) != set(ordered):
            raise ValueError("Window representation has a different source corpus")
        for text, key in text_to_key.items():
            if contents[key]["original_source_text"] != text:
                raise ValueError("Window representation changed source identity")
        representations[group.group_id] = (text_to_key, contents)
        original_weights = np.asarray(ig.Graph.Read_Pickle(str(source_graph)).es["weight"], dtype=np.float64)
        canonical_weights = np.load(sources["canonical"] / "memory" / group.group_id / "edge_weights.npy")
        for variant in VARIANTS:
            directory = args.output_root / variant / slug / "memory" / group.group_id
            directory.mkdir(parents=True)
            np.save(directory / "edge_weights.npy", canonical_weights if variant.startswith("canonical") else original_weights)
            write_json(directory / "lexical_source_keys.json", ordered)
            metadata = dict(source_graph=str(source_graph), frozen=True, variant=variant, rank_fusion=fusion,
                            construction_seconds_shared=perf_counter() - start,
                            edge_order="unchanged source graph edge order")
            if variant.endswith("window"):
                metadata["compiled_source_file"] = str(contents_path)
            write_json(directory / "graph.json", metadata)
    # Both source indices and all readout references are frozen before reading queries.
    records = {}
    for name, directory in sources.items():
        with (directory / "retrieval.jsonl").open() as stream:
            rows = [json.loads(line) for line in stream]
        records[name] = {(row["group_id"], row["case"]["case_id"]): row for row in rows}
        if len(rows) != len(expected) or set(records[name]) != expected:
            raise ValueError("Source retrieval case coverage differs")
    with ExitStack() as stack:
        streams = {variant: stack.enter_context((args.output_root / variant / slug / "retrieval.jsonl").open("x"))
                   for variant in VARIANTS}
        total = 0
        for group in groups:
            text_to_key, contents = representations[group.group_id]
            for case in group.cases:
                key = (group.group_id, case.case_id)
                start = perf_counter()
                lexical_items = indices[group.group_id].retrieve(case.question, RANK_WINDOW)
                lexical_seconds = perf_counter() - start
                for name in ("original", "canonical"):
                    row = records[name][key]
                    if row["case"]["question"] != case.question:
                        raise ValueError("Source query does not match loader")
                    start = perf_counter()
                    items = fuse_rankings([RetrievedItem(**item) for item in row["retrieved"]], lexical_items, 5)
                    elapsed = lexical_seconds + perf_counter() - start
                    for suffix in ("", "_window"):
                        variant = f"{name}_graph_rrf{suffix}"
                        selected = items if not suffix else [compiled_item(item, text_to_key[item.text],
                                                                          contents[text_to_key[item.text]]) for item in items]
                        output = dict(row, retrieved=[asdict(item) for item in selected], retrieval_protocol=variant,
                                      retrieval_seconds=elapsed, incremental_timing_only=True,
                                      graph_retrieval_seconds=row.get("retrieval_seconds"),
                                      additional_query_llm_calls=0)
                        streams[variant].write(json.dumps(output, ensure_ascii=False) + "\n")
                total += 1
            print(json.dumps(dict(task=args.task, group=group.group_id, complete=total)), flush=True)
    for variant in VARIANTS:
        write_json(args.output_root / variant / slug / "retrieval_complete.json",
                   dict(questions=total, source_graph_retrieval_verified=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed-root", type=Path, default=BASE / "optimization_context_graph_seed42_20260913")
    parser.add_argument("--canonical-root", type=Path, default=BASE / "optimization_canonical_latest_seed42_20260913")
    parser.add_argument("--window-root", type=Path, default=BASE / "optimization_source_window_seed42_20260913")
    parser.add_argument("--path", default="/oscar/scratch/zliu328/agent-memory-data/locomo/locomo10.json")
    parser.add_argument("--data-root", default=str(Path(__file__).resolve().parents[1] / "baseline_algorithms/HippoRAG/reproduce/dataset"))
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()

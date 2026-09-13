"""Prepare source-preserving packing controls on a fixed consolidated hybrid graph."""

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from baseline.base import RetrievedItem
from experiments.run_anchormem import TASKS
from main import _load_groups, build_parser
from optimization.graph_construction.compiled_sources import compile_sources
from optimization.graph_construction.source_consolidation import retained_statements
from optimization.graph_construction.source_window import attach_source_windows
from optimization.retriever.compiled_sources import PACKING_VARIANTS as VARIANTS, pack_source_items
from optimization.run_graph import BASE, write_json


def prepare(args):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline_algorithms/HippoRAG/src"))
    from hipporag.utils.misc_utils import text_processing
    slug = args.task.replace(" ", "_")
    source = args.source_root / "canonical_graph_rrf" / slug
    complete = json.loads((source / "retrieval_complete.json").read_text())
    common = build_parser().parse_args([
        "--task", args.task, "--baseline", "bm25", "--output-dir", str(args.output_root),
        "--chunk-size", "512", "--path", args.path, "--data-root", args.data_root])
    groups = list(_load_groups(common))
    expected = {(group.group_id, case.case_id) for group in groups for case in group.cases}
    if complete["questions"] != len(expected) or not complete["source_graph_retrieval_verified"]:
        raise ValueError("Packing requires the complete verified hybrid retrieval")
    for variant, format_name in VARIANTS.items():
        directory = args.output_root / variant / slug
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / "settings.json", dict(
            task=args.task, variant=variant, seed=42, top_k=5, full_questions=len(expected),
            source_retrieval=str(source), graph_and_retrieval_unchanged=True,
            test_set_used_as_development_set=True, construction_reads_questions=False,
            source_window_size=3, deduplicate_source_windows=True, facts_format=format_name,
            preserve_all_unique_retrieved_source_records=True, additional_llm_calls=0,
            note="Readout-only control. Sentence facts concatenate original triple fields without rewriting them. "
                 "Not equal-token and not a graph-topology improvement."))
    banks = {}
    for group in groups:
        start = perf_counter()
        original = source / "memory" / group.group_id
        metadata = json.loads((original / "graph.json").read_text())
        source_graph = Path(metadata["source_graph"])
        documents = json.loads(next(source_graph.parent.parent.glob("openie_results_ner_*.json")).read_text())["docs"]
        text_to_key = {doc["passage"]: doc["idx"] for doc in documents}
        if set(text_to_key) != set(group.memory_items):
            raise ValueError("Original loader and graph sources disagree")
        ordered = list(dict.fromkeys(text_to_key[text] for text in group.memory_items))
        if ordered != json.loads((original / "lexical_source_keys.json").read_text()):
            raise ValueError("Lexical source order differs")
        timestamps = {}
        if group.memory_timestamps:
            for text, timestamp in zip(group.memory_items, group.memory_timestamps, strict=True):
                timestamps.setdefault(text_to_key[text], timestamp)
        schema_dir = args.schema_root / slug / group.group_id
        schema = json.loads((schema_dir / "schema.json").read_text())
        if json.loads((schema_dir / "complete.json").read_text())["relations"] != len(schema):
            raise ValueError("Incomplete relation schema")
        retained, _ = retained_statements(documents, ordered, text_processing, schema, consolidate_multiple=True)
        banks[group.group_id] = {}
        for variant, format_name in VARIANTS.items():
            contents = compile_sources(documents, ordered, retained, timestamps, text_processing,
                                       with_source=True, facts_format=format_name)
            contents = attach_source_windows(contents, ordered, 3)
            directory = args.output_root / variant / slug / "memory" / group.group_id
            directory.mkdir(parents=True)
            write_json(directory / "contents.json", contents)
            write_json(directory / "lexical_source_keys.json", ordered)
            np.save(directory / "edge_weights.npy", np.load(original / "edge_weights.npy"))
            write_json(directory / "graph.json", dict(metadata, variant=variant, pack_source_windows=True,
                compiled_source_file=str(directory / "contents.json"), construction_seconds_shared=perf_counter() - start))
            banks[group.group_id][variant] = (text_to_key, contents)
    with ExitStack() as stack:
        streams = {variant: stack.enter_context((args.output_root / variant / slug / "retrieval.jsonl").open("x"))
                   for variant in VARIANTS}
        seen = set()
        with (source / "retrieval.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                key = (row["group_id"], row["case"]["case_id"])
                if key not in expected or key in seen:
                    raise ValueError("Packing source has missing or repeated cases")
                items = [RetrievedItem(**item) for item in row["retrieved"]]
                for variant in VARIANTS:
                    text_to_key, contents = banks[row["group_id"]][variant]
                    start = perf_counter()
                    packed = pack_source_items(items, text_to_key, contents)
                    output = dict(row, retrieved=[asdict(item) for item in packed], retrieval_protocol=variant,
                                  context_packing_seconds=perf_counter() - start)
                    streams[variant].write(json.dumps(output, ensure_ascii=False) + "\n")
                seen.add(key)
    if seen != expected:
        raise ValueError("Incomplete packed retrieval")
    for variant in VARIANTS:
        write_json(args.output_root / variant / slug / "retrieval_complete.json", dict(
            questions=len(seen), source_rankings_preserved=True, all_unique_context_sources_preserved=True))
    print(json.dumps(dict(task=args.task, questions=len(seen), variants=list(VARIANTS))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=BASE / "optimization_hybrid_graph_seed42_20260913")
    parser.add_argument("--schema-root", type=Path, default=BASE / "optimization_canonical_schema_seed42_20260913")
    parser.add_argument("--path", default="/oscar/scratch/zliu328/agent-memory-data/locomo/locomo10.json")
    parser.add_argument("--data-root", default=str(Path(__file__).resolve().parents[1] / "baseline_algorithms/HippoRAG/reproduce/dataset"))
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()

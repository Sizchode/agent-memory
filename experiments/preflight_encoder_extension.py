"""Read-only source and storage preflight for the four approved missing tasks."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


TASKS = ("SH-Doc_QA", "MH-Doc_QA", "FactConsolidation-SH", "FactConsolidation-MH")
READERS = ("Qwen/Qwen3.5-4B", "meta-llama/Llama-3.1-8B-Instruct")
BASE = Path("/oscar/scratch/zliu328/agent-memory-outputs")


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def emit(**record):
    print(json.dumps(record), flush=True)


def main(results, graph, queries):
    with (results / "qa_results.csv").open() as stream:
        manifest = list(csv.DictReader(stream))
    protocol = json.loads((results / "protocol.json").read_text())
    total_groups = 0
    for task in TASKS:
        question_identity = None
        for model in READERS:
            for variant in ("full", "without_propagation"):
                matches = [r for r in manifest if r["task"] == task and r["model"] == model
                           and r["setting"] == "one_shot" and r["variant"] == variant]
                if len(matches) != 1:
                    raise ValueError(f"Nonunique source: {task}/{model}/{variant}")
                source = Path(matches[0]["source"]) / "retrieval.jsonl"
                with source.open() as stream:
                    rows = [json.loads(line) for line in stream if line.strip()]
                identities = [(r["group_id"], r["case"]["case_id"], r["case"]["question"])
                              for r in rows]
                if len(rows) != protocol["tasks"][task][0] or len(rows) != 100:
                    raise ValueError(f"Unexpected question count: {task}")
                if len(set(identities)) != len(rows):
                    raise ValueError(f"Repeated question: {task}")
                if question_identity is None:
                    question_identity = identities
                if identities != question_identity:
                    raise ValueError(f"Question identity/order mismatch: {task}/{model}/{variant}")
                emit(task=task, model=model, variant=variant, questions=len(rows),
                     source=str(source), sha256=digest(source))
        groups = Counter(group for group, _, _ in question_identity)
        for group, count in groups.items():
            memory = graph / task / "memory" / group
            metadata_paths = [p for p in (memory / "graph.json", memory / "construction.json")
                              if p.is_file()]
            if len(metadata_paths) != 1:
                raise ValueError(f"Nonunique memory metadata: {task}/{group}")
            metadata = json.loads(metadata_paths[0].read_text())
            original = Path(metadata["source_graph"])
            selected = (Path(protocol["graph_source"]) / "groups" / task / group
                        / (protocol["graph_variant"] + ".pickle"))
            if not selected.is_file() or not original.is_file():
                raise FileNotFoundError(f"Missing graph: {task}/{group}")
            query = queries / task / group / "queries.npz"
            with np.load(query, allow_pickle=False) as vectors:
                expected = {q for g, _, q in question_identity if g == group}
                if set(vectors["queries"].tolist()) != expected:
                    raise ValueError(f"Query cache identity mismatch: {task}/{group}")
                for kind in ("triple", "passage"):
                    if len(vectors[kind]) != len(expected) or not np.isfinite(vectors[kind]).all():
                        raise ValueError(f"Invalid query vectors: {task}/{group}/{kind}")
            original_store = original.parent.parent
            sizes = {str(p): p.stat().st_size for p in original_store.rglob("*")
                     if p.is_file() and p.suffix in (".parquet", ".npy", ".npz")}
            facts = json.loads(Path(metadata["retained_fact_keys_file"]).read_text())
            if len(set(facts)) != len(facts):
                raise ValueError(f"Repeated retained fact: {task}/{group}")
            emit(task=task, group=group, questions=count, retained_facts=len(facts),
                 source_store_bytes=sum(sizes.values()), source_store_files=sizes,
                 graph_sha256=digest(selected), queries_sha256=digest(query))
        total_groups += len(groups)
    emit(preflight_complete=True, tasks=TASKS, readers=READERS, questions=400,
         groups=total_groups)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path,
                        default=BASE / "optimization_simplified_amor_seed42_20260929")
    parser.add_argument("--graph", type=Path, default=BASE /
                        "optimization_retained_fact_index_seed42_20260914/statement_projection_loop_free_retained_index_rrf_window")
    parser.add_argument("--queries", type=Path,
                        default=BASE / "optimization_query_embeddings_seed42_20260919")
    args = parser.parse_args()
    main(args.results, args.graph, args.queries)

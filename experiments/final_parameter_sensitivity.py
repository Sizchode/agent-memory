"""One-factor sensitivity on the final AMOR graph, with fresh native QA."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import random

import numpy as np

from experiments.encoder_robustness import DEFAULT_RESULTS, RecognitionAudit, digest, json_save, source_rows


SETTINGS = {
    "reference": (0.50, 0.05),
    "alpha_025": (0.25, 0.05),
    "alpha_075": (0.75, 0.05),
    "lambda_001": (0.50, 0.01),
    "lambda_010": (0.50, 0.10),
}
TASKS = ("SH-Doc_QA", "MH-Doc_QA", "FactConsolidation-SH", "FactConsolidation-MH", "LoCoMo", "2WikiMultiHopQA")
READERS = ("Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B", "google/gemma-3-4b-it", "meta-llama/Llama-3.1-8B-Instruct")


def validate_settings(settings):
    reference = settings["reference"]
    for name, pair in settings.items():
        changed = sum(x != y for x, y in zip(pair, reference, strict=True))
        if changed != (name != "reference"):
            raise ValueError(f"Not a one-factor intervention: {name}")


def initialize(args):
    validate_settings(SETTINGS)
    source = json.loads((args.results / "protocol.json").read_text())
    if source["graph_variant"] != "without_projected_connections":
        raise ValueError("Expected the final membership-only graph with fact-derived weights")
    json_save(args.output / "protocol.json", dict(
        experiment="final_graph_one_factor_parameter_sensitivity", settings=SETTINGS,
        tasks=TASKS, readers=READERS, source_results=str(args.results),
        source_protocol_sha256=digest(args.results / "protocol.json"),
        graph_source=source["graph_source"], graph_variant=source["graph_variant"],
        source_k=5, fusion_window=5, seed=42, fresh_qa_reference=True,
        fixed="final graph, facts, recognized facts, embeddings, source texts, BM25, RRF and augmentation policy",
        recognition="native historical cache replay, including separately audited native parse failures",
        reporting="all tasks, readers and predefined settings; no per-task selection",
        qa_requests=67720, test_as_dev=True, code_sha256=digest(__file__)))


def prepare(args):
    from experiments.analyze_geometry import GRAPH, QUERIES
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
    source_protocol = json.loads((args.results / "protocol.json").read_text())
    for group in dict.fromkeys(row.group_id for row in expected):
        rows = [row for row in expected if row.group_id == group]
        directory = args.output / "groups" / args.task / group
        directory.mkdir(parents=True, exist_ok=True)
        marker = directory / "complete.json"
        if marker.exists():
            complete = json.loads(marker.read_text())
            for setting in SETTINGS:
                if digest(directory / setting / "retrieval.jsonl") != complete["retrieval_sha256"][setting]:
                    raise ValueError("Completed retrieval changed")
            continue
        runtime = args.output / "runtime" / args.task / group
        memory, cache_guard = load_frozen(args.task, group, GRAPH, runtime, args.results)
        engine = memory._memory
        control = PropagationControl(engine)
        audit = RecognitionAudit(engine.rerank_filter, directory / "recognition.jsonl", policy="historical_replay")
        try:
            selector = QueryFactContext(engine, memory.contents)
            graph_path = Path(source_protocol["graph_source"]) / "groups" / args.task / group / (source_protocol["graph_variant"] + ".pickle")
            graph_edges = engine.graph.get_edgelist()
            graph_weights = np.asarray(engine.graph.es["weight"]).copy()
            collected = {setting: [] for setting in SETTINGS}
            for index, row in enumerate(rows):
                default_reset = None
                for setting, (alpha, weight) in SETTINGS.items():
                    engine.global_config.damping = alpha
                    engine.global_config.passage_node_weight = weight
                    memory.base.rank_window = 5
                    control.begin()
                    centers = memory.retrieve(row.case.question, 5)
                    rendered = selector.render(row.case.question, centers)
                    cache_guard.check()
                    audit.check()
                    if setting == "reference":
                        if [item.text for item in rendered] != [item.text for item in row.retrieved]:
                            raise ValueError(f"Final-graph reference replay differs: {row.case.case_id}")
                        default_reset = None if control.reset is None else control.reset.copy()
                    elif weight == SETTINGS["reference"][1]:
                        if default_reset is None:
                            if control.reset is not None:
                                raise ValueError("Propagation eligibility changed during alpha-only intervention")
                        else:
                            np.testing.assert_array_equal(default_reset, control.reset)
                    collected[setting].append(replace(row, retrieved=rendered, top_k=5))
                if index % 50 == 0:
                    print("prepared", args.task, group, index, len(rows), flush=True)
            if graph_edges != engine.graph.get_edgelist():
                raise ValueError("Graph connectivity changed")
            np.testing.assert_array_equal(graph_weights, engine.graph.es["weight"])
            hashes = {}
            for setting, selected in collected.items():
                path = directory / setting / "retrieval.jsonl"
                save_rows(path, selected)
                hashes[setting] = digest(path)
            json_save(directory / "recognition_audit.json", dict(policy=audit.policy, responses=audit.responses))
            json_save(marker, dict(complete=True, questions=len(rows), final_reference_exact=True,
                graph_sha256=digest(graph_path), query_embeddings_sha256=digest(QUERIES / args.task / group / "queries.npz"),
                source_retrieval_sha256=digest(sources["full"] / "retrieval.jsonl"),
                graph_unchanged=True, recognition_cache_only=True, retrieval_sha256=hashes))
        finally:
            audit.close()
            control.close()
            memory.close()
            memory = engine = selector = control = None
            release_accelerator_memory()
    for setting in SETTINGS:
        collected = []
        for group in dict.fromkeys(row.group_id for row in expected):
            collected.extend(_read_retrieval_records(args.output / "groups" / args.task / group / setting / "retrieval.jsonl"))
        by_key = {(row.group_id, row.case.case_id): row for row in collected}
        expected_keys = [(row.group_id, row.case.case_id) for row in expected]
        if len(by_key) != len(collected) or set(by_key) != set(expected_keys):
            raise ValueError("Incomplete or duplicated retrieval")
        destination = args.output / "inputs" / setting / args.task / "retrieval.jsonl"
        save_rows(destination, [by_key[key] for key in expected_keys])
    json_save(args.output / "inputs" / (args.task + ".json"), dict(complete=True, questions=len(expected), settings=SETTINGS))


def qa(args):
    from optimization.ircot import Reader, evaluate_task, BATCH_SIZE
    from experiments.runner import _read_retrieval_records, _answer_prompt, _official_generation
    from experiments.ablate_components import save_rows

    for task in TASKS:
        if not json.loads((args.output / "inputs" / (task + ".json")).read_text())["complete"]:
            raise ValueError("Retrieval preparation incomplete")
    reader = Reader(args.model)
    try:
        for setting in SETTINGS:
            for task in TASKS:
                directory = args.output / ("qa_smoke" if args.smoke else "qa") / args.model.replace("/", "_") / setting / task
                if (directory / "qa_complete.json").exists():
                    continue
                records = list(_read_retrieval_records(args.output / "inputs" / setting / task / "retrieval.jsonl"))
                if args.smoke:
                    records = [next(row for row in records if row.case.category == category)
                        for category in dict.fromkeys(row.case.category for row in records)]
                directory.mkdir(parents=True, exist_ok=True)
                save_rows(directory / "retrieval.jsonl", records)
                json_save(directory / "reader.json", reader.metadata)
                rng = random.Random(42)
                requests = [dict(phase="answer", messages=_answer_prompt(row.case, row.retrieved, rng)[0],
                    generation=_official_generation(row.case)) for row in records]
                for start in range(0, len(requests), BATCH_SIZE):
                    result = reader.request(phase="check_context", items=requests[start:start+BATCH_SIZE])
                    if result["overflow"]:
                        raise ValueError(f"Context overflow: {task}/{setting}; refusing truncation")
                evaluate_task(directory, task, reader, args.model, args.smoke)
                marker = json.loads((directory / "qa_complete.json").read_text())
                json_save(directory / "verified.json", dict(complete=True, model=args.model, task=task,
                    setting=setting, pilot=args.smoke, questions=len(records), score=marker["score"],
                    retrieval_sha256=digest(directory / "retrieval.jsonl"),
                    predictions_sha256=digest(directory / "evaluations" / args.model.replace("/", "_") / "predictions.jsonl")))
                print("QA complete", args.model, setting, task, marker["score"], flush=True)
    finally:
        reader.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "qa"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--model", choices=READERS)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare" and args.task is None:
        parser.error("prepare requires a task")
    if args.phase == "qa" and args.model is None:
        parser.error("qa requires a model")
    {"prepare": prepare, "qa": qa}[args.phase](args)


if __name__ == "__main__":
    main()

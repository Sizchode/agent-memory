"""Prepare frozen source contexts and score published weighting controls."""

import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from baseline.base import RetrievedItem
from experiments.analyze_geometry import GRAPH, QUERIES
from experiments.ablate_components import save_rows
from experiments.mine_cases import csv_rows, read_lines, write_json
from experiments.report_paper_costs import TASKS, COUNTS
from experiments.runner import _read_retrieval_records
from optimization.run_graph import retrieval_config
from optimization.retriever.hipporag import load_optimized_memory, load_query_embeddings, CacheMissGuard
from optimization.retriever.compiled_sources import compiled_item
from optimization.retriever.query_fact_context import QueryFactContext


CONDITIONS = ("unit_released", "amor_released")


def prepare(results):
    root = results / "analysis/fact_weighting"
    assert json.loads((root / "complete.json").read_text())["complete"]
    entries = csv_rows(results / "qa_results.csv")
    original = csv_rows(Path("/oscar/scratch/zliu328/agent-memory-outputs/optimization_paper_ablation_seed42_20260927/connection_controls/qa_results.csv"))
    checks = []
    for task in TASKS:
        outputs = {c: root / "inputs" / c / task / "retrieval.jsonl" for c in CONDITIONS}
        if all(p.exists() for p in outputs.values()) and (root / "inputs" / task / "complete.json").exists():
            checks.append(json.loads((root / "inputs" / task / "complete.json").read_text()))
            continue
        entry, = [r for r in entries if r["task"] == task and r["setting"] == "one_shot"
                  and r["variant"] == "full" and r["model"] == "Qwen/Qwen3.5-4B"]
        rows = list(_read_retrieval_records(Path(entry["source"]) / "retrieval.jsonl"))
        unit_entry, = [r for r in original if r["task"] == task and r["variant"] == "without_projection"
                       and r["model"] == "Qwen/Qwen3.5-4B"]
        unit_rows = {r.case.case_id: r for r in _read_retrieval_records(Path(unit_entry["source"]) / "retrieval.jsonl")}
        selections = {r["case_id"]: r for r in read_lines(root / (task + ".jsonl"))}
        assert len(rows) == len(selections) == len(unit_rows) == COUNTS[task]
        groups = defaultdict(list)
        for row in rows:
            groups[row.group_id].append(row)
        rendered = {c: {} for c in CONDITIONS}
        for group, group_rows in groups.items():
            with TemporaryDirectory(prefix="weighting-context-") as temporary:
                runtime = Path(temporary)
                config, _ = retrieval_config(SimpleNamespace(task=task.replace("_", " "), output_root=runtime,
                    generator_base_url="http://127.0.0.1:1/v1",
                    path="/oscar/scratch/zliu328/agent-memory-data/locomo/locomo10.json",
                    data_root=str(Path.cwd() / "baseline_algorithms/HippoRAG/reproduce/dataset")))
                memory = load_optimized_memory(config, GRAPH / task / "memory" / group, runtime)
                try:
                    guard = CacheMissGuard(memory._generator)
                    hippo = memory._memory
                    load_query_embeddings(hippo, QUERIES / task / group / "queries.npz")
                    def reject(*args, **kwargs):
                        raise RuntimeError("Weighting contexts require frozen embeddings")
                    hippo.embedding_model.batch_encode = reject
                    selector = QueryFactContext(hippo, memory.contents)
                    def context(query, keys):
                        items = [compiled_item(RetrievedItem(memory.contents[key]["original_source_text"],
                                 None, {"source_passage": key}), key, memory.contents[key]) for key in keys]
                        return selector.render(query, items)
                    for row in group_rows:
                        selected = selections[row.case.case_id]["selected"]
                        for control, expected in [("amor", row), ("unit", unit_rows[row.case.case_id])]:
                            actual = context(row.case.question, selected[control])
                            assert [i.text for i in actual] == [i.text for i in expected.retrieved], (task, row.case.case_id, control)
                        for condition in CONDITIONS:
                            rendered[condition][row.case.case_id] = replace(row, retrieved=context(row.case.question, selected[condition]))
                    guard.check()
                finally:
                    memory.close()
        for condition in CONDITIONS:
            save_rows(outputs[condition], [rendered[condition][row.case.case_id] for row in rows])
        check = dict(task=task, questions=len(rows), exact_amor_context=True, exact_existing_unit_context=True,
                     new_extraction=False, new_embeddings=False, conditions=CONDITIONS)
        path = root / "inputs" / task / "complete.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, check)
        checks.append(check)
        print("Prepared", task, len(rows), flush=True)
    write_json(root / "inputs/complete.json", dict(complete=True, tasks=checks, questions=sum(COUNTS.values())))


def qa(results, model):
    from optimization.ircot import Reader, MODELS, evaluate_task
    root = results / "analysis/fact_weighting"
    assert json.loads((root / "inputs/complete.json").read_text())["complete"]
    reader = Reader(model)
    try:
        output = root / "qa" / model.replace("/", "_")
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "reader.json", reader.metadata)
        for task in TASKS:
            for condition in CONDITIONS:
                source = root / "inputs" / condition / task / "retrieval.jsonl"
                directory = output / condition / task
                directory.mkdir(parents=True, exist_ok=True)
                target = directory / "retrieval.jsonl"
                if not target.exists():
                    target.symlink_to(source)
                assert target.resolve() == source
                done = directory / "qa_complete.json"
                if done.exists():
                    status = json.loads(done.read_text())
                    assert status["complete"] and status["questions"] == COUNTS[task] and not status["pilot"]
                    continue
                pilot = directory / "pilot"
                pilot.mkdir(exist_ok=True)
                rows = list(_read_retrieval_records(source))
                assert len(rows) == COUNTS[task]
                save_rows(pilot / "retrieval.jsonl", rows[:1])
                evaluate_task(pilot, task, reader, model, True)
                evaluate_task(directory, task, reader, model, False)
                print("Verified", model, condition, task, flush=True)
        write_json(output / "complete.json", dict(complete=True, model=model, revision=MODELS[model],
                   tasks=TASKS, conditions=CONDITIONS, full_data=True))
    finally:
        reader.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "qa"])
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--model")
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.results)
    else:
        assert args.model
        qa(args.results, args.model)

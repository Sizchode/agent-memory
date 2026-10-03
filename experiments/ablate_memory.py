"""Evaluate AMOR components and parameter sensitivity with existing evaluators."""

import argparse
from collections import Counter
import csv
from dataclasses import replace
import gc
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import zipfile

from experiments import ablate_components as components
from experiments.runner import _read_retrieval_records
from optimization import ircot
from optimization.ircot import BASE, GRAPH, MODELS, write_json
from optimization.report_results import TASK_METRICS

ROOT = BASE / "optimization_paper_ablation_seed42_20260927"
RAW = BASE / "optimization_retained_index_raw_relations_seed42_20260916/statement_projection_loop_free_raw_relations_retained_index_rrf_window"
CORE = ("without_projection", "without_propagation")
FINE = ("binary_weights", "alpha_025", "alpha_075", "lambda_001", "lambda_010", "raw_relations")
PARAMETERS = {"alpha_025": (0.25, 0.05), "alpha_075": (0.75, 0.05),
              "lambda_001": (0.5, 0.01), "lambda_010": (0.5, 0.1)}
CAPS = (1, 3, 5)
CONNECTIONS = ("without_projected_connections", "without_source_reweighting")
CONNECTION_ROOT = ROOT / "connection_controls"
SIMPLE_ROOT = BASE / "optimization_simplified_amor_seed42_20260929"
SIMPLE_VARIANTS = ("full", "without_propagation", "without_context_augmentation")


def initialize_simple():
    protocol = json.loads(json.dumps(dict(seed=42, models=MODELS, tasks=TASK_METRICS,
        variants=SIMPLE_VARIANTS, caps=CAPS, graph_source=str(CONNECTION_ROOT),
        graph_variant="without_projected_connections", new_reasoning_trajectories=True,
        recommendation_ablation="Dense ranking replaces PPR; BM25 fusion and augmentation retained",
        augmentation_ablation="Same selected sources and reasoning; only original source text in final QA",
        one_shot="Fresh QA for all three conditions; verified frozen original-question selections",
        native_baselines="Reuse unchanged native baseline results; never reuse old AMOR trajectories",
        test_as_dev=True, new_extraction=False, new_relation_normalization=False)))
    path = ROOT / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == protocol
    else:
        write_json(path, protocol)


def original_text_only(row):
    selected = []
    for item in row.retrieved:
        meta = item.metadata
        if meta.get("context_representation") != "original_source_and_frozen_window":
            continue
        lines = [f"Source position: {meta['source_position']}"]
        if meta["timestamp"] is not None:
            lines.append(f"Source timestamp: {meta['timestamp']}")
        lines.extend(["Original source record:", meta["original_source_text"]])
        selected.append(replace(item, text="\n".join(lines)))
    assert selected
    return replace(row, retrieved=selected)


def prepare_simple():
    audit = json.loads((CONNECTION_ROOT / "complete.json").read_text())
    assert audit["complete"] and audit["native_metrics_recomputed"]
    for task, (count, _) in TASK_METRICS.items():
        full = list(_read_retrieval_records(CONNECTION_ROOT / "inputs/without_projected_connections" / task / "retrieval.jsonl"))
        dense = list(_read_retrieval_records(components.ROOT / "inputs/without_propagation" / task / "retrieval.jsonl"))
        old_full = list(_read_retrieval_records(components.REFERENCE / "inputs/graph_retained_index_retained" / task / "retrieval.jsonl"))
        old_plain = list(_read_retrieval_records(BASE / "optimization_method_analysis_seed42_20260921/context/Qwen_Qwen3.5-4B/main/one_shot/sources" / task / "retrieval.jsonl"))
        for augmented, plain in zip(old_full, old_plain, strict=True):
            assert (augmented.group_id, augmented.case) == (plain.group_id, plain.case)
            assert [i.text for i in original_text_only(augmented).retrieved] == [i.text for i in plain.retrieved]
        assert len(full) == count
        assert [(r.group_id, r.case) for r in full] == [(r.group_id, r.case) for r in dense]
        for group in dict.fromkeys(r.group_id for r in full):
            marker = json.loads((CONNECTION_ROOT / "groups" / task / group / "complete.json").read_text())
            assert marker["full_context_exact"] and marker["reset_vectors_exact"]
            assert marker["new_llm_calls"] == marker["new_embeddings"] == 0
        variants = dict(full=full, without_propagation=dense,
                        without_context_augmentation=[original_text_only(r) for r in full])
        for variant, rows in variants.items():
            components.save_rows(ROOT / "one_shot/inputs" / variant / task / "retrieval.jsonl", rows)
        write_json(ROOT / "one_shot/inputs" / (task + ".json"), dict(complete=True, questions=count,
            variants=SIMPLE_VARIANTS, dense_reuse="PPR bypass is independent of graph adjacency",
            full_source=str(CONNECTION_ROOT), original_questions_only=True))
    print("Simplified one-shot inputs prepared", flush=True)


def initialize():
    ROOT.mkdir(parents=True, exist_ok=True)
    protocol = dict(seed=42, models=MODELS, tasks=TASK_METRICS, core=CORE, fine=FINE,
        parameters=PARAMETERS, default_parameters=(0.5, 0.05), caps=CAPS,
        graph=str(GRAPH), raw_graph=str(RAW), original_components=str(components.ROOT),
        recommendation_ablation="Replace PPR ranking with dense ranking; retain BM25 and augmentation",
        binary_weights="Keep full graph topology and replace all nonzero weights with one",
        parameter_selection="Predefined one-factor sensitivity; never select per-task configurations",
        main_text="one-shot and IRCoT component ablations", appendix="fine-grained controls and sensitivity",
        new_extraction=False, new_relation_normalization=False, test_as_dev=True)
    protocol = json.loads(json.dumps(protocol))
    path = ROOT / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == protocol
    else:
        write_json(path, protocol)
    with zipfile.ZipFile(ROOT / f"code_{os.environ['SLURM_JOB_ID']}.zip", "x", zipfile.ZIP_DEFLATED) as output:
        for name in ("experiments/ablate_memory.py", "experiments/ablate_memory.sbatch",
                     "experiments/ablate_components.py", "optimization/ircot.py",
                     "optimization/retriever/query_fact_context.py"):
            output.write(name, name)


def config_for(task, service="http://127.0.0.1:1/v1"):
    from optimization.run_graph import retrieval_config
    config, _ = retrieval_config(SimpleNamespace(task=task.replace("_", " "), output_root=ROOT,
        generator_base_url=service, path="/oscar/scratch/zliu328/agent-memory-data/locomo/locomo10.json",
        data_root=str(Path.cwd() / "baseline_algorithms/HippoRAG/reproduce/dataset")))
    return config


def copy_cache(source_runtime, destination_runtime):
    source, = (source_runtime / "llm_cache").glob("*.sqlite")
    target = destination_runtime / "llm_cache" / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        if source.with_name(source.name + "-wal").exists():
            raise RuntimeError("Refuse to copy an active SQLite WAL cache")
        shutil.copy2(source, target)


def binary_weights(graph):
    result = graph.copy()
    assert all(weight > 0 for weight in result.es["weight"])
    result.es["weight"] = [1.0] * result.ecount()
    assert result.get_edgelist() == graph.get_edgelist()
    assert result.vs["name"] == graph.vs["name"]
    return result


def check_raw_graph(memory, task, group):
    import igraph as ig
    from hipporag.utils.misc_utils import text_processing
    from optimization.graph_construction.source_consolidation import retained_statements, statement_weights
    from optimization.graph_construction.compiled_sources import compile_sources
    from optimization.graph_construction.source_window import attach_source_windows
    from optimization.graph_construction.statement_incidence import statement_incidence_graph, project_statement_graph

    raw_meta = json.loads((RAW / task / "memory" / group / "graph.json").read_text())
    full_meta = json.loads((GRAPH / task / "memory" / group / "graph.json").read_text())
    assert raw_meta["source_graph"] == full_meta["source_graph"]
    assert raw_meta["rank_fusion"] == full_meta["rank_fusion"]
    hippo = memory._memory
    original = ig.Graph.Read_Pickle(raw_meta["source_graph"])
    openie, = Path(raw_meta["source_graph"]).parent.parent.glob("openie_results_ner_*.json")
    docs = json.loads(openie.read_text())["docs"]
    ordered = json.loads((RAW / task / "memory" / group / "lexical_source_keys.json").read_text())
    assert ordered == json.loads((GRAPH / task / "memory" / group / "lexical_source_keys.json").read_text())
    selected, stats = retained_statements(docs, ordered, text_processing, schema=None)
    full_contents = json.loads(Path(full_meta["compiled_source_file"]).read_text())
    timestamps = {key: value["timestamp"] for key, value in full_contents.items() if value["timestamp"] is not None}
    contents = attach_source_windows(compile_sources(docs, ordered, selected, timestamps, text_processing), ordered, 3)
    assert contents == memory.contents, "Raw relation control has a different context procedure"
    entities = {row["content"]: key for key, row in hippo.entity_embedding_store.get_all_id_to_rows().items()}
    original.es["weight"] = statement_weights(original, selected, entities).tolist()
    rebuilt = project_statement_graph(statement_incidence_graph(original, docs, selected, entities,
                                                               text_processing, refined=True), original.vcount())
    assert rebuilt.vs["name"] == hippo.graph.vs["name"]
    assert (rebuilt.get_adjacency_sparse(attribute="weight") != hippo.graph.get_adjacency_sparse(attribute="weight")).nnz == 0
    actual = {hippo.fact_embedding_store.get_row(key)["content"] for key in hippo.fact_node_keys}
    assert actual == {str(triple) for _, triple in selected}
    return dict(stats, graph_exact=True, context_exact=True, candidates_exact=True)


def prepare(task):
    import numpy as np
    from optimization.retriever.hipporag import CacheMissGuard, load_optimized_memory, load_query_embeddings
    from optimization.retriever.query_fact_context import QueryFactContext
    from utils.models import release_accelerator_memory

    expected = list(_read_retrieval_records(components.REFERENCE / "inputs/graph_retained_index_retained" / task / "retrieval.jsonl"))
    assert len(expected) == TASK_METRICS[task][0]
    for group in dict.fromkeys(row.group_id for row in expected):
        rows = [row for row in expected if row.group_id == group]
        for raw in (False, True):
            variants = ("raw_relations",) if raw else FINE[:-1]
            directory = ROOT / "one_shot/groups" / task / group
            marker = directory / ("raw_complete.json" if raw else "complete.json")
            if marker.exists():
                for variant in variants:
                    saved = list(_read_retrieval_records(directory / variant / "retrieval.jsonl"))
                    assert [(r.group_id, r.case) for r in saved] == [(r.group_id, r.case) for r in rows]
                continue
            graph_root = RAW if raw else GRAPH
            runtime = ROOT / "one_shot/runtime" / ("raw" if raw else "full") / task / group
            copy_cache(graph_root / task / "runtime" / group, runtime)
            memory = load_optimized_memory(config_for(task), graph_root / task / "memory" / group, runtime)
            control = selector = None
            try:
                hippo = memory._memory
                guard = CacheMissGuard(memory._generator)
                load_query_embeddings(hippo, components.QUERY_CACHE / task / group / "queries.npz")
                def reject_encoding(*args, **kwargs):
                    raise RuntimeError("Missing frozen original-question embedding")
                hippo.embedding_model.batch_encode = reject_encoding
                assert hippo.global_config.damping == 0.5
                assert hippo.global_config.passage_node_weight == 0.05
                raw_check = check_raw_graph(memory, task, group) if raw else None
                full = hippo.graph
                binary = None if raw else binary_weights(full)
                selector = QueryFactContext(hippo, memory.contents)
                control = components.PropagationControl(hippo)
                outputs = {variant: [] for variant in variants}
                for row in rows:
                    hippo.graph = full
                    hippo.global_config.damping, hippo.global_config.passage_node_weight = 0.5, 0.05
                    control.begin()
                    items = memory.retrieve(row.case.question, 5)
                    rendered = selector.render(row.case.question, items)
                    guard.check()
                    default_reset = control.reset
                    if not raw:
                        assert [r.text for r in rendered] == [r.text for r in row.retrieved]
                    for variant in variants:
                        if raw:
                            selected = rendered
                        else:
                            hippo.graph = binary if variant == "binary_weights" else full
                            alpha, weight = PARAMETERS.get(variant, (0.5, 0.05))
                            hippo.global_config.damping, hippo.global_config.passage_node_weight = alpha, weight
                            control.begin()
                            items = memory.retrieve(row.case.question, 5)
                            selected = selector.render(row.case.question, items)
                            guard.check()
                            if weight == 0.05:
                                if default_reset is None:
                                    assert control.reset is None
                                else:
                                    np.testing.assert_array_equal(default_reset, control.reset)
                        outputs[variant].append(replace(row, retrieved=selected))
                for variant, selected in outputs.items():
                    components.save_rows(directory / variant / "retrieval.jsonl", selected)
                write_json(marker, dict(complete=True, questions=len(rows), variants=variants,
                    original_context_exact=not raw, raw_reconstruction=raw_check,
                    new_embeddings=0, new_recognition_calls=guard.misses))
                print("Prepared", task, group, variants, flush=True)
            finally:
                if control is not None:
                    control.close()
                memory.close()
                memory = selector = control = None
                gc.collect()
                release_accelerator_memory()
    for variant in FINE:
        collected = []
        for group in dict.fromkeys(row.group_id for row in expected):
            collected.extend(_read_retrieval_records(ROOT / "one_shot/groups" / task / group / variant / "retrieval.jsonl"))
        assert [(r.group_id, r.case) for r in collected] == [(r.group_id, r.case) for r in expected]
        components.save_rows(ROOT / "one_shot/inputs" / variant / task / "retrieval.jsonl", collected)
    write_json(ROOT / "one_shot/inputs" / (task + ".json"), dict(complete=True, questions=len(expected), variants=FINE))


def qa(model):
    components.ROOT = ROOT / "one_shot"
    components.VARIANTS = FINE
    components.qa(model)


def iterative_backend(variant, config, task, group, directory, *, simplified=False):
    import igraph as ig

    runtime = directory / "runtime" / task / group.group_id
    copy_cache(GRAPH / task / "runtime" / group.group_id, runtime)
    memory = ircot.load_backend("optimized_graph", config, task.replace("_", " "), group, directory)
    try:
        if simplified:
            assert variant in ("full", "without_propagation")
            path = CONNECTION_ROOT / "groups" / task / group.group_id / "without_projected_connections.pickle"
            replacement = ig.Graph.Read_Pickle(str(path))
            assert replacement.vs["name"] == memory._memory.graph.vs["name"]
            memory._memory.graph = replacement
        elif variant == "without_projection":
            path = components.ROOT / "groups" / task / group.group_id / "without_projection.pickle"
            replacement = ig.Graph.Read_Pickle(str(path))
            assert replacement.vs["name"] == memory._memory.graph.vs["name"]
            memory._memory.graph = replacement
        if variant == "without_propagation":
            control = components.PropagationControl(memory._memory)
            control.begin(disabled=True)
            memory.propagation_control = control
        return memory
    except BaseException:
        memory.close()
        raise


def context_from_sources(selector, items):
    from optimization.retriever.compiled_sources import compiled_item
    return [compiled_item(item, selector.text_to_key[item.text],
                          selector.contents[selector.text_to_key[item.text]]) for item in items]


def prepare_native_participants():
    # The published GPT-3 adapter opens an unused diskcache during import.
    # Keep that cache job-local; restore HOME before loading any real models.
    home = os.environ.get("HOME")
    local = Path(os.environ.get("TMPDIR", "/tmp")) / f"amor-native-{os.environ['SLURM_JOB_ID']}"
    local.mkdir(parents=True, exist_ok=True)
    try:
        os.environ["HOME"] = str(local)
        importlib.import_module("commaqa.inference.ircot")
    finally:
        if home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = home
    adapter = importlib.import_module("commaqa.models.gpt3generator")
    assert Path(adapter.cache.directory).is_relative_to(local)
    return str(local)


def run_iterative(model, variant, generator_job, *, simplified=False):
    from baseline.base import RetrievedItem
    from experiments.runner import RetrievedCase, _answer_prompt, _official_generation
    from optimization.retriever.hipporag import load_query_embeddings
    from optimization.retriever.query_fact_context import QueryFactContext
    from utils.models import release_accelerator_memory
    import random

    local_cache = prepare_native_participants()
    service = ircot.endpoint(generator_job)
    reader = ircot.Reader(model)
    directory = ROOT / "ircot" / model.replace("/", "_") / variant
    output_variants = (variant, "without_context_augmentation") if simplified and variant == "full" else (variant,)
    def output_stage(name, pilot):
        return ROOT / "ircot" / model.replace("/", "_") / name / ("pilot" if pilot else "main")
    try:
        old = json.loads((BASE / "optimization_ircot_fact_context_seed42_20260920/main" /
                          model.replace("/", "_") / "reader.json").read_text())
        assert {k: v for k, v in old.items() if k != "gpu"} == {k: v for k, v in reader.metadata.items() if k != "gpu"}
        write_json(directory / f"reader_{os.environ['SLURM_JOB_ID']}.json", reader.metadata)
        write_json(directory / f"native_cache_{os.environ['SLURM_JOB_ID']}.json",
                   dict(directory=local_cache, unused_legacy_generation_cache=True, home_restored=True))
        for pilot in (True, False):
            stage = directory / ("pilot" if pilot else "main")
            for task, (count, _) in TASK_METRICS.items():
                if not pilot:
                    for name in output_variants:
                        for cap in CAPS:
                            gate = json.loads((output_stage(name, True) / f"cap_{cap}" / task / "qa_complete.json").read_text())
                            assert gate["complete"] and gate["pilot"]
                groups = ircot.groups_for(task.replace("_", " "))
                expected = list(_read_retrieval_records(components.REFERENCE / "inputs/graph_retained_index_retained" / task / "retrieval.jsonl"))
                assert [(g.group_id, c) for g in groups for c in g.cases] == [(r.group_id, r.case) for r in expected]
                selected_groups = groups[:1] if pilot else groups
                collected = {cap: [] for cap in CAPS}
                task_complete = all((output_stage(name, pilot) / f"cap_{cap}" / task / "qa_complete.json").exists()
                                    for name in output_variants for cap in CAPS)
                if task_complete:
                    for name, cap in ((name, cap) for name in output_variants for cap in CAPS):
                        target = output_stage(name, pilot) / f"cap_{cap}" / task
                        complete = json.loads((target / "qa_complete.json").read_text())
                        assert complete["complete"] and complete["pilot"] == pilot
                        if not pilot:
                            score = components.audited_score(target / "evaluations" / model.replace("/", "_"), task,
                                {(r.group_id, r.case.case_id) for r in expected})
                            assert score == complete["score"]
                    continue
                for group in selected_groups:
                    cases = group.cases[:2] if pilot else group.cases
                    runtime_dir = directory / "backend"
                    memory = iterative_backend(variant, config_for(task, service), task, group, runtime_dir,
                                               simplified=simplified)
                    selector = pipeline = None
                    try:
                        hippo = memory._memory
                        load_query_embeddings(hippo, components.QUERY_CACHE / task / group.group_id / "queries.npz")
                        meta = json.loads((GRAPH / task / "memory" / group.group_id / "graph.json").read_text())
                        selector = QueryFactContext(hippo, json.loads(Path(meta["compiled_source_file"]).read_text()))
                        assert set(selector.text_to_key) == set(group.memory_items)
                        pipeline = ircot.Pipeline(memory, group)
                        traces = []
                        for start in range(0, len(cases), ircot.BATCH_SIZE):
                            batch = cases[start:start + ircot.BATCH_SIZE]
                            saved = stage / "traces" / task / group.group_id / f"{start}.json"
                            if saved.exists():
                                output = json.loads(saved.read_text())
                                assert output["complete"] and output["pilot"] == pilot
                                assert output["case_ids"] == [c.case_id for c in batch]
                                if pilot:
                                    assert output["native_controller_verified"]
                                part = output["traces"]
                            else:
                                part, timings, _ = pipeline.batch(batch, reader, variant,
                                    f"{task}/{group.group_id}", pilot=pilot)
                                write_json(saved, dict(complete=True, pilot=pilot, model=model, variant=variant,
                                    case_ids=[c.case_id for c in batch], traces=part, batch_timings=timings,
                                    native_controller_verified=pilot))
                            traces.extend(part)
                        for case, trace in zip(cases, traces, strict=True):
                            assert trace["case_id"] == case.case_id and trace["question"] == case.question
                            assert 1 <= len(trace["rounds"]) <= max(CAPS) and trace["rounds"][-1]["stopped"]
                            for cap in CAPS:
                                indices = trace["rounds"][min(cap, len(trace["rounds"])) - 1]["selected_sources"]
                                items = [RetrievedItem(group.memory_items[i]) for i in indices]
                                context = selector.render(case.question, context_from_sources(selector, items))
                                collected[cap].append(RetrievedCase(group.group_id, case, context, 15))
                        print("Traces complete", model, variant, pilot, task, group.group_id, len(traces), flush=True)
                    finally:
                        memory.close()
                        memory = selector = pipeline = None
                        gc.collect()
                        release_accelerator_memory()
                for name, cap in ((name, cap) for name in output_variants for cap in CAPS):
                    target = output_stage(name, pilot) / f"cap_{cap}" / task
                    if (target / "qa_complete.json").exists():
                        continue
                    rows = collected[cap]
                    if name == "without_context_augmentation":
                        rows = [original_text_only(row) for row in rows]
                    if not pilot:
                        assert len(rows) == count
                        assert [(r.group_id, r.case) for r in rows] == [(r.group_id, r.case) for r in expected]
                    components.save_rows(target / "retrieval.jsonl", rows)
                    rng = random.Random(42)
                    requests = [dict(phase="answer", messages=_answer_prompt(r.case, r.retrieved, rng)[0],
                                     generation=_official_generation(r.case)) for r in rows]
                    for offset in range(0, len(requests), ircot.BATCH_SIZE):
                        check = reader.request(phase="check_context", items=requests[offset:offset + ircot.BATCH_SIZE])
                        assert not check["overflow"], str(target)
                    ircot.evaluate_task(target, task, reader, model, pilot)
                    print("QA complete", model, name, pilot, task, cap, flush=True)
            for name in output_variants:
                write_json(output_stage(name, pilot) / "complete.json", dict(complete=True, model=model, variant=name,
                    pilot=pilot, tasks=list(TASK_METRICS), caps=CAPS, native_controller_verified=pilot,
                    trajectory_directory=str(stage / "traces"), simplified_graph=simplified))
    finally:
        reader.close()


def stop_services(jobs):
    allowed = set()
    for name in ("generator.json", "elasticsearch.json"):
        path = ROOT / "services" / name
        if path.exists():
            allowed.add(str(json.loads(path.read_text())["job_id"]))
    for job in jobs:
        if str(job) not in allowed:
            raise ValueError(f"Refuse to cancel unrecorded service {job}")
        subprocess.run(["scancel", str(job)], check=True)
    write_json(ROOT / "services/stopped.json", dict(jobs=jobs, stopped=True))


def wait_for_service(generator_job):
    # First use on a new GPU architecture can compile kernels before serving.
    for attempt in range(4):
        try:
            service = ircot.endpoint(generator_job)
            write_json(ROOT / "services/ready.json", dict(ready=True, job_id=generator_job, base_url=service))
            return
        except RuntimeError:
            if attempt == 3:
                raise
            print("Recognition service is still starting; waiting on CPU", flush=True)


def audit_result(directory, model, task, setting, variant):
    from experiments.runner import _score
    from utils.hipporag_metrics import gold_passage_recall_at_k

    expected = list(_read_retrieval_records(components.REFERENCE / "inputs/graph_retained_index_retained" / task / "retrieval.jsonl"))
    inputs = list(_read_retrieval_records(directory / "retrieval.jsonl"))
    assert [(r.group_id, r.case) for r in inputs] == [(r.group_id, r.case) for r in expected]
    marker = json.loads((directory / "qa_complete.json").read_text())
    assert marker["complete"] and not marker["pilot"] and marker["questions"] == len(expected)
    output = directory / "evaluations" / model.replace("/", "_")
    score = components.audited_score(output, task, {(r.group_id, r.case.case_id) for r in expected})
    assert score == marker["score"]
    with (output / "predictions.jsonl").open() as stream:
        for row in inputs:
            prediction = json.loads(next(stream))
            assert (prediction["group_id"], prediction["case_id"]) == (row.group_id, row.case.case_id)
            assert prediction["metrics"] == _score(row.case, prediction["prediction"], row.retrieved, row.top_k)
            assert [i["text"] for i in prediction["retrieved"]] == [i.text for i in row.retrieved]
        assert next(stream, None) is None
    with (output / "qa_usage.jsonl").open() as stream:
        usage = [json.loads(line) for line in stream]
    assert len(usage) == len(inputs)
    assert [r["case_id"] for r in usage] == [r.case.case_id for r in inputs]
    tokens = sum(r["input_tokens"] + r["output_tokens"] for r in usage)
    assert tokens == marker["input_tokens"] + marker["output_tokens"]
    result = dict(model=model, task=task, setting=setting, variant=variant, questions=len(inputs),
        metric=TASK_METRICS[task][1], score=100 * score, qa_tokens=tokens, source=str(directory))
    evidence = None
    if task == "2WikiMultiHopQA":
        k = 5 if setting == "one_shot" else 15
        recalls, complete = [], 0
        for row in inputs:
            sources = [i.metadata["original_source_text"] for i in row.retrieved
                       if i.metadata.get("context_representation") == "original_source_and_frozen_window"]
            assert 0 < len(sources) <= k
            recalls.append(gold_passage_recall_at_k(row.case.gold_passages, sources, k))
            complete += set(row.case.gold_passages).issubset(sources)
        evidence = dict(model=model, setting=setting, variant=variant, k=k, questions=len(inputs),
            recall=100 * sum(recalls) / len(inputs), all_support=100 * complete / len(inputs))
    return result, evidence


def report(allow_incomplete=False):
    previous = json.loads((BASE / "optimization_method_analysis_seed42_20260921/ablation_summary.json").read_text())
    old = {(r["model"], r["task"], r["setting"]): r["conditions"] for r in previous["results"]}
    rows, evidence, missing = [], [], []
    additional = ("graph_all_index_all", "graph_all_index_retained", "graph_retained_index_all", "without_rrf")
    for model in MODELS:
        slug = model.replace("/", "_")
        for task in TASK_METRICS:
            for setting in ("one_shot", "cap_1", "cap_3", "cap_5"):
                inherited = old[model, task, setting]
                conditions = {"full": Path(inherited["full"]["directory"]),
                              "without_context_augmentation": Path(inherited["sources"]["directory"])}
                if setting == "one_shot":
                    conditions["full"] = components.REFERENCE / slug / "main/graph_retained_index_retained" / task
                    conditions.update({v: components.ROOT / slug / "main" / v / task for v in CORE})
                    conditions.update({v: ROOT / "one_shot" / slug / "main" / v / task for v in FINE})
                    conditions.update({v: Path(inherited[v]["directory"]) for v in additional})
                else:
                    conditions.update({v: ROOT / "ircot" / slug / v / "main" / setting / task for v in CORE})
                for variant, directory in conditions.items():
                    if not (directory / "qa_complete.json").exists():
                        missing.append(dict(model=model, task=task, setting=setting, variant=variant, path=str(directory)))
                        continue
                    row, retrieval = audit_result(directory, model, task, setting, variant)
                    rows.append(row)
                    if retrieval:
                        evidence.append(retrieval)
    for name, values in (("qa_results.csv", rows), ("evidence_results.csv", evidence)):
        if values:
            with (ROOT / name).open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values)
    write_json(ROOT / "audit.json", dict(complete=not missing, conditions=len(rows), missing=missing,
        native_scores_recomputed=True, questions_and_context_checked=True, seed=42, test_as_dev=True))
    scores = {(r["model"], r["task"], r["setting"], r["variant"]): r["score"] for r in rows}
    sensitivity = ("full", *PARAMETERS)
    if all((m, t, "one_shot", v) in scores for m in MODELS for t in TASK_METRICS for v in sensitivity):
        write_sensitivity(scores)
    if missing:
        if allow_incomplete:
            print(json.dumps(dict(complete=False, audited=len(rows), missing=len(missing))), flush=True)
            return
        raise RuntimeError(f"Incomplete ablation matrix: {len(missing)} conditions missing")
    tables = ROOT / "tables"
    tables.mkdir(exist_ok=True)
    labels = {"full": "AMOR", "without_projection": "w/o projection",
        "without_propagation": "w/o recommendation", "without_context_augmentation": "w/o context augmentation"}
    for setting in ("one_shot", "cap_1", "cap_3", "cap_5"):
        setting_label = "One-shot QA" if setting == "one_shot" else f"IRCoT with a maximum of {setting[-1]} iterations"
        lines = [r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
            r"\caption{" + setting_label + r": component ablations (\%). Each task uses its native metric. Recommendation is ablated by replacing graph propagation with dense retrieval while retaining BM25 fusion.}",
            r"\label{tab:ablation-" + setting.replace("_", "-") + "}",
            r"\begin{tabular}{lrrrrrr}", r"\toprule",
            r"Method & SH-Doc & MH-Doc & FC-SH & FC-MH & LoCoMo & 2Wiki \\"]
        for model in MODELS:
            lines.extend([r"\midrule", r"\multicolumn{7}{c}{\textbf{" + model.split("/")[-1] + r"}} \\"])
            for variant, label in labels.items():
                lines.append(label + " & " + " & ".join(f"{scores[model, task, setting, variant]:.2f}" for task in TASK_METRICS) + r" \\")
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
        (tables / (setting + "_components.tex")).write_text("\n".join(lines) + "\n")
        paper = [line.replace("tab:ablation-", "tab:paper-ablation-").replace(
            ": component ablations", ": recommendation and context augmentation ablations")
            for line in lines if not line.startswith("w/o projection &")]
        (tables / (setting + "_paper_components.tex")).write_text("\n".join(paper) + "\n")
    fine_labels = {"full": "AMOR", "binary_weights": "Binary edge weights",
        "raw_relations": "w/o relation normalization", "alpha_025": r"$\alpha=0.25$",
        "alpha_075": r"$\alpha=0.75$", "lambda_001": r"$\lambda=0.01$", "lambda_010": r"$\lambda=0.10$",
        "graph_all_index_all": "All facts in graph and candidates",
        "graph_all_index_retained": "All facts in graph only",
        "graph_retained_index_all": "All facts in candidates only", "without_rrf": "w/o BM25 fusion"}
    for model in MODELS:
        lines = [r"\begin{table*}[t]", r"\centering\small", r"\caption{One-shot fine-grained controls with " +
            model.split("/")[-1] + r" (\%). Only the named component or parameter differs from AMOR.}",
            r"\begin{tabular}{lrrrrrr}", r"\toprule",
            r"Condition & SH-Doc & MH-Doc & FC-SH & FC-MH & LoCoMo & 2Wiki \\", r"\midrule"]
        for variant, label in fine_labels.items():
            lines.append(label + " & " + " & ".join(f"{scores[model, task, 'one_shot', variant]:.2f}" for task in TASK_METRICS) + r" \\")
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
        (tables / (model.replace("/", "_") + "_appendix.tex")).write_text("\n".join(lines) + "\n")
    write_analysis(tables, scores, evidence)
    write_json(ROOT / "complete.json", dict(complete=True, conditions=len(rows), tasks=list(TASK_METRICS),
        models=MODELS, core=CORE, fine=FINE, caps=CAPS, seed=42,
        full_matrix_checked=True, main_method_unchanged=True))


def write_sensitivity(scores):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    plt.rcParams.update({"font.family": "serif", "font.weight": "bold", "axes.labelweight": "bold",
                         "axes.titleweight": "bold"})
    figures = ROOT / "figures"
    figures.mkdir(exist_ok=True)
    model_labels = ("Qwen3.5-4B", "Qwen3.5-9B", "Gemma3-4B", "Llama3.1-8B")
    task_labels = ("SH-Doc", "MH-Doc", "FC-SH", "FC-MH", "LoCoMo", "2Wiki")
    for parameter, values, variants in (("alpha", [0.25, 0.5, 0.75], ["alpha_025", "full", "alpha_075"]),
                                       ("lambda", [0.01, 0.05, 0.1], ["lambda_001", "full", "lambda_010"])):
        fig, axes = plt.subplots(4, 6, figsize=(12, 6.4), layout="constrained")
        fig.get_layout_engine().set(h_pad=0.015, w_pad=0.025, hspace=0.025, wspace=0.03)
        for r, model in enumerate(MODELS):
            for c, task in enumerate(TASK_METRICS):
                ax = axes[r, c]
                ax.plot(values, [scores[model, task, "one_shot", v] for v in variants], "o-", color="#5E81AC")
                ax.set_xticks(values)
                ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
                ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
                ax.grid(alpha=0.3)
                if r == 0:
                    metric = "Substring EM (%)" if c < 4 else ("F1 / accuracy (%)" if c == 4 else "Answer F1 (%)")
                    ax.set_title(task_labels[c] + "\n" + metric, fontsize=9, fontweight="bold")
                if c == 0:
                    ax.set_ylabel(model_labels[r], fontsize=9, fontweight="bold")
                if r == 3:
                    ax.set_xlabel("$\\" + parameter + "$", fontweight="bold")
                ax.tick_params(labelsize=8)
        fig.savefig(figures / f"sensitivity_{parameter}.pdf", bbox_inches="tight")
        fig.savefig(figures / f"sensitivity_{parameter}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    tables = ROOT / "tables"
    tables.mkdir(exist_ok=True)
    variants = {"full": "Ref.", "alpha_025": r"$\alpha=0.25$", "alpha_075": r"$\alpha=0.75$",
                "lambda_001": r"$\lambda=0.01$", "lambda_010": r"$\lambda=0.10$"}
    lines = [r"\begin{table}[t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{2pt}",
        r"\renewcommand{\arraystretch}{0.95}",
        r"\caption{Parameter sensitivity in one-shot QA (\%).",
        r"Ref.\ uses $\alpha=0.50$ and $\lambda=0.05$.",
        r"Each other row changes only the specified parameter.",
        r"Metrics follow the experimental setup.}", r"\label{tab:parameter-sensitivity}",
        r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}l*{6}{r}@{}}", r"\toprule",
        "Setting & " + " & ".join(task_labels) + r" \\"]
    for model in MODELS:
        name = model.split("/")[-1].replace("gemma-", "Gemma-")
        lines.extend([r"\midrule", r"\multicolumn{7}{c}{\textbf{" + name + r"}} \\"])
        for variant, label in variants.items():
            values = [f"{scores[model, t, 'one_shot', variant]:.2f}" for t in TASK_METRICS]
            lines.append(label + " & " + " & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular*}", r"\end{table}"])
    (tables / "parameter_sensitivity.tex").write_text("\n".join(lines) + "\n")


def write_analysis(directory, scores, evidence):
    def comparison(setting, variant):
        deltas = [scores[model, task, setting, "full"] - scores[model, task, setting, variant]
                  for model in MODELS for task in TASK_METRICS]
        return sum(d > 0 for d in deltas), sum(d == 0 for d in deltas), sum(d < 0 for d in deltas)

    labels = {"without_projection": "hypergraph projection", "without_propagation": "graph propagation",
              "without_context_augmentation": "context augmentation"}
    main = [r"\subsection{Component Ablations}", r"\label{sec:component-ablation}",
        r"We examine context recommendation and context augmentation in both one-shot QA and IRCoT.",
        r"\textit{w/o recommendation} replaces graph propagation with dense retrieval while retaining BM25 fusion and context augmentation.",
        r"\textit{w/o context augmentation} keeps the selected records fixed and removes the additional facts and neighboring text.",
        r"The recommendation control reruns retrieval while preserving the corpus, embedding model, context construction procedure, answer prompts, and evaluation metrics.",
        r"For IRCoT, removing recommendation requires new reasoning trajectories; the augmentation ablation changes only the final answer context.",
        r"Tables~\ref{tab:paper-ablation-one-shot}, \ref{tab:paper-ablation-cap-1}, \ref{tab:paper-ablation-cap-3}, and~\ref{tab:paper-ablation-cap-5} report all six tasks and four LLMs."]
    summary = []
    for setting in ("one_shot", "cap_1", "cap_3", "cap_5"):
        for variant in labels:
            wins, ties, losses = comparison(setting, variant)
            summary.append(dict(setting=setting, variant=variant, full_wins=wins, ties=ties, full_losses=losses))
    for variant, heading in (("without_propagation", "Context Recommendation"),
                             ("without_context_augmentation", "Context Augmentation")):
        counts = [comparison(setting, variant)[0] for setting in ("one_shot", "cap_1", "cap_3", "cap_5")]
        main.extend([r"\paragraph{" + heading + ".}",
            f"Retaining this component improves {counts[0]} of 24 one-shot settings.",
            f"Within IRCoT, it improves {counts[1]}, {counts[2]}, and {counts[3]} of 24 settings at iteration limits of 1, 3, and 5, respectively."])
        if variant == "without_propagation":
            deltas = [scores[m, "2WikiMultiHopQA", "one_shot", "full"] -
                      scores[m, "2WikiMultiHopQA", "one_shot", variant] for m in MODELS]
            main.extend([f"On 2Wiki, removing graph propagation reduces one-shot answer F1 by {min(deltas):.2f} to {max(deltas):.2f} percentage points across the four LLMs.",
                r"These gains are obtained while retaining dense matching, BM25 fusion, and context augmentation in the ablated condition, demonstrating the contribution of graph propagation beyond direct matching."])
        else:
            main.append(r"Because the selected records and reasoning trajectories are fixed, this comparison isolates the effect of supplementing the final answer context rather than changing the evidence selected during reasoning.")
    main.extend([r"These comparisons assess the components within AMOR rather than establishing that any component is necessary for every memory task.",
        r"Additional controls for edge weights, fact selection, relation normalization, and parameter sensitivity appear in Appendix~\ref{app:fine-grained-ablation}."])
    (directory / "component_analysis.tex").write_text("\n".join(main) + "\n")
    appendix = [r"\subsection{Fine-Grained Ablations}", r"\label{app:fine-grained-ablation}",
        r"We retain AMOR's graph topology and replace all nonzero edge weights with one to isolate the contribution of edge weighting.",
        r"The relation normalization control groups facts by their original relation labels while preserving the subsequent selection rule and graph construction procedure.",
        r"We also vary whether the graph and fact candidates use all extracted facts or only the selected facts, and remove BM25 fusion as a separate control."]
    for variant, label in (("binary_weights", "binary edge weights"), ("raw_relations", "original relation labels"),
                           ("graph_all_index_all", "all facts for both graph construction and candidate matching"),
                           ("graph_all_index_retained", "all facts for graph construction only"),
                           ("graph_retained_index_all", "all facts for candidate matching only"),
                           ("without_rrf", "graph ranking without BM25 fusion")):
        wins, ties, losses = comparison("one_shot", variant)
        appendix.append(f"Compared with {label}, full AMOR wins in {wins} settings, ties in {ties}, and loses in {losses}.")
    appendix.extend([r"\paragraph{Parameter Sensitivity.}",
        r"Starting from the reference configuration $\alpha=0.50$ and $\lambda=0.05$, we vary $\alpha\in\{0.25,0.50,0.75\}$ and $\lambda\in\{0.01,0.05,0.10\}$ one at a time, keeping all other settings fixed.",
        r"Every setting uses the full six-task evaluation across four LLMs.",
        r"We do not use this sensitivity analysis to select parameter values or revise the reference configuration.",
        r"Table~\ref{tab:parameter-sensitivity} reports each task's original QA metric rather than averaging scores across different metrics.",
        r"\input{tables/parameter_sensitivity}"])
    for parameter, variants in (("alpha", ("alpha_025", "alpha_075")), ("lambda", ("lambda_001", "lambda_010"))):
        changes = [abs(scores[m, t, "one_shot", v] - scores[m, t, "one_shot", "full"])
                   for m in MODELS for t in TASK_METRICS for v in variants]
        appendix.append(f"Across the evaluated settings, changing $\\{parameter}$ produces a maximum absolute change of {max(changes):.2f} percentage points relative to the reference configuration.")
    appendix.extend([r"\paragraph{Evidence Coverage.}",
        r"We measure supporting passage recall and the fraction of questions with all supporting passages retrieved on 2Wiki.",
        r"One-shot results use the five retrieved records, while IRCoT results use the accumulated set of at most fifteen records.",
        r"Additional facts and neighboring text are excluded from these retrieval metrics."])
    coverage = [
        r"\begin{table}[t]", r"\centering\small",
        r"\caption{Supporting passage coverage on 2Wiki (\%). All support measures the fraction of questions for which all annotated supporting passages are retrieved.}",
        r"\label{tab:construction-evidence}", r"\begin{tabular}{lrr}", r"\toprule",
        r"Condition & Recall@5 & All support \\", r"\midrule"]
    for variant, label in (("without_projection", "w/o projection"), ("binary_weights", "Binary edge weights"),
                           ("full", "AMOR"), ("without_propagation", "w/o recommendation")):
        records = [r for r in evidence if r["setting"] == "one_shot" and r["variant"] == variant]
        assert len(records) == len(MODELS)
        assert len({(r["recall"], r["all_support"]) for r in records}) == 1
        coverage.append(f"{label} & {records[0]['recall']:.2f} & {records[0]['all_support']:.2f}" + r" \\")
    coverage.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (directory / "construction_evidence.tex").write_text("\n".join(coverage) + "\n")
    (directory / "appendix_analysis.tex").write_text("\n".join(appendix) + "\n")
    write_json(ROOT / "component_comparisons.json", summary)


def report_connections():
    """Report both new controls together with the existing full and removal runs."""
    with (ROOT / "qa_results.csv").open() as stream:
        prior = list(csv.DictReader(stream))
    prior = {(row["model"], row["task"], row["variant"]): row for row in prior
             if row["setting"] == "one_shot" and row["variant"] in ("full", "without_projection")}
    qa_rows, evidence_rows = [], []
    for model in MODELS:
        for task in TASK_METRICS:
            for variant in ("without_projection", *CONNECTIONS, "full"):
                if variant in CONNECTIONS:
                    directory = CONNECTION_ROOT / model.replace("/", "_") / "main" / variant / task
                else:
                    directory = Path(prior[model, task, variant]["source"])
                qa, evidence = audit_result(directory, model, task, "one_shot", variant)
                qa_rows.append(qa)
                if evidence is not None:
                    evidence_rows.append(evidence)
    for name, rows in (("qa_results", qa_rows), ("evidence_results", evidence_rows)):
        with (CONNECTION_ROOT / (name + ".csv")).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    scores = {(r["model"], r["task"], r["variant"]): r["score"] for r in qa_rows}
    comparisons = {}
    for variant in ("without_projection", *CONNECTIONS):
        deltas = [scores[model, task, "full"] - scores[model, task, variant]
                  for model in MODELS for task in TASK_METRICS]
        comparisons[variant] = dict(win=sum(d > 0 for d in deltas), tie=sum(d == 0 for d in deltas),
                                    loss=sum(d < 0 for d in deltas))
    baseline_path = BASE / "paper_costs_20260925/data/one_shot_qa_cost.csv"
    with baseline_path.open() as stream:
        baselines = [r for r in csv.DictReader(stream) if r["method"] != "libra"]
    baseline_rows = []
    for row in qa_rows:
        reference = [r for r in baselines if r["model"] == row["model"].replace("/", "_")
                     and r["task"] == row["task"]]
        assert len(reference) == 7 and len({r["method"] for r in reference}) == 7
        assert all(int(r["questions"]) == row["questions"] for r in reference)
        best = max(float(r["score"]) for r in reference)
        baseline_rows.append(dict(model=row["model"], task=row["task"], variant=row["variant"],
            score=row["score"], best_native_baseline=best, delta=row["score"] - best,
            best_methods=";".join(r["method"] for r in reference if float(r["score"]) == best)))
    with (CONNECTION_ROOT / "native_baseline_comparisons.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(baseline_rows[0]))
        writer.writeheader()
        writer.writerows(baseline_rows)
    external = {variant: dict(win=sum(r["delta"] > 0 for r in baseline_rows if r["variant"] == variant),
        tie=sum(r["delta"] == 0 for r in baseline_rows if r["variant"] == variant),
        loss=sum(r["delta"] < 0 for r in baseline_rows if r["variant"] == variant))
        for variant in ("without_projection", *CONNECTIONS, "full")}
    write_json(CONNECTION_ROOT / "complete.json", dict(complete=True, conditions=len(qa_rows),
        new_conditions=len(MODELS) * len(TASK_METRICS) * len(CONNECTIONS),
        tasks=list(TASK_METRICS), models=MODELS, comparisons=comparisons,
        native_metrics_recomputed=True, questions_and_context_checked=True,
        external_baseline_source=str(baseline_path), external_baseline_comparisons=external))
    print(json.dumps(comparisons, indent=2), flush=True)
    print("Native baseline comparisons:", json.dumps(external, indent=2), flush=True)


def analyze_construction():
    """Audit frozen construction and report all native 2Wiki question types."""
    from collections import defaultdict
    import igraph as ig
    import numpy as np
    from baseline.official import _prepend
    _prepend(Path(__file__).resolve().parents[1] / "baseline_algorithms/HippoRAG/src")
    from hipporag.utils.misc_utils import text_processing
    from optimization.graph_construction.statement_incidence import statement_incidence_graph, project_statement_graph
    from optimization.graph_construction.source_consolidation import statement_weights
    from utils.hipporag_metrics import gold_passage_recall_at_k

    output = ROOT / "construction_analysis"
    output.mkdir(exist_ok=True)
    graph_rows = []
    for task in TASK_METRICS:
        paths = sorted((GRAPH / task / "memory").glob("*/graph.json"))
        assert paths, task
        for path in paths:
            metadata = json.loads(path.read_text())
            original = ig.Graph.Read_Pickle(metadata["source_graph"])
            full = ig.Graph.Read_Pickle(metadata["constructed_graph_file"])
            contents = json.loads(Path(metadata["compiled_source_file"]).read_text())
            entities = {v["content"]: v["name"] for v in original.vs
                        if v["name"].startswith("entity-")}
            membership, selected = components.membership_graph(original, contents, entities, text_processing)
            original.es["weight"] = statement_weights(original, selected, entities).tolist()
            openie, = Path(metadata["source_graph"]).parent.parent.glob("openie_results_ner_*.json")
            docs = json.loads(openie.read_text())["docs"]
            incidence_graph = statement_incidence_graph(original, docs, selected, entities,
                                                        text_processing, refined=True)
            reconstructed = project_statement_graph(incidence_graph, original.vcount())
            assert full.vs["name"] == reconstructed.vs["name"] == membership.vs["name"]
            adjacency = full.get_adjacency_sparse(attribute="weight").tocsr()
            assert (adjacency != reconstructed.get_adjacency_sparse(attribute="weight")).nnz == 0
            incidence = incidence_graph.get_adjacency_sparse(attribute="weight")[:full.vcount(), full.vcount():].tocsc()
            sizes = np.asarray(incidence.sum(axis=0)).ravel()
            active = sizes > 0
            assert np.all(sizes[active] >= 2)
            projected = adjacency - membership.get_adjacency_sparse(attribute="weight")
            expected_degrees = np.asarray(incidence[:, active].sum(axis=1)).ravel()
            actual_degrees = np.asarray(projected.sum(axis=1)).ravel()
            np.testing.assert_allclose(actual_degrees, expected_degrees, rtol=1e-12, atol=1e-12)
            counts = Counter(int(value) for value in sizes[active])
            members = Counter(tuple(incidence.indices[incidence.indptr[i]:incidence.indptr[i + 1]])
                              for i in np.flatnonzero(active))
            entity_edges = sum(full.vs[e.source]["name"].startswith("entity-") and
                               full.vs[e.target]["name"].startswith("entity-") for e in full.es)
            graph_rows.append(dict(task=task, group=path.parent.name, retained_facts=int(active.sum()),
                hyperedge_sizes=json.dumps(dict(sorted(counts.items()))),
                distinct_member_sets=len(members), repeated_member_sets=sum(n > 1 for n in members.values()),
                nodes=full.vcount(), full_edges=full.ecount(), membership_edges=membership.ecount(),
                entity_entity_edges=entity_edges, max_projected_degree=float(actual_degrees.max()),
                reconstruction_exact=True, projected_degree_identity=True))
            print("Graph verified:", task, path.parent.name, dict(counts), flush=True)

    with (ROOT / "qa_results.csv").open() as stream:
        reported = list(csv.DictReader(stream))
    connection_audit = json.loads((CONNECTION_ROOT / "complete.json").read_text())
    assert connection_audit["complete"] and connection_audit["native_metrics_recomputed"]
    with (CONNECTION_ROOT / "qa_results.csv").open() as stream:
        additional = [row for row in csv.DictReader(stream) if row["variant"] in CONNECTIONS]
    assert len(additional) == len(MODELS) * len(TASK_METRICS) * len(CONNECTIONS)
    reported.extend(additional)
    native_path = Path("baseline_algorithms/HippoRAG/reproduce/dataset/2wikimultihopqa.json")
    native = {row["_id"]: row for row in json.loads(native_path.read_text())}
    assert len(native) == TASK_METRICS["2WikiMultiHopQA"][0]
    native_types = sorted({r["type"] for r in native.values()})
    variants = ("full", "without_projection", "binary_weights", *CONNECTIONS)
    qa_rows, type_rows, selections = [], [], {}
    scores = {}
    for row in reported:
        if row["variant"] not in variants or (row["setting"] != "one_shot" and row["variant"] == "binary_weights"):
            continue
        checked, evidence = audit_result(Path(row["source"]), row["model"], row["task"], row["setting"], row["variant"])
        assert checked["score"] == float(row["score"])
        scores[row["model"], row["task"], row["setting"], row["variant"]] = checked["score"]
        qa_rows.append(checked)
        if row["task"] != "2WikiMultiHopQA":
            continue
        inputs = list(_read_retrieval_records(Path(row["source"]) / "retrieval.jsonl"))
        assert {r.case.case_id for r in inputs} == set(native)
        predictions_path = Path(row["source"]) / "evaluations" / row["model"].replace("/", "_") / "predictions.jsonl"
        with predictions_path.open() as stream:
            predictions = {p["case_id"]: p for p in map(json.loads, stream)}
        by_type = defaultdict(list)
        selected = {}
        k = 5 if row["setting"] == "one_shot" else 15
        for item in inputs:
            case = item.case
            assert case.question == native[case.case_id]["question"]
            sources = [r.metadata["original_source_text"] for r in item.retrieved
                       if r.metadata.get("context_representation") == "original_source_and_frozen_window"]
            assert 0 < len(sources) <= k
            gold = set(case.gold_passages)
            selected[case.case_id] = (gold, set(sources))
            result = (gold_passage_recall_at_k(case.gold_passages, sources, k),
                      float(gold.issubset(sources)), predictions[case.case_id]["metrics"]["answer_f1"])
            by_type[native[case.case_id]["type"]].append(result)
            by_type["ALL"].append(result)
        assert set(by_type) == set(native_types) | {"ALL"}
        for kind in ["ALL", *native_types]:
            values = by_type[kind]
            means = np.mean(values, axis=0) * 100
            type_rows.append(dict(model=row["model"], setting=row["setting"], variant=row["variant"],
                question_type=kind, questions=len(values), k=k,
                recall=float(means[0]), all_support=float(means[1]), answer_f1=float(means[2])))
        if row["setting"] == "one_shot":
            key = ("shared", row["setting"], row["variant"])
            if key in selections:
                assert selections[key] == selected, "One-shot source selection differs across readers"
            selections[key] = selected
        else:
            selections[row["model"], row["setting"], row["variant"]] = selected
        print("Predictions verified:", row["model"], row["setting"], row["variant"], flush=True)

    comparisons = []
    for (model, task, setting, variant), score in scores.items():
        if variant == "full":
            continue
        full = scores[model, task, setting, "full"]
        comparisons.append(dict(model=model, task=task, setting=setting, control=variant,
                                amor=full, control_score=score, delta=full - score))
    evidence_pairs = []
    for (model, setting, variant), selected in selections.items():
        if variant == "full":
            continue
        full = selections[model, setting, "full"]
        for kind in ["ALL", *native_types]:
            ids = [key for key in native if kind == "ALL" or native[key]["type"] == kind]
            totals = Counter()
            for key in ids:
                gold, control = selected[key]
                gold_full, amor = full[key]
                assert gold == gold_full
                totals.update(dict(gold_occurrences=len(gold), missed=len(gold - control),
                    recovered=len((gold - control) & amor), lost=len((gold & control) - amor),
                    complete_recovered=int(gold.issubset(amor) and not gold.issubset(control)),
                    complete_lost=int(gold.issubset(control) and not gold.issubset(amor))))
            evidence_pairs.append(dict(model=model, setting=setting, control=variant,
                                       question_type=kind, questions=len(ids), **totals))
    for name, rows in (("graphs", graph_rows), ("qa_verified", qa_rows), ("qa_deltas", comparisons),
                       ("native_question_types", type_rows), ("evidence_pairs", evidence_pairs)):
        with (output / f"{name}.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(output / "audit.json", dict(complete=True, graphs=len(graph_rows), qa_conditions=len(qa_rows),
        tasks=list(TASK_METRICS), models=MODELS, native_question_types=native_types,
        native_question_type_source=str(native_path), questions=1000, no_new_inference=True,
        new_parameter_selection=False, split="Existing development-used evaluation questions; no new held-out evidence",
        hypergraph_reduction_reference="https://doi.org/10.1007/s41109-020-00300-3",
        question_type_reference="https://aclanthology.org/2020.coling-main.580/"))
    print("Construction analysis complete:", output, flush=True)


def report_simple(allow_incomplete=False):
    rows, evidence, missing = [], [], []
    for model in MODELS:
        slug = model.replace("/", "_")
        for setting in ("one_shot", "cap_1", "cap_3", "cap_5"):
            for task, (count, _) in TASK_METRICS.items():
                contexts = {}
                for variant in SIMPLE_VARIANTS:
                    if setting == "one_shot":
                        directory = ROOT / "one_shot" / slug / "main" / variant / task
                    else:
                        directory = ROOT / "ircot" / slug / variant / "main" / setting / task
                    if not (directory / "qa_complete.json").exists():
                        missing.append(dict(model=model, task=task, setting=setting, variant=variant))
                        continue
                    row, support = audit_result(directory, model, task, setting, variant)
                    inputs = list(_read_retrieval_records(directory / "retrieval.jsonl"))
                    contexts[variant] = inputs
                    row.update(reasoning_input_tokens=0, reasoning_output_tokens=0,
                               recognition_input_observed=0, recognition_output_observed=0,
                               recognition_calls_observed=0, recognition_missing_usage=0,
                               trajectory_source="")
                    if setting != "one_shot":
                        trajectory_variant = "full" if variant == "without_context_augmentation" else variant
                        traces = ROOT / "ircot" / slug / trajectory_variant / "main/traces" / task
                        row["trajectory_source"] = str(traces)
                        lookup = {(item.group_id, item.case.case_id): item for item in inputs}
                        groups = {g.group_id: g for g in ircot.groups_for(task.replace("_", " "))}
                        seen = set()
                        cap = int(setting[-1])
                        for path in sorted(traces.glob("*/*.json")):
                            saved = json.loads(path.read_text())
                            assert saved["complete"] and not saved["pilot"]
                            assert saved["model"] == model and saved["variant"] == trajectory_variant
                            assert saved["case_ids"] == [t["case_id"] for t in saved["traces"]]
                            for trace in saved["traces"]:
                                key = (path.parent.name, trace["case_id"])
                                assert key not in seen
                                seen.add(key)
                                item = lookup[key]
                                assert trace["question"] == item.case.question
                                assert 1 <= len(trace["rounds"]) <= max(CAPS) and trace["rounds"][-1]["stopped"]
                                prefix = trace["rounds"][:cap]
                                actual = [i.metadata["original_source_text"] for i in original_text_only(item).retrieved]
                                assert actual == [groups[key[0]].memory_items[i] for i in prefix[-1]["selected_sources"]]
                                for step in prefix:
                                    usage = step["reasoning"]["usage"]
                                    for name in ("input_tokens", "output_tokens"):
                                        row["reasoning_" + name] += usage[name]
                                    for call in step["recognition"]:
                                        row["recognition_calls_observed"] += 1
                                        if call.get("prompt_tokens") is None or call.get("completion_tokens") is None:
                                            row["recognition_missing_usage"] += 1
                                        else:
                                            row["recognition_input_observed"] += call["prompt_tokens"]
                                            row["recognition_output_observed"] += call["completion_tokens"]
                        assert seen == set(lookup) and len(seen) == count
                    row["reader_tokens_per_question"] = (row["qa_tokens"] + row["reasoning_input_tokens"] +
                                                          row["reasoning_output_tokens"]) / count
                    rows.append(row)
                    if support:
                        evidence.append(support)
                if {"full", "without_context_augmentation"}.issubset(contexts):
                    for full, plain in zip(contexts["full"], contexts["without_context_augmentation"], strict=True):
                        assert (full.group_id, full.case) == (plain.group_id, plain.case)
                        assert [i.text for i in original_text_only(full).retrieved] == [i.text for i in plain.retrieved]
    for name, values in (("qa_results.csv", rows), ("evidence_results.csv", evidence)):
        if values:
            with (ROOT / name).open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values)
    scores = {(r["model"], r["task"], r["setting"], r["variant"]): r["score"] for r in rows}
    comparisons = []
    for (model, task, setting, variant), score in scores.items():
        if variant != "full" and (model, task, setting, "full") in scores:
            comparisons.append(dict(model=model, task=task, setting=setting, control=variant,
                full=scores[model, task, setting, "full"], control_score=score,
                delta=scores[model, task, setting, "full"] - score))
    if comparisons:
        with (ROOT / "component_comparisons.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
            writer.writeheader()
            writer.writerows(comparisons)
    with (CONNECTION_ROOT / "native_baseline_comparisons.csv").open() as stream:
        native = {(r["model"], r["task"]): r for r in csv.DictReader(stream) if r["variant"] == "full"}
    with (CONNECTION_ROOT / "qa_results.csv").open() as stream:
        previous_simple = {(r["model"], r["task"]): r for r in csv.DictReader(stream)
                           if r["variant"] == "without_projected_connections"}
    with (CONNECTION_ROOT.parent / "qa_results.csv").open() as stream:
        old = {(r["model"], r["task"], r["setting"]): r for r in csv.DictReader(stream) if r["variant"] == "full"}
    baseline_path = BASE / "paper_costs_20260925/data/ircot_cost.csv"
    with baseline_path.open() as stream:
        iterative = {(r["model"], r["task"], f"cap_{int(r['cap'])}", r["method"]): r
                     for r in csv.DictReader(stream) if r["method"] in ("bm25_native", "bm25_same_context")}
    baseline_rows = []
    for row in rows:
        if row["variant"] != "full":
            continue
        model, task, setting = row["model"], row["task"], row["setting"]
        reference = old[model, task, setting]
        targets = [("previous_full_AMOR", float(reference["score"]), reference["source"])]
        if setting == "one_shot":
            reference = native[model, task]
            targets.append(("best_native_baseline", float(reference["best_native_baseline"]),
                            str(CONNECTION_ROOT / "native_baseline_comparisons.csv")))
            reference = previous_simple[model, task]
            targets.append(("previous_simple_AMOR_repeat", float(reference["score"]), reference["source"]))
            current_inputs = list(_read_retrieval_records(Path(row["source"]) / "retrieval.jsonl"))
            previous_inputs = list(_read_retrieval_records(Path(reference["source"]) / "retrieval.jsonl"))
            for current, previous in zip(current_inputs, previous_inputs, strict=True):
                assert (current.group_id, current.case) == (previous.group_id, previous.case)
                assert [i.text for i in current.retrieved] == [i.text for i in previous.retrieved]
        else:
            for method in ("bm25_native", "bm25_same_context"):
                reference = iterative[model.replace("/", "_"), task, setting, method]
                targets.append((method, float(reference["score"]), reference["directory"]))
        for method, score, source in targets:
            baseline_rows.append(dict(model=model, task=task, setting=setting, comparison=method,
                full=row["score"], reference_score=score, delta=row["score"] - score,
                reference_source=source, reference_reused=True))
    if baseline_rows:
        with (ROOT / "baseline_comparisons.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(baseline_rows[0]))
            writer.writeheader()
            writer.writerows(baseline_rows)
    expected = len(MODELS) * len(TASK_METRICS) * (1 + len(CAPS)) * len(SIMPLE_VARIANTS)
    assert len(rows) + len(missing) == expected
    write_json(ROOT / "audit.json", dict(complete=not missing, conditions=len(rows), expected=expected,
        missing=missing, native_metrics_recomputed=True, contexts_and_trajectories_checked=True,
        test_as_dev=True, seed=42, recognition_usage="Observed API calls only; cache hits excluded"))
    print(json.dumps(dict(complete=not missing, conditions=len(rows), missing=len(missing))), flush=True)
    if missing and not allow_incomplete:
        raise RuntimeError("Simplified AMOR evaluation is incomplete")


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "qa", "ircot", "check_native", "serve", "serve_es", "index_es", "ready", "stop_services", "report", "sensitivity", "construction_analysis", "prepare_connections", "qa_connections", "report_connections"))
    parser.add_argument("--task", choices=TASK_METRICS)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--variant", choices=("full", *CORE))
    parser.add_argument("--simplified", action="store_true")
    parser.add_argument("--generator-job")
    parser.add_argument("--service-jobs", nargs="+", default=[])
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if args.simplified:
        ROOT = SIMPLE_ROOT
        initialize_simple()
        ircot.ROOT = ircot.SERVICE_ROOT = ROOT / "services"
        ircot.CAPS = CAPS
        ircot.GRAPH_RETRIEVAL = "hybrid"
        if args.phase == "prepare":
            prepare_simple()
        elif args.phase == "qa":
            if not args.model:
                parser.error("qa requires --model")
            components.ROOT = ROOT / "one_shot"
            components.VARIANTS = SIMPLE_VARIANTS
            components.qa(args.model)
        elif args.phase == "ircot":
            if not args.model or args.variant not in ("full", "without_propagation") or not args.generator_job:
                parser.error("simplified ircot requires model, full/without_propagation, generator-job")
            run_iterative(args.model, args.variant, args.generator_job, simplified=True)
        elif args.phase == "report":
            report_simple(args.allow_incomplete)
        elif args.phase == "serve":
            ircot.serve()
        elif args.phase == "serve_es":
            ircot.serve_elasticsearch()
        elif args.phase == "index_es":
            ircot.index_elasticsearch()
        elif args.phase == "ready":
            if not args.generator_job:
                parser.error("ready requires --generator-job")
            wait_for_service(args.generator_job)
        elif args.phase == "check_native":
            print(json.dumps(dict(native_import=True, local_cache=prepare_native_participants())))
        elif args.phase == "stop_services":
            stop_services(args.service_jobs)
        else:
            parser.error("This phase is not defined for the simplified graph")
        return
    if args.phase == "construction_analysis":
        analyze_construction()
        return
    if args.phase == "report_connections":
        report_connections()
        return
    if args.phase in ("prepare_connections", "qa_connections"):
        components.ROOT = CONNECTION_ROOT
        components.VARIANTS = CONNECTIONS
        if args.phase == "prepare_connections":
            for task in ([args.task] if args.task else TASK_METRICS):
                components.prepare(task, edge_controls=True)
        else:
            if not args.model:
                parser.error("qa_connections requires --model")
            components.qa(args.model)
        return
    initialize()
    ircot.ROOT = ROOT / "services"
    ircot.SERVICE_ROOT = ircot.ROOT
    ircot.CAPS = CAPS
    ircot.GRAPH_RETRIEVAL = "hybrid"
    if args.phase == "prepare":
        if args.task == "SH-Doc_QA":
            import unittest
            suite = unittest.defaultTestLoader.loadTestsFromName("optimization.test_graph_construction")
            if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
                raise RuntimeError("Construction tests failed")
        for task in ([args.task] if args.task else TASK_METRICS):
            prepare(task)
    elif args.phase == "qa":
        if not args.model:
            parser.error("qa requires --model")
        qa(args.model)
    elif args.phase == "ircot":
        if not all((args.model, args.variant, args.generator_job)):
            parser.error("ircot requires --model, --variant and --generator-job")
        run_iterative(args.model, args.variant, args.generator_job)
    elif args.phase == "serve":
        ircot.serve()
    elif args.phase == "check_native":
        print(json.dumps(dict(native_import=True, local_cache=prepare_native_participants(), home=os.environ.get("HOME"))), flush=True)
    elif args.phase == "serve_es":
        ircot.serve_elasticsearch()
    elif args.phase == "index_es":
        ircot.index_elasticsearch()
    elif args.phase == "ready":
        if not args.generator_job:
            parser.error("ready requires --generator-job")
        wait_for_service(args.generator_job)
    elif args.phase == "stop_services":
        stop_services(args.service_jobs)
    elif args.phase == "sensitivity":
        audit = json.loads((ROOT / "audit.json").read_text())
        assert audit["native_scores_recomputed"] and audit["questions_and_context_checked"]
        with (ROOT / "qa_results.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == audit["conditions"]
        write_sensitivity({(r["model"], r["task"], r["setting"], r["variant"]): float(r["score"]) for r in rows})
    else:
        report(args.allow_incomplete)


if __name__ == "__main__":
    main()

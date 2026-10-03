"""Online encoder intervention on frozen AMOR memory, with native QA evaluation.

All outputs are new artifacts. Historical stores, graphs, and caches are read-only.
Gold annotations are used by reporting, never candidate generation or ranking.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np


TASKS = ("2WikiMultiHopQA", "LoCoMo")
SUPPORTED_TASKS = ("SH-Doc_QA", "MH-Doc_QA", "FactConsolidation-SH",
                   "FactConsolidation-MH", "LoCoMo", "2WikiMultiHopQA")
READERS = ("Qwen/Qwen3.5-4B", "meta-llama/Llama-3.1-8B-Instruct")
ENCODERS = {
    "qwen": dict(model="Qwen/Qwen3-Embedding-0.6B", reference=True),
    "bge": dict(model="BAAI/bge-m3", revision="5617a9f61b028005a4858fdac845db406aefb181",
                max_length=8192, query_instruction="", reference=False),
    "nv": dict(model="nvidia/NV-Embed-v2", revision="3fa59658547db50a1e8e3346cf057fd0c77ed6ef",
               max_length=32768,
               query_instruction="Instruct: Given a question, retrieve passages that answer the question\nQuery: ",
               reference=False),
}
DEFAULT_RESULTS = Path("/oscar/scratch/zliu328/agent-memory-outputs/optimization_simplified_amor_seed42_20260929")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_save(path, value):
    """Immutable protocol/complete markers; never silently overwrite a changed run."""
    path = Path(path)
    value = json.loads(json.dumps(value))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Existing artifact differs: {path}")
        return
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        try:
            os.link(temporary, path)
        except FileExistsError:
            if json.loads(path.read_text()) != value:
                raise ValueError(f"Concurrent artifact differs: {path}")
    finally:
        Path(temporary).unlink()


class RecognitionAudit:
    """Observe native parsing, with explicit historical-replay or strict policy."""

    def __init__(self, native_filter, path, *, policy="strict"):
        if policy not in ("strict", "historical_replay"):
            raise ValueError("Unknown recognition audit policy")
        self.policy = policy
        self.failures = []
        self.responses = {}
        self.stream = Path(path).open("a")
        infer, parse = native_filter.llm_infer_fn, native_filter.parse_filter

        def checked_infer(*args, **kwargs):
            result = infer(*args, **kwargs)
            if self.policy == "historical_replay" and result[-1] is not True:
                raise RuntimeError("Historical replay must use cached recognition")
            self.stream.write(json.dumps(dict(request=kwargs, response=result), default=str) + "\n")
            self.stream.flush()
            return result

        def checked_parse(response):
            actual = parse(response)
            key = hashlib.sha256(response.encode()).hexdigest()
            try:
                from pydantic import TypeAdapter
                from hipporag.rerank import Fact
                sections = re.split(r"(?m)^\s*\[\[ ## (\w+) ## \]\]\s*$", response)
                values = [sections[i + 1].strip() for i in range(1, len(sections), 2)
                          if sections[i] == "fact_after_filter"]
                if len(values) != 1:
                    raise ValueError("Recognition must contain exactly one fact_after_filter field")
                try:
                    payload = json.loads(values[0])
                except json.JSONDecodeError:
                    payload = ast.literal_eval(values[0])
                validated = TypeAdapter(Fact).validate_python(payload).fact
                if actual != validated:
                    raise ValueError("Native recognition parser disagrees with audit")
                self.responses[key] = dict(status="valid", facts=len(actual))
                return actual
            except Exception as error:
                self.failures.append(str(error))
                self.responses[key] = dict(status="parse_failure", error=str(error))
                # Preserve the existing parser's output. Strict execution fails
                # at check(), whereas replay must reproduce historical outputs.
                return actual

        native_filter.llm_infer_fn = checked_infer
        native_filter.parse_filter = checked_parse

    def check(self):
        if self.failures and self.policy == "strict":
            raise RuntimeError(f"Malformed recognition responses: {self.failures}")

    def close(self):
        self.stream.close()


def validate_vectors(ids, vectors, label):
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError(f"Empty or repeated {label} identities")
    vectors = np.asarray(vectors)
    if vectors.ndim != 2 or vectors.shape[0] != len(ids) or vectors.shape[1] == 0:
        raise ValueError(f"Invalid {label} embedding shape")
    if not np.isfinite(vectors).all():
        raise ValueError(f"Nonfinite {label} embeddings")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, rtol=0, atol=2e-3):
        raise ValueError(f"Unnormalized {label} embeddings")


def validate_bundle(source, bundle):
    dimensions = set()
    for name in ("passage", "fact", "query"):
        keys = source[name + "_ids"]
        if bundle[name + "_ids"].tolist() != keys:
            raise ValueError(f"{name} ID order changed")
        fields = ("query_triple", "query_passage") if name == "query" else (name,)
        for field in fields:
            validate_vectors(keys, bundle[field], field)
            dimensions.add(bundle[field].shape[1])
    if len(dimensions) != 1:
        raise ValueError("Query/fact/passage dimensions differ")


def source_rows(results, task):
    from experiments.recommendation_findings import entries_for
    from experiments.runner import _read_retrieval_records
    paths = entries_for(results, task)
    return list(_read_retrieval_records(paths["full"] / "retrieval.jsonl")), paths


def validate_tasks(tasks):
    tasks = tuple(tasks)
    if not tasks or len(set(tasks)) != len(tasks) or not set(tasks) <= set(SUPPORTED_TASKS):
        raise ValueError("Task scope must be a nonempty, unique subset of the approved six datasets")
    return tasks


def protocol_tasks(root):
    return validate_tasks(json.loads((Path(root) / "protocol.json").read_text())["tasks"])


def freeze_protocol(args):
    tasks = validate_tasks(args.tasks or TASKS)
    protocol = dict(experiment="fixed_memory_online_encoder_robustness", encoders=ENCODERS,
        tasks=list(tasks), readers=list(READERS), source_results=str(args.results), seed=42,
        retrieval_k=[5, 10, 15], qa_k=5, methods=["amor", "without_recommendation", "dense"],
        qa_methods=["amor", "without_recommendation"],
        fixed="original records, extracted/retained facts, relation mapping, final graph, RRF, augmentation policy",
        recomputed="passage/fact/query vectors, recognized candidates, restart scores, augmentation fact ranking",
        qwen_reference="verified original ranking/context replay; historical main QA",
        reporting="all specified encoders and datasets, including negative results",
        recognition_policy="historical reference replay preserves and reports native parse failures; new encoders reject them",
        reused_export=str(args.reuse_export) if args.reuse_export else None,
        encoder_batch_size=8, nv_dtype="float16", bge_dtype="float32",
        native_length_limits=True, test_as_dev=True,
        code_sha256=digest(__file__))
    json_save(args.output / "protocol.json", protocol)


def export(args):
    from experiments.analyze_geometry import GRAPH, QUERIES
    from experiments.recommendation_findings import load_frozen
    from utils.models import release_accelerator_memory
    freeze_protocol(args)
    if args.reuse_export:
        if protocol_tasks(args.reuse_export) != protocol_tasks(args.output):
            raise ValueError("Reused export has a different task scope")
        previous = json.loads((args.reuse_export / "sources.json").read_text())
        entries = []
        for entry in previous["entries"]:
            source_dir = Path(entry["directory"])
            saved = json.loads((source_dir / "complete.json").read_text())
            if digest(source_dir / "source.json") != saved["source_sha256"]:
                raise ValueError("Invalid reused source export")
            source = json.loads((source_dir / "source.json").read_text())
            if digest(source["reference_retrieval"]) != source["reference_retrieval_sha256"]:
                raise ValueError("Reference inputs changed since export")
            suffix = Path(entry["task"]) / entry["group"]
            vectors = args.reuse_export / "vectors/qwen" / suffix
            vm = json.loads((vectors / "complete.json").read_text())
            if vm["source_sha256"] != saved["source_sha256"] or digest(vectors / "embeddings.npz") != vm["embeddings_sha256"]:
                raise ValueError("Invalid reused reference vectors")
            directory = args.output / "sources" / suffix
            directory.mkdir(parents=True, exist_ok=True)
            json_save(directory / "source.json", source)
            json_save(directory / "complete.json", saved)
            destination = args.output / "vectors/qwen" / suffix
            destination.mkdir(parents=True, exist_ok=True)
            vector_file = destination / "embeddings.npz"
            if not vector_file.exists():
                shutil.copy2(vectors / "embeddings.npz", vector_file)
            if digest(vector_file) != vm["embeddings_sha256"]:
                raise ValueError("Reused vector copy differs")
            json_save(destination / "complete.json", vm)
            old_retrieval = args.reuse_export / "retrieval/qwen" / suffix
            new_retrieval = args.output / "retrieval/qwen" / suffix
            if (old_retrieval / "complete.json").exists() and not new_retrieval.exists():
                completed = json.loads((old_retrieval / "complete.json").read_text())
                if not completed["reference_replayed"] or completed["embeddings_sha256"] != vm["embeddings_sha256"]:
                    raise ValueError("Invalid historical reference replay")
                shutil.copytree(old_retrieval, new_retrieval)
            entries.append(dict(entry, directory=str(directory)))
        json_save(args.output / "sources.json", dict(complete=True, entries=entries))
        return
    entries = []
    for task in protocol_tasks(args.output):
        rows, paths = source_rows(args.results, task)
        groups = defaultdict(list)
        for row in rows:
            groups[row.group_id].append(row)
        for group, cases in groups.items():
            directory = args.output / "sources" / task / group
            directory.mkdir(parents=True, exist_ok=True)
            if (directory / "complete.json").exists():
                entries.append(dict(task=task, group=group, directory=str(directory)))
                continue
            memory, guard = load_frozen(task, group, GRAPH,
                args.output / "reference_runtime" / task / group, args.results)
            try:
                engine = memory._memory
                query_ids = list(dict.fromkeys(row.case.question for row in cases))
                source = dict(task=task, group=group, passage_ids=list(engine.passage_node_keys),
                    fact_ids=list(engine.fact_node_keys), query_ids=query_ids,
                    passage_texts=[engine.chunk_embedding_store.get_row(k)["content"] for k in engine.passage_node_keys],
                    fact_texts=[engine.fact_embedding_store.get_row(k)["content"] for k in engine.fact_node_keys],
                    graph_names=engine.graph.vs["name"],
                    graph_edges=engine.graph.get_edgelist(), graph_weights=engine.graph.es["weight"],
                    reference_retrieval=str(paths["full"] / "retrieval.jsonl"),
                    reference_retrieval_sha256=digest(paths["full"] / "retrieval.jsonl"),
                    query_cache_sha256=digest(QUERIES / task / group / "queries.npz"))
                vectors = dict(passage_ids=np.asarray(source["passage_ids"]),
                    fact_ids=np.asarray(source["fact_ids"]), query_ids=np.asarray(query_ids),
                    passage=engine.passage_embeddings, fact=engine.fact_embeddings,
                    query_triple=np.stack([engine.query_to_embedding["triple"][q].reshape(-1) for q in query_ids]),
                    query_passage=np.stack([engine.query_to_embedding["passage"][q].reshape(-1) for q in query_ids]))
                validate_bundle(source, vectors)
                json_save(directory / "source.json", source)
                target = args.output / "vectors/qwen" / task / group
                target.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(target / "embeddings.npz", **vectors)
                json_save(target / "complete.json", dict(encoder="qwen", reused=True,
                    source_sha256=digest(directory / "source.json"),
                    embeddings_sha256=digest(target / "embeddings.npz")))
                guard.check()
                json_save(directory / "complete.json", dict(complete=True, source_sha256=digest(directory / "source.json")))
                entries.append(dict(task=task, group=group, directory=str(directory)))
                print("export", task, group, len(source["passage_ids"]), len(source["fact_ids"]), flush=True)
            finally:
                memory.close()
                del memory
                release_accelerator_memory()
    json_save(args.output / "sources.json", dict(complete=True, entries=entries))


def encode(args):
    """Use each encoder's native implementation, isolated from the retrieval runtime."""
    import importlib.metadata
    import torch
    from huggingface_hub import snapshot_download
    spec = ENCODERS[args.encoder]
    if spec["reference"]:
        raise ValueError("The Qwen reference is exported from its original frozen store")
    snapshot = snapshot_download(spec["model"], revision=spec["revision"])
    if args.encoder == "bge":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(snapshot, device="cuda")
        model.max_seq_length = spec["max_length"]
        tokenizer = model.tokenizer
        def batch_encode(texts, query):
            return model.encode(texts, batch_size=args.batch_size, normalize_embeddings=True,
                show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
    elif args.encoder == "nv":
        from transformers import AutoConfig, AutoModel
        config = AutoConfig.from_pretrained(snapshot, trust_remote_code=True)
        config.text_config._name_or_path = snapshot
        model = AutoModel.from_pretrained(snapshot, config=config, trust_remote_code=True,
            torch_dtype=torch.float16).to("cuda").eval()
        tokenizer = model.tokenizer
        def batch_encode(texts, query):
            with torch.inference_mode():
                output = model.encode(texts, instruction=spec["query_instruction"] if query else "",
                    max_length=spec["max_length"])
            return torch.nn.functional.normalize(output.float(), p=2, dim=1).cpu().numpy()
    else:
        raise ValueError("Unsupported encoder")
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "sentence-transformers")}
    entries = json.loads((args.output / "sources.json").read_text())["entries"]
    for entry in entries:
        source_path = Path(entry["directory"]) / "source.json"
        source = json.loads(source_path.read_text())
        target = args.output / "vectors" / args.encoder / entry["task"] / entry["group"]
        marker = target / ("smoke.json" if args.smoke else "complete.json")
        if marker.exists():
            continue
        target.mkdir(parents=True, exist_ok=True)
        arrays, counts = {}, {}
        started = perf_counter()
        for kind in ("passage", "fact", "query"):
            texts = source["query_ids"] if kind == "query" else source[kind + "_texts"]
            ids = source[kind + "_ids"]
            if args.smoke:
                texts, ids = texts[:8], ids[:8]
            chunks, truncated, longest = [], 0, 0
            prefix = spec["query_instruction"] if kind == "query" else ""
            for start in range(0, len(texts), args.batch_size):
                batch = texts[start:start + args.batch_size]
                lengths = [len(tokenizer.encode(prefix + t, add_special_tokens=True))
                           + (1 if args.encoder == "nv" else 0) for t in batch]
                truncated += sum(n > spec["max_length"] for n in lengths)
                longest = max(longest, *lengths)
                chunks.append(batch_encode(batch, kind == "query"))
                if args.smoke and start == 0:
                    singleton = batch_encode(batch[:1], kind == "query")
                    np.testing.assert_allclose(chunks[-1][:1] @ singleton.T, 1, rtol=0, atol=2e-3)
            values = np.concatenate(chunks, axis=0)
            validate_vectors(ids, values, kind)
            arrays[kind + "_ids"] = np.asarray(ids)
            arrays[kind] = values
            counts[kind] = dict(rows=len(ids), truncated=truncated, maximum_tokens=longest)
        arrays["query_triple"] = arrays.pop("query")
        arrays["query_passage"] = arrays["query_triple"].copy()
        if not args.smoke:
            validate_bundle(source, arrays)
            np.savez_compressed(target / "embeddings.npz", **arrays)
        json_save(marker, dict(encoder=args.encoder, spec=spec, versions=versions, counts=counts,
            source_sha256=digest(source_path), seconds=perf_counter()-started, smoke=args.smoke,
            embeddings_sha256=None if args.smoke else digest(target / "embeddings.npz")))
        print("encode", args.encoder, entry["task"], entry["group"], counts, flush=True)
        if args.smoke:
            break


def install_bundle(engine, source, bundle):
    validate_bundle(source, bundle)
    if engine.fact_node_keys != source["fact_ids"] or engine.passage_node_keys != source["passage_ids"]:
        raise ValueError("Runtime store ordering differs from export")
    if engine.graph.vs["name"] != source["graph_names"]:
        raise ValueError("Graph node identities changed")
    if [list(e) for e in engine.graph.get_edgelist()] != source["graph_edges"]:
        raise ValueError("Graph connectivity changed")
    np.testing.assert_array_equal(engine.graph.es["weight"], source["graph_weights"])
    engine.fact_embeddings = bundle["fact"].copy()
    engine.passage_embeddings = bundle["passage"].copy()
    engine.query_to_embedding = {kind: dict(zip(source["query_ids"], bundle["query_" + kind], strict=True))
                                 for kind in ("triple", "passage")}
    def reject_encoding(*unused_args, **unused_kwargs):
        raise RuntimeError("Frozen embedding bundle missed a query; refusing mixed encoders")
    engine.embedding_model.batch_encode = reject_encoding


def retrieve(args):
    from experiments.analyze_geometry import GRAPH
    from experiments.recommendation_findings import load_frozen
    from experiments.ablate_memory import config_for
    from experiments.ablate_components import PropagationControl, save_rows
    from experiments.runner import _read_retrieval_records
    from optimization.retriever.hipporag import load_optimized_memory, GenerationFailureGuard
    from optimization.retriever.query_fact_context import QueryFactContext
    import igraph as ig
    from utils.models import release_accelerator_memory
    protocol = json.loads((args.results / "protocol.json").read_text())
    entries = json.loads((args.output / "sources.json").read_text())["entries"]
    for entry in entries:
        task, group = entry["task"], entry["group"]
        target = args.output / "retrieval" / args.encoder / task / group
        marker = target / ("smoke.json" if args.smoke else "complete.json")
        if marker.exists():
            continue
        target.mkdir(parents=True, exist_ok=True)
        source_path = Path(entry["directory"]) / "source.json"
        source = json.loads(source_path.read_text())
        vector_dir = args.output / "vectors" / args.encoder / task / group
        vector_manifest = json.loads((vector_dir / "complete.json").read_text())
        if digest(source_path) != vector_manifest["source_sha256"] or digest(vector_dir / "embeddings.npz") != vector_manifest["embeddings_sha256"]:
            raise ValueError("Embedding provenance mismatch")
        runtime = args.output / "runtime" / args.encoder / task / group
        if args.encoder == "qwen":
            memory, guard = load_frozen(task, group, GRAPH, runtime, args.results)
        else:
            if not args.endpoint:
                raise ValueError("A live recognition endpoint is required for new encoders")
            memory = load_optimized_memory(config_for(task, args.endpoint), GRAPH / task / "memory" / group, runtime)
            graph_path = Path(protocol["graph_source"]) / "groups" / task / group / (protocol["graph_variant"] + ".pickle")
            memory._memory.graph = ig.Graph.Read_Pickle(str(graph_path))
            fresh_cache = runtime / "new_encoder_recognition.sqlite"
            fresh_cache.parent.mkdir(parents=True, exist_ok=True)
            memory._generator.cache_file_name = str(fresh_cache)
            guard = GenerationFailureGuard(memory._generator)
        control = None
        recognition = RecognitionAudit(memory._memory.rerank_filter, target / "recognition.jsonl",
            policy="historical_replay" if ENCODERS[args.encoder]["reference"] else "strict")
        try:
            with np.load(vector_dir / "embeddings.npz", allow_pickle=False) as bundle:
                install_bundle(memory._memory, source, bundle)
            control = PropagationControl(memory._memory)
            selector = QueryFactContext(memory._memory, memory.contents)
            all_rows, paths = source_rows(args.results, task)
            rows = [row for row in all_rows if row.group_id == group]
            baseline = {row.case.case_id: row for row in _read_retrieval_records(paths["without_propagation"] / "retrieval.jsonl")}
            if args.smoke:
                rows = rows[:2]
            output_rows, traces = defaultdict(list), []
            for index, row in enumerate(rows):
                query = row.case.question
                for method, disabled in (("amor", False), ("without_recommendation", True)):
                    for k in (5, 10, 15):
                        memory.base.rank_window = k
                        control.begin(disabled=disabled)
                        centers = memory.retrieve(query, k)
                        guard.check()
                        recognition.check()
                        original = [item.metadata["original_source_text"] for item in centers]
                        traces.append(dict(case_id=row.case.case_id, group_id=group, task=task,
                            method=method, k=k, passages=original,
                            ids=[item.metadata["source_passage"] for item in centers]))
                        if k == 5:
                            rendered = selector.render(query, centers)
                            if args.encoder == "qwen":
                                expected = row if method == "amor" else baseline[row.case.case_id]
                                if [i.text for i in rendered] != [i.text for i in expected.retrieved]:
                                    raise ValueError(f"Reference context replay failed: {row.case.case_id}/{method}")
                            output_rows[method].append(replace(row, retrieved=rendered, top_k=k))
                dense_order, dense_scores = memory._memory.dense_passage_retrieval(query)
                for k in (5, 10, 15):
                    keys = [memory._memory.passage_node_keys[i] for i in dense_order[:k]]
                    traces.append(dict(case_id=row.case.case_id, group_id=group, task=task,
                        method="dense", k=k, ids=keys,
                        passages=[memory.contents[key]["original_source_text"] for key in keys]))
                if index % 50 == 0:
                    print("retrieve", args.encoder, task, group, index, len(rows), flush=True)
            suffix = "smoke_" if args.smoke else ""
            for method, records in output_rows.items():
                save_rows(target / (suffix + method) / "retrieval.jsonl", records)
            json_save(target / (suffix + "rankings.json"), traces)
            json_save(target / (suffix + "recognition_audit.json"),
                dict(policy=recognition.policy, responses=recognition.responses))
            json_save(marker, dict(complete=True, questions=len(rows), encoder=args.encoder,
                smoke=args.smoke, graph_fixed=True, reference_replayed=args.encoder == "qwen",
                embeddings_sha256=vector_manifest["embeddings_sha256"]))
        finally:
            recognition.close()
            if control is not None:
                control.close()
            memory.close()
            del memory
            release_accelerator_memory()
        if args.smoke:
            break
    if args.smoke:
        return
    prepared = []
    for task in protocol_tasks(args.output):
        for method in ("amor", "without_recommendation"):
            rows = []
            for entry in entries:
                if entry["task"] == task:
                    path = args.output / "retrieval" / args.encoder / task / entry["group"] / method / "retrieval.jsonl"
                    rows.extend(_read_retrieval_records(path))
            originals, _ = source_rows(args.results, task)
            by_id = {r.case.case_id: r for r in rows}
            if len(by_id) != len(rows) or set(by_id) != {r.case.case_id for r in originals}:
                raise ValueError("Incomplete or duplicated task retrieval")
            ordered = [by_id[r.case.case_id] for r in originals]
            path = args.output / "qa_inputs" / args.encoder / method / task / "retrieval.jsonl"
            save_rows(path, ordered)
            prepared.append(dict(task=task, method=method, questions=len(ordered), path=str(path)))
    json_save(args.output / "qa_inputs" / args.encoder / "prepared.json", dict(complete=True, entries=prepared))


def qa(args):
    from optimization.ircot import Reader, evaluate_task
    from experiments.runner import _read_retrieval_records, _answer_prompt, _official_generation
    from optimization.ircot import SEED, BATCH_SIZE
    from experiments.ablate_components import save_rows
    import random
    if args.model not in READERS or args.encoder == "qwen":
        raise ValueError("New QA uses the two declared readers and new encoders only")
    prepared = json.loads((args.output / "qa_inputs" / args.encoder / "prepared.json").read_text())
    if not prepared["complete"]:
        raise ValueError("Retrieval must finish before QA")
    reader = Reader(args.model)
    try:
        for entry in prepared["entries"]:
            directory = args.output / ("qa_smoke" if args.smoke else "qa_runs") / args.encoder / args.model.replace("/", "_") / entry["method"] / entry["task"]
            directory.mkdir(parents=True, exist_ok=True)
            if (directory / "qa_complete.json").exists():
                continue
            records = list(_read_retrieval_records(Path(entry["path"])))
            if args.smoke:
                # Exercise every native question category, including adversarial LoCoMo.
                categories = dict.fromkeys(row.case.category for row in records)
                records = [next(row for row in records if row.case.category == category)
                           for category in categories]
            save_rows(directory / "retrieval.jsonl", records)
            json_save(directory / "reader.json", reader.metadata)
            rng = random.Random(SEED)
            requests = []
            for row in records:
                messages, _ = _answer_prompt(row.case, row.retrieved, rng)
                requests.append(dict(phase="answer", messages=messages, generation=_official_generation(row.case)))
            for start in range(0, len(requests), BATCH_SIZE):
                if reader.request(phase="check_context", items=requests[start:start+BATCH_SIZE])["overflow"]:
                    raise ValueError("Reader context overflow; no silent truncation is allowed")
            evaluate_task(directory, entry["task"], reader, args.model, args.smoke)
    finally:
        reader.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("export", "encode", "retrieve", "qa"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--reuse-export", type=Path)
    parser.add_argument("--tasks", nargs="+", choices=SUPPORTED_TASKS,
                        help="Explicit dataset scope, frozen during export; other phases read the protocol")
    parser.add_argument("--encoder", choices=tuple(ENCODERS))
    parser.add_argument("--model", choices=READERS)
    parser.add_argument("--endpoint")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch size must be positive")
    if args.phase != "export" and args.encoder is None:
        parser.error("encoder is required outside export")
    if args.phase != "export" and args.tasks is not None:
        parser.error("tasks are specified only during export and frozen thereafter")
    {"export": export, "encode": encode, "retrieve": retrieve, "qa": qa}[args.phase](args)


if __name__ == "__main__":
    main()

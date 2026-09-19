"""Retrieve, evaluate and verify the frozen main graph and its original-graph control."""

import argparse
import gc
import json
import shutil
from pathlib import Path
from time import perf_counter

from experiments.runner import RetrievedCase, _retrieval_record, evaluate_retrieval
from main import _config_from_args, _load_groups, _seed_everything, build_parser


BASE = Path("/oscar/scratch/zliu328/agent-memory-outputs")
SOURCE = BASE / "final_qwen3_30b_seed42_clean_20260910"
MODELS = ("Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-2B",
          "meta-llama/Llama-3.1-8B-Instruct", "google/gemma-3-4b-it")
TASKS = ("SH-Doc QA", "MH-Doc QA", "FactConsolidation-SH", "FactConsolidation-MH", "LoCoMo", "2WikiMultiHopQA")
MODULES = ("fact_consolidation", "graph", "candidate_index", "evidence_context")


class MeasuredFinalAnswer:
    """Record actual QA usage without modifying benchmark generation settings."""

    def __init__(self, model, stream):
        self.model, self.stream = model, stream

    def answer(self, messages, **generation):
        start = perf_counter()
        response = self.model.answer(messages, **generation)
        self.stream.write(json.dumps({**self.model.last_usage, "seconds": perf_counter() - start,
                                      "generation_settings": generation}) + "\n")
        self.stream.flush()
        return response


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def evaluate(args):
    from utils.models import HuggingFaceChatModel
    model = HuggingFaceChatModel(args.evaluation_backbone, max_tokens=2048,
                                dtype="bfloat16", device_map="cuda", seed=42)
    model._ensure_loaded()
    for variant in args.variants:
        directory = args.output_root / variant / args.task.replace(" ", "_")
        complete = json.loads((directory / "retrieval_complete.json").read_text())
        output = directory / "evaluations" / args.evaluation_backbone.replace("/", "_")
        output.mkdir(parents=True, exist_ok=False)
        _seed_everything(42)
        with (output / "qa_usage.jsonl").open("x") as stream:
            summary = evaluate_retrieval(directory / "retrieval.jsonl",
                answer_model=MeasuredFinalAnswer(model, stream), output_dir=output, seed=42)
        with (output / "predictions.jsonl").open() as stream:
            count = sum(1 for _ in stream)
        if count != complete["questions"]:
            raise ValueError("Incomplete evaluation")
        print(json.dumps(dict(task=args.task, model=args.evaluation_backbone, variant=variant,
                              questions=count, summary=summary)), flush=True)


def retrieval_config(args):
    common = build_parser().parse_args([
        "--task", args.task, "--baseline", "hipporag2", "--output-dir", str(args.output_root),
        "--generator-base-url", args.generator_base_url, "--chunk-size", "512",
        "--path", args.path, "--data-root", args.data_root])
    official = json.loads((Path(__file__).resolve().parents[1] / "experiments/configs/hipporag2.json").read_text())
    return _config_from_args(common).hipporag_config(official), common


def completed_group_prefix(rows, groups):
    actual = [(r["group_id"], r["case"]["case_id"], r["case"]["question"]) for r in rows]
    expected = [(g.group_id, c.case_id, c.question) for g in groups for c in g.cases]
    if actual != expected[:len(actual)]:
        raise ValueError("Existing retrieval is not the original task prefix")
    completed, offset = set(), 0
    for group in groups:
        stop = offset + len(group.cases)
        if offset < len(rows) < stop:
            raise ValueError("Resume requires complete source groups, not a partial group")
        if stop <= len(rows):
            completed.add(group.group_id)
        offset = stop
    return completed


def retrieve(args):
    from optimization.retriever.hipporag import CacheMissGuard, GenerationFailureGuard, load_optimized_memory

    config, common = retrieval_config(args)
    groups = list(_load_groups(common))
    cache_root = getattr(args, "recognition_cache_root", None)
    if cache_root is not None:
        cache_root = cache_root / args.task.replace(" ", "_")
        previous = json.loads((cache_root / "settings.json").read_text())
        if previous["config"] != config:
            raise ValueError("Recognition cache reuse requires the same original model and request configuration")
    for variant in args.variants:
        directory = args.output_root / variant / args.task.replace(" ", "_")
        built = json.loads((directory / "build_complete.json").read_text())
        if built["groups"] != [group.group_id for group in groups]:
            raise ValueError("Built graph groups differ from the original loader")
        retrieval_path = directory / "retrieval.jsonl"
        existing = ([json.loads(line) for line in retrieval_path.open()]
                    if args.resume_completed_groups and retrieval_path.exists() else [])
        completed = completed_group_prefix(existing, groups)
        count, provider_calls = len(existing), 0
        if args.resume_completed_groups and (directory / "retrieval_complete.json").exists():
            if count != sum(len(group.cases) for group in groups):
                raise ValueError("Completed retrieval has incomplete records")
            print(json.dumps(dict(task=args.task, variant=variant, preserved_complete=count)), flush=True)
            continue
        if args.allow_generator_calls:
            for group_id in completed:
                with (directory / "runtime" / group_id / "recognition_usage.jsonl").open() as usage:
                    provider_calls += sum(1 for _ in usage)
        with retrieval_path.open("a" if args.resume_completed_groups else "x") as stream:
            for group in groups:
                if group.group_id in completed:
                    continue
                if cache_root is not None:
                    source_cache, = (cache_root / "runtime" / group.group_id / "llm_cache").glob("*.sqlite")
                    if source_cache.with_name(source_cache.name + "-wal").exists():
                        raise ValueError("Use a closed historical cache, not an active SQLite WAL database")
                    target_cache = directory / "runtime" / group.group_id / "llm_cache" / source_cache.name
                    if not target_cache.exists():
                        target_cache.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source_cache, target_cache)
                memory = load_optimized_memory(config, directory / "memory" / group.group_id,
                                               directory / "runtime" / group.group_id)
                usage_stream = None
                try:
                    if args.allow_generator_calls:
                        from baseline.graph_usage import GraphUsageRecorder
                        usage_stream = (directory / "runtime" / group.group_id / "recognition_usage.jsonl").open("x")
                        client = memory._generator.openai_client
                        client.chat.completions.create = GraphUsageRecorder(usage_stream).wrap(
                            client.chat.completions.create, method=variant, stage="recognition", group_id=group.group_id)
                    guard = (GenerationFailureGuard if args.allow_generator_calls else CacheMissGuard)(memory._generator)
                    rows = memory._memory.chunk_embedding_store.get_all_id_to_rows()
                    keys = json.loads((directory / "memory" / group.group_id / "lexical_source_keys.json").read_text())
                    if [rows[key]["content"] for key in keys] != list(dict.fromkeys(group.memory_items)):
                        raise ValueError("Built source order differs from the original loader")
                    for case in group.cases:
                        start = perf_counter()
                        items = tuple(memory.retrieve(case.question, 5))
                        guard.check()
                        row = _retrieval_record(RetrievedCase(group.group_id, case, items, 5, perf_counter() - start))
                        row["retrieval_protocol"] = variant
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                        stream.flush()
                        count += 1
                    write_json(directory / "runtime" / group.group_id / "retrieval_cost.json", memory.efficiency_metrics())
                finally:
                    try:
                        memory.close()
                    finally:
                        if usage_stream is not None:
                            usage_stream.close()
                        memory = guard = client = None
                        gc.collect()
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                if usage_stream is not None:
                    with open(usage_stream.name) as usage:
                        provider_calls += sum(1 for _ in usage)
                print(json.dumps(dict(task=args.task, variant=variant, group=group.group_id, questions=count)), flush=True)
        settings = json.loads((directory / "settings.json").read_text())
        write_json(directory / "settings.json", dict(settings, config=config, full_questions=count,
            additional_generator_calls=provider_calls,
            recognition_cache_source=None if cache_root is None else str(cache_root)))
        write_json(directory / "retrieval_complete.json", dict(questions=count, uses_saved_query_resets=False))


def verify(args):
    from optimization.retriever.hipporag import CacheMissGuard, GenerationFailureGuard, load_optimized_memory

    config, _ = retrieval_config(args)
    for variant in args.variants:
        directory = args.output_root / variant / args.task.replace(" ", "_")
        complete = json.loads((directory / "retrieval_complete.json").read_text())
        rows = [json.loads(line) for line in (directory / "retrieval.jsonl").open()]
        if len(rows) != complete["questions"]:
            raise ValueError("Incomplete retrieval artifact")
        matches = 0
        for group_id in dict.fromkeys(row["group_id"] for row in rows):
            memory = load_optimized_memory(config, directory / "memory" / group_id,
                                           directory / "verification" / group_id)
            guard = (GenerationFailureGuard if args.allow_generator_calls else CacheMissGuard)(memory._generator)
            try:
                for row in rows:
                    if row["group_id"] != group_id:
                        continue
                    items = memory.retrieve(row["case"]["question"], 5)
                    guard.check()
                    if [item.text for item in items] != [item["text"] for item in row["retrieved"]]:
                        raise ValueError(f"Saved graph retrieval mismatch: {group_id}, {row['case']['case_id']}")
                    matches += 1
            finally:
                memory.close()
        write_json(directory / "loaded_graph_verified.json", dict(questions=matches, matched=matches,
            uses_saved_query_resets=False, interface="optimization.retriever.hipporag.load_optimized_memory"))
        print(json.dumps(dict(task=args.task, variant=variant, loaded_graph_matches=matches)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("retrieve", "evaluate", "verify"), required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=("canonical_latest_rrf_window", "original_graph_rrf_window",
                        "statement_projection_loop_free_rrf_window", "statement_projection_loop_free_refined_rrf_window",
                        "statement_projection_loop_free_retained_index_rrf_window",
                        "canonical_latest_retained_index_rrf_window") + tuple(f"without_{module}" for module in MODULES),
                        default=["statement_projection_loop_free_retained_index_rrf_window"])
    parser.add_argument("--evaluation-backbone", choices=MODELS[:2] + MODELS[3:], default=MODELS[0])
    parser.add_argument("--generator-base-url", default="http://127.0.0.1:9/v1")
    parser.add_argument("--allow-generator-calls", action="store_true")
    parser.add_argument("--recognition-cache-root", type=Path,
                        help="Reuse closed historical recognition caches with exactly matching request configuration")
    parser.add_argument("--resume-completed-groups", action="store_true",
                        help="Append only after an intact prefix of complete source groups")
    parser.add_argument("--path", default="/oscar/scratch/zliu328/agent-memory-data/locomo/locomo10.json")
    parser.add_argument("--data-root", default=str(Path(__file__).resolve().parents[1] / "baseline_algorithms/HippoRAG/reproduce/dataset"))
    args = parser.parse_args()
    if len(set(args.variants)) != len(args.variants):
        parser.error("Do not repeat variants")
    _seed_everything(42)
    {"retrieve": retrieve, "evaluate": evaluate, "verify": verify}[args.phase](args)


if __name__ == "__main__":
    main()

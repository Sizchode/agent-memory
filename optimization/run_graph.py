"""Retrieve, evaluate and verify the frozen main graph and its original-graph control."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from experiments.run_anchormem import TASKS
from experiments.run_gap_query_memory import MeasuredFinalAnswer
from experiments.runner import RetrievedCase, _retrieval_record, evaluate_retrieval
from main import _config_from_args, _load_groups, _seed_everything, build_parser


BASE = Path("/oscar/scratch/zliu328/agent-memory-outputs")
SOURCE = BASE / "final_qwen3_30b_seed42_clean_20260910"
MODELS = ("Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-2B")


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


def retrieve(args):
    from optimization.retriever.hipporag import CacheMissGuard, GenerationFailureGuard, load_optimized_memory

    config, common = retrieval_config(args)
    groups = list(_load_groups(common))
    for variant in args.variants:
        directory = args.output_root / variant / args.task.replace(" ", "_")
        built = json.loads((directory / "build_complete.json").read_text())
        if built["groups"] != [group.group_id for group in groups]:
            raise ValueError("Built graph groups differ from the original loader")
        count = 0
        with (directory / "retrieval.jsonl").open("x") as stream:
            for group in groups:
                memory = load_optimized_memory(config, directory / "memory" / group.group_id,
                                               directory / "runtime" / group.group_id)
                guard = (GenerationFailureGuard if args.allow_generator_calls else CacheMissGuard)(memory._generator)
                try:
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
                    memory.close()
                print(json.dumps(dict(task=args.task, variant=variant, group=group.group_id, questions=count)), flush=True)
        settings = json.loads((directory / "settings.json").read_text())
        write_json(directory / "settings.json", dict(settings, config=config, full_questions=count))
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
    parser.add_argument("--variants", nargs="+", choices=("canonical_latest_rrf_window", "original_graph_rrf_window"),
                        default=["canonical_latest_rrf_window"])
    parser.add_argument("--evaluation-backbone", choices=MODELS[:2], default=MODELS[0])
    parser.add_argument("--generator-base-url", default="http://127.0.0.1:9/v1")
    parser.add_argument("--allow-generator-calls", action="store_true")
    parser.add_argument("--path", default="/oscar/scratch/zliu328/agent-memory-data/locomo/locomo10.json")
    parser.add_argument("--data-root", default=str(Path(__file__).resolve().parents[1] / "baseline_algorithms/HippoRAG/reproduce/dataset"))
    args = parser.parse_args()
    if len(set(args.variants)) != len(args.variants):
        parser.error("Do not repeat variants")
    _seed_everything(42)
    {"retrieve": retrieve, "evaluate": evaluate, "verify": verify}[args.phase](args)


if __name__ == "__main__":
    main()

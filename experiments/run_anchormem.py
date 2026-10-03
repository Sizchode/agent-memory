"""Retrieve and evaluate both AnchorMem controls without rebuilding memory."""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
from time import perf_counter

from experiments.runner import RetrievedCase, _retrieval_record, evaluate_retrieval
from main import build_parser, _load_groups, _seed_everything, EMBEDDING_MODEL
from utils.models import HuggingFaceChatModel, release_accelerator_memory


VARIANTS = ("anchormem_dense_top5", "anchormem_official")
TASKS = ("SH-Doc QA", "MH-Doc QA", "FactConsolidation-SH", "FactConsolidation-MH", "LoCoMo", "2WikiMultiHopQA")


def completed_sources(task_dir):
    sources = {}
    for path in sorted(task_dir.glob("*/construction.jsonl")):
        records = [json.loads(line) for line in path.read_text().splitlines() if line]
        for record in records:
            if record["status"] != "completed":
                continue
            group = record["group_id"]
            if group in sources:
                raise ValueError(f"Multiple completed memories for {group}; select one explicitly")
            sources[group] = path.parent / group
    return sources


def retrieve(args):
    from baseline.anchormem_retrieval import AnchorMemRetrieval

    os.environ.setdefault("OPENAI_API_KEY", "unused-local-retrieval")
    common = build_parser().parse_args([
        "--task", args.task, "--baseline", "bm25", "--output-dir", str(args.output_root),
        "--chunk-size", "512", "--path", args.path, "--data-root", args.data_root,
    ])
    groups = list(_load_groups(common))
    slug = args.task.replace(" ", "_")
    sources = completed_sources(args.memory_root / "anchormem" / slug)
    if set(sources) != {g.group_id for g in groups}:
        raise ValueError("Completed memories do not cover exactly the requested benchmark groups")
    with ExitStack() as stack:
        streams, timing_streams = {}, {}
        for variant in VARIANTS:
            directory = args.output_root / variant / slug
            directory.mkdir(parents=True, exist_ok=False)
            streams[variant] = stack.enter_context((directory / "retrieval.jsonl").open("w"))
            timing_streams[variant] = stack.enter_context((directory / "retrieval_efficiency.jsonl").open("w"))
            (directory / "settings.json").write_text(json.dumps({
                "variant": variant, "memory_root": str(args.memory_root), "seed": args.seed,
                "generator_model": args.generator_model, "embedding_model": EMBEDDING_MODEL,
                "final_items": 5 if variant.endswith("top5") else None,
                "official_fact_top_k": 5, "official_event_top_k": 5,
                "passage_metrics": False,
            }, indent=2) + "\n")
        for group in groups:
            model = AnchorMemRetrieval(
                sources[group.group_id], args.output_root / "retrieval_runtime" / slug / group.group_id,
                generator_model=args.generator_model, embedding_model=EMBEDDING_MODEL, seed=args.seed,
            )
            try:
                for case in group.cases:
                    for variant, method in zip(VARIANTS, (model.retrieve_top5, model.retrieve_official), strict=True):
                        start = perf_counter()
                        items = tuple(method(case.question))
                        seconds = perf_counter() - start
                        record = _retrieval_record(RetrievedCase(group.group_id, case, items, 5, seconds))
                        record.update(retrieval_protocol=variant, final_context_items=len(items),
                                      top_k_semantics="final_items" if variant.endswith("top5") else "per_layer")
                        streams[variant].write(json.dumps(record, ensure_ascii=False) + "\n")
                        streams[variant].flush()
                        timing_streams[variant].write(json.dumps({
                            "group_id": group.group_id, "case_id": case.case_id,
                            "retrieval_seconds": seconds, "context_items": len(items),
                            "context_characters": sum(len(i.text) for i in items),
                        }) + "\n")
                        timing_streams[variant].flush()
                usage = model.builder.efficiency_metrics()["generation_client"]
                if usage["client_calls_started"]:
                    raise RuntimeError("Retrieval unexpectedly attempted generation")
                print(json.dumps({"group_id": group.group_id, "questions": len(group.cases),
                                  "status": "completed", "generation_client": usage}), flush=True)
            finally:
                model.close()
                del model
                release_accelerator_memory()


class MeasuredAnswerModel:
    def __init__(self, model, stream):
        self.model, self.stream = model, stream

    def answer(self, messages, **kwargs):
        start = perf_counter()
        tokens = self.model._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, enable_thinking=False,
            return_dict=True, return_tensors="pt",
        )
        result = self.model.answer(messages, **kwargs)
        self.stream.write(json.dumps({"input_tokens": int(tokens["input_ids"].shape[-1]), "output_characters": len(result),
                                      "answer_seconds": perf_counter() - start,
                                      "generation_settings": kwargs}) + "\n")
        self.stream.flush()
        return result


def evaluate(args):
    slug = args.task.replace(" ", "_")
    model = HuggingFaceChatModel(args.evaluation_backbone, max_tokens=2048,
                                dtype="bfloat16", device_map="cuda", seed=args.seed)
    model._ensure_loaded()
    for variant in VARIANTS:
        # Each setting gets the same initial RNG state and question order.
        _seed_everything(args.seed)
        directory = args.output_root / variant / slug
        output = directory / "evaluations" / args.evaluation_backbone.replace("/", "_")
        output.mkdir(parents=True, exist_ok=False)
        with (output / "qa_usage.jsonl").open("w") as stream:
            summary = evaluate_retrieval(
                directory / "retrieval.jsonl", answer_model=MeasuredAnswerModel(model, stream),
                output_dir=output, seed=args.seed, report_passage_metrics=False,
            )
        print(json.dumps({"variant": variant, "task": args.task, "summary": summary}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["retrieve", "evaluate"], required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--memory-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--generator-model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--evaluation-backbone", choices=["Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-2B"])
    parser.add_argument("--path", default="/oscar/scratch/zliu328/agent-memory-data/locomo/locomo10.json")
    parser.add_argument("--data-root", default=str(Path(__file__).resolve().parents[1] / "baseline_algorithms/HippoRAG/reproduce/dataset"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.phase == "evaluate" and not args.evaluation_backbone:
        parser.error("--evaluation-backbone is required for evaluation")
    _seed_everything(args.seed)
    (retrieve if args.phase == "retrieve" else evaluate)(args)


if __name__ == "__main__":
    main()

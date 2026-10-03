"""Build/reuse CatRAG memory on the existing full benchmark task definitions."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

from baseline.catrag_memory import CatRAGMemory
from experiments.run_anchormem import TASKS
from experiments.run_gap_query_memory import MeasuredFinalAnswer
from experiments.runner import RetrievedCase, _retrieval_record, evaluate_retrieval
from main import EMBEDDING_MODEL, _load_groups, _seed_everything, build_parser
from utils.models import HuggingFaceChatModel, release_accelerator_memory


def run(args):
    slug = args.task.replace(" ", "_")
    directory = args.output_root / "catrag_top5" / slug
    _seed_everything(42)
    if args.phase == "evaluate":
        output = directory / "evaluations" / args.evaluation_backbone.replace("/", "_")
        if not (directory / "retrieval_complete.json").is_file():
            raise RuntimeError("Complete retrieval is required before benchmark evaluation")
        output.mkdir(parents=True, exist_ok=False)
        model = HuggingFaceChatModel(args.evaluation_backbone, max_tokens=2048,
                                    dtype="bfloat16", device_map="cuda", seed=42)
        with (output / "qa_usage.jsonl").open("x") as stream:
            summary = evaluate_retrieval(directory / "retrieval.jsonl",
                answer_model=MeasuredFinalAnswer(model, stream), output_dir=output, seed=42)
        print(json.dumps(summary), flush=True)
        return

    common = build_parser().parse_args([
        "--task", args.task, "--baseline", "bm25", "--output-dir", str(directory),
        "--chunk-size", "512", "--path", args.path, "--data-root", args.data_root])
    groups = list(_load_groups(common))
    directory.mkdir(parents=True, exist_ok=False)
    memory_task = (args.memory_root or args.output_root) / "catrag_top5" / slug / "memory"
    settings = dict(task=args.task, method="catrag", seed=42, thinking=False,
                    final_top_k=5, generator_model=args.generator_model,
                    embedding_model=EMBEDDING_MODEL, phase=args.phase,
                    memory_root=str(memory_task), group_count=len(groups),
                    question_count=sum(len(g.cases) for g in groups),
                    chunk_size=512, locomo_path=args.path, data_root=args.data_root)
    (directory / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    os.environ.setdefault("OPENAI_API_KEY", "unused-local")
    count = 0
    with (directory / "retrieval.jsonl").open("x") as retrieval, (directory / "usage.jsonl").open("x") as usage:
        for group in groups:
            memory = CatRAGMemory(memory_task / group.group_id, usage,
                group_id=group.group_id, generator_model=args.generator_model,
                generator_base_url=args.generator_base_url, embedding_model=EMBEDDING_MODEL,
                load_existing=args.phase == "retrieve-existing")
            try:
                if args.phase == "build-retrieve":
                    memory.build(group.memory_items)
                for case in group.cases:
                    start = perf_counter()
                    items = tuple(memory.retrieve(case.question, 5, case_id=case.case_id))
                    record = _retrieval_record(RetrievedCase(group.group_id, case, items, 5,
                                                           perf_counter() - start))
                    retrieval.write(json.dumps(record, ensure_ascii=False) + "\n")
                    retrieval.flush()
                    count += 1
                print(json.dumps(dict(group_id=group.group_id, completed_questions=count)), flush=True)
            finally:
                memory.close()
                del memory
                release_accelerator_memory()
    (directory / "retrieval_complete.json").write_text(json.dumps(dict(questions=count)) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("build-retrieve", "retrieve-existing", "evaluate"), required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--memory-root", type=Path)
    parser.add_argument("--generator-model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--generator-base-url", default="http://localhost:8000/v1")
    parser.add_argument("--evaluation-backbone", choices=("Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-2B"))
    parser.add_argument("--path", default="/oscar/scratch/zliu328/agent-memory-data/locomo/locomo10.json")
    parser.add_argument("--data-root", default=str(Path(__file__).resolve().parents[1] / "baseline_algorithms/HippoRAG/reproduce/dataset"))
    args = parser.parse_args()
    if args.phase == "evaluate" and args.evaluation_backbone is None:
        parser.error("--evaluation-backbone required for evaluation")
    if args.phase == "retrieve-existing" and args.memory_root is None:
        parser.error("--memory-root required for existing memory")
    run(args)


if __name__ == "__main__":
    main()

"""Command-line entry point for MemoryAgentBench data inspection."""

import argparse

from baseline import KeywordMemory
from dataset_loader import (
    TaskName,
    load_hipporag2_dataset,
    load_locomo,
    load_memory_agent_bench,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        required=True,
        choices=[task.value for task in TaskName] + ["LoCoMo", "MuSiQue", "2WikiMultiHopQA", "HotpotQA"],
    )
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--max-contexts", type=int, default=None)
    parser.add_argument("--path", help="LoCoMo JSON path")
    parser.add_argument("--data-root", help="HippoRAG 2 reproduce/dataset directory")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.task == "LoCoMo":
        if not args.path:
            raise SystemExit("--path is required for LoCoMo")
        conversations = load_locomo(args.path)
        print(f"conversations={len(conversations)}")
        print(f"questions={sum(len(item.questions) for item in conversations)}")
        return
    if args.task in {"MuSiQue", "2WikiMultiHopQA", "HotpotQA"}:
        if not args.data_root:
            raise SystemExit("--data-root is required for HippoRAG 2 datasets")
        dataset_name = {"MuSiQue": "musique", "2WikiMultiHopQA": "2wikimultihopqa", "HotpotQA": "hotpotqa"}[args.task]
        queries = load_hipporag2_dataset(args.data_root, dataset_name)
        print(f"dataset={dataset_name} queries={len(queries)}")
        return

    samples = load_memory_agent_bench(args.task, chunk_size=args.chunk_size, max_contexts=args.max_contexts)
    print(f"contexts={len(samples)}")
    for context_index, sample in enumerate(samples):
        memory = KeywordMemory().build(sample.chunks)
        print(
            f"context={context_index} source={sample.source} "
            f"chunks={len(sample.chunks)} questions={len(sample.question_answers)} "
            f"first_retrieval_chars={len(memory.retrieve(sample.question_answers[0].question))}"
        )


if __name__ == "__main__":
    main()

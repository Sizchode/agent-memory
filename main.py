"""Command-line entry point for MemoryAgentBench data inspection."""

import argparse

from baseline import KeywordMemory
from dataset_loader import TaskName, load_memory_agent_bench


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=[task.value for task in TaskName])
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--max-contexts", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
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

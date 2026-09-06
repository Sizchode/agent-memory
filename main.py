"""Experiment entry point and the single source of shared comparison controls."""

import argparse
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from baseline import BM25Baseline
from dataset_loader import (
    TaskName,
    load_hipporag2_dataset,
    load_locomo,
    load_memory_agent_bench,
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Resources held fixed whenever an algorithm invokes the corresponding service."""

    llm_model: str
    embedding_model: str
    embedding_dimensions: int
    input_chunk_size: int
    final_retrieval_top_k: int
    answer_max_tokens: int
    internal_max_tokens: int
    temperature: float = 0.0
    llm_base_url: str | None = None
    embedding_base_url: str | None = None
    answer_system_prompt: str = "Answer only from the supplied evidence. Return only the answer."

    def __post_init__(self) -> None:
        if not self.llm_model or not self.embedding_model:
            raise ValueError("llm_model and embedding_model are required")
        if self.embedding_dimensions <= 0:
            raise ValueError("embedding_dimensions must be positive")
        if min(self.input_chunk_size, self.final_retrieval_top_k, self.answer_max_tokens, self.internal_max_tokens) <= 0:
            raise ValueError("all experiment budgets must be positive")

    def hipporag_config(self, official_config: dict[str, Any]) -> dict[str, Any]:
        """Apply only shared service choices; retain HippoRAG graph/OpenIE settings."""
        result = deepcopy(official_config)
        result.update(
            llm_name=self.llm_model,
            embedding_model_name=self.embedding_model,
            retrieval_top_k=self.final_retrieval_top_k,
            qa_top_k=self.final_retrieval_top_k,
            max_new_tokens=self.internal_max_tokens,
        )
        if self.llm_base_url is not None:
            result["llm_base_url"] = self.llm_base_url
        if self.embedding_base_url is not None:
            result["embedding_base_url"] = self.embedding_base_url
        return result

    def lightmem_config(self, official_config: dict[str, Any]) -> dict[str, Any]:
        """Apply shared model and generation budgets without changing LightMem stages."""
        result = deepcopy(official_config)
        result.setdefault("memory_manager", {}).setdefault("configs", {}).update(
            model=self.llm_model,
            max_tokens=self.internal_max_tokens,
            temperature=self.temperature,
        )
        result.setdefault("text_embedder", {}).setdefault("configs", {}).update(
            model=self.embedding_model,
            embedding_dims=self.embedding_dimensions,
        )
        for name in ("embedding_retriever", "summary_retriever"):
            if name in result:
                result[name].setdefault("configs", {})["embedding_model_dims"] = self.embedding_dimensions
        return result

    def mem0_config(self, official_config: dict[str, Any]) -> dict[str, Any]:
        """Apply shared model and generation budgets without replacing Mem0's policy."""
        result = deepcopy(official_config)
        result.setdefault("llm", {}).setdefault("config", {}).update(
            model=self.llm_model,
            temperature=self.temperature,
            max_tokens=self.internal_max_tokens,
        )
        result.setdefault("embedder", {}).setdefault("config", {})["model"] = self.embedding_model
        if self.llm_base_url is not None:
            result["llm"]["config"]["openai_base_url"] = self.llm_base_url
        return result

    def magma_arguments(self) -> dict[str, str]:
        """Return MAGMA's supported shared model arguments; retain its graph settings."""
        return {"llm_model": self.llm_model, "embedding_model": self.embedding_model}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        required=True,
        choices=[task.value for task in TaskName] + ["LoCoMo", "MuSiQue", "2WikiMultiHopQA", "HotpotQA"],
    )
    parser.add_argument("--chunk-size", type=int, default=4096, help="Shared source chunking budget.")
    parser.add_argument("--top-k", type=int, default=5, help="Shared final evidence budget.")
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
        memory = BM25Baseline()
        memory.build(sample.chunks)
        print(
            f"context={context_index} source={sample.source} "
            f"chunks={len(sample.chunks)} questions={len(sample.question_answers)} "
            f"first_retrieval_chars={len(memory.retrieve(sample.question_answers[0].question, args.top_k)[0].text)}"
        )


if __name__ == "__main__":
    main()

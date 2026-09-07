"""Run a controlled memory benchmark: data loading, retrieval, QA, and deterministic evaluation."""

import argparse
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from baseline import BM25Baseline, DenseRetrievalBaseline, HippoRAG2Baseline, LightMemBaseline, Mem0Baseline
from baseline.base import MemoryBaseline
from dataset_loader import TaskName, load_hipporag2_dataset, load_locomo, load_memory_agent_bench
from experiments.runner import (
    MemoryGroup,
    hipporag_groups,
    locomo_groups,
    memory_agent_bench_groups,
    run_groups,
)
from utils.models import HuggingFaceChatModel, HuggingFaceEmbedder, ModelEndpoint


BENCHMARKS = [task.value for task in TaskName] + ["LoCoMo", "MuSiQue", "2WikiMultiHopQA", "HotpotQA"]
BASELINES = ["bm25", "dense", "lightmem", "hipporag2", "mem0"]
EVALUATION_BACKBONES = [
    "Qwen/Qwen3.5-35B-A3B",
    "Qwen/Qwen3.5-27B",
    "google/gemma-4-12B-it",
    "google/gemma-4-26B-A4B-it",
]


@dataclass(frozen=True)
class ExperimentConfig:
    """Comparison controls, with generator and evaluation roles kept separate."""

    generator: ModelEndpoint
    embedding_model: str
    embedding_dimensions: int
    input_chunk_size: int
    final_retrieval_top_k: int
    answer_max_tokens: int
    internal_max_tokens: int
    temperature: float = 0.0
    answer_system_prompt: str = "Answer only from the supplied evidence. Return only the answer."

    def __post_init__(self) -> None:
        if min(self.embedding_dimensions, self.input_chunk_size, self.final_retrieval_top_k, self.answer_max_tokens, self.internal_max_tokens) <= 0:
            raise ValueError("all resource budgets must be positive")

    def hipporag_config(self, official_config: dict[str, Any]) -> dict[str, Any]:
        result = deepcopy(official_config)
        result.update(
            llm_name=self.generator.model,
            llm_base_url=self.generator.base_url,
            embedding_model_name=f"Transformers/{self.embedding_model}",
            retrieval_top_k=self.final_retrieval_top_k,
            qa_top_k=self.final_retrieval_top_k,
            max_new_tokens=self.internal_max_tokens,
            temperature=self.temperature,
        )
        return result

    def lightmem_config(self, official_config: dict[str, Any]) -> dict[str, Any]:
        result = deepcopy(official_config)
        result.setdefault("memory_manager", {}).setdefault("configs", {}).update(
            model=self.generator.model,
            api_key=self.generator.api_key(),
            openai_base_url=self.generator.base_url,
            max_tokens=self.internal_max_tokens,
            temperature=self.temperature,
        )
        result["text_embedder"]["model_name"] = "huggingface"
        result.setdefault("text_embedder", {}).setdefault("configs", {}).update(
            model=self.embedding_model,
            embedding_dims=self.embedding_dimensions,
            model_kwargs={"device": "cuda"},
        )
        return result

    def mem0_config(self, official_config: dict[str, Any]) -> dict[str, Any]:
        result = deepcopy(official_config)
        result.setdefault("llm", {}).setdefault("config", {}).update(
            model=self.generator.model,
            api_key=self.generator.api_key(),
            openai_base_url=self.generator.base_url,
            temperature=self.temperature,
            max_tokens=self.internal_max_tokens,
        )
        result["embedder"]["provider"] = "huggingface"
        result.setdefault("embedder", {}).setdefault("config", {}).update(
            model=self.embedding_model,
            embedding_dims=self.embedding_dimensions,
            model_kwargs={"device": "cuda"},
        )
        result.setdefault("vector_store", {}).setdefault("config", {}).setdefault("embedding_model_dims", self.embedding_dimensions)
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=BENCHMARKS)
    parser.add_argument("--baseline", required=True, choices=BASELINES)
    parser.add_argument("--output-dir", required=True, help="Directory for predictions.jsonl and summary.json.")
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-contexts", type=int)
    parser.add_argument("--max-queries", type=int, default=1000)
    parser.add_argument("--path", help="Official LoCoMo JSON path.")
    parser.add_argument("--data-root", help="HippoRAG 2 reproduce/dataset directory.")
    parser.add_argument("--official-config", help="JSON config retained from the selected official baseline.")
    parser.add_argument("--generator-model", default="deepseek-v4-flash-0731")
    parser.add_argument("--generator-base-url", default="https://api.deepseek.com")
    parser.add_argument("--generator-api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--evaluation-backbone", required=True, choices=EVALUATION_BACKBONES)
    parser.add_argument("--evaluation-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--evaluation-device-map", default="auto")
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embedding-dimensions", type=int, default=1024)
    parser.add_argument("--answer-max-tokens", type=int, default=256)
    parser.add_argument("--internal-max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = _config_from_args(args)
    groups = _load_groups(args)
    official_config = _read_config(args.official_config)
    answer_model = HuggingFaceChatModel(
        args.evaluation_backbone,
        max_tokens=config.answer_max_tokens,
        dtype=args.evaluation_dtype,
        device_map=args.evaluation_device_map,
    )
    summary = run_groups(
        groups,
        create_baseline=lambda group: _create_baseline(args.baseline, config, official_config, group, args.output_dir),
        answer_model=answer_model,
        top_k=config.final_retrieval_top_k,
        output_dir=args.output_dir,
        system_prompt=config.answer_system_prompt,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        generator=ModelEndpoint(args.generator_model, args.generator_base_url, args.generator_api_key_env),
        embedding_model=args.embedding_model,
        embedding_dimensions=args.embedding_dimensions,
        input_chunk_size=args.chunk_size,
        final_retrieval_top_k=args.top_k,
        answer_max_tokens=args.answer_max_tokens,
        internal_max_tokens=args.internal_max_tokens,
        temperature=args.temperature,
    )


def _load_groups(args: argparse.Namespace):
    if args.task == "LoCoMo":
        if not args.path:
            raise SystemExit("--path is required for LoCoMo")
        return locomo_groups(load_locomo(args.path))
    external = {"MuSiQue": "musique", "2WikiMultiHopQA": "2wikimultihopqa", "HotpotQA": "hotpotqa"}
    if args.task in external:
        if not args.data_root:
            raise SystemExit("--data-root is required for HippoRAG 2 datasets")
        return hipporag_groups(load_hipporag2_dataset(args.data_root, external[args.task], max_queries=args.max_queries))
    return memory_agent_bench_groups(load_memory_agent_bench(args.task, chunk_size=args.chunk_size, max_contexts=args.max_contexts))


def _create_baseline(kind: str, config: ExperimentConfig, official_config: dict[str, Any], group: MemoryGroup, output_dir: str) -> MemoryBaseline:
    if kind == "bm25":
        return BM25Baseline()
    if kind == "dense":
        return DenseRetrievalBaseline(HuggingFaceEmbedder(config.embedding_model))
    if kind == "lightmem":
        resolved = config.lightmem_config(_required_official_config(kind, official_config))
        resolved["embedding_retriever"]["configs"].update(
            collection_name=group.group_id,
            path=str(Path(output_dir) / "lightmem_indices" / group.group_id),
        )
        return LightMemBaseline(resolved)
    if kind == "hipporag2":
        save_dir = Path(output_dir) / "hipporag_indices" / group.group_id
        return HippoRAG2Baseline(config.hipporag_config(_required_official_config(kind, official_config)), save_dir)
    if kind == "mem0":
        resolved = config.mem0_config(_required_official_config(kind, official_config))
        resolved["vector_store"]["config"]["path"] = str(Path(output_dir) / "mem0_indices" / group.group_id)
        return Mem0Baseline(resolved, group.group_id)
    raise ValueError(f"unsupported baseline {kind!r}")


def _required_official_config(kind: str, config: dict[str, Any]) -> dict[str, Any]:
    if not config:
        raise ValueError(f"--official-config is required for {kind}; it preserves the official algorithm-specific settings while shared services are overridden centrally")
    return config


def _read_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    with Path(path).open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("--official-config must contain a JSON object")
    return payload


if __name__ == "__main__":
    main()

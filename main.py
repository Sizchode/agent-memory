"""Run a controlled memory benchmark: data loading, retrieval, QA, and deterministic evaluation."""

import argparse
from copy import deepcopy
from dataclasses import dataclass
from itertools import islice
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
    evaluate_retrieval,
    retrieve_existing_memory_groups,
    retrieve_groups,
    run_groups,
)
from utils.models import HuggingFaceChatModel, HuggingFaceEmbedder, ModelEndpoint
from utils.prompts import DOCUMENT_FACT_EXTRACTION_INSTRUCTIONS, DOCUMENT_FACT_EXTRACTION_PROMPT


BENCHMARKS = [task.value for task in TaskName] + ["LoCoMo", "2WikiMultiHopQA"]
BASELINES = ["bm25", "dense", "lightmem", "hipporag2", "mem0"]
EVALUATION_BACKBONES = [
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-2B",
    "meta-llama/Llama-3.1-8B-Instruct",
]
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"


@dataclass(frozen=True)
class ExperimentConfig:
    """Comparison controls, with generator and evaluation roles kept separate."""

    generator: ModelEndpoint
    embedding_model: str
    embedding_dimensions: int
    input_chunk_size: int
    final_retrieval_top_k: int
    answer_max_tokens: int
    lightmem_generator_max_tokens: int
    mem0_generator_max_tokens: int
    hipporag_ner_max_tokens: int
    hipporag_triple_max_tokens: int
    seed: int
    temperature: float = 0.0

    def __post_init__(self) -> None:
        budgets = (
            self.embedding_dimensions,
            self.input_chunk_size,
            self.final_retrieval_top_k,
            self.answer_max_tokens,
            self.lightmem_generator_max_tokens,
            self.mem0_generator_max_tokens,
            self.hipporag_ner_max_tokens,
            self.hipporag_triple_max_tokens,
        )
        if min(budgets) <= 0:
            raise ValueError("all resource budgets must be positive")

    def hipporag_config(self, official_config: dict[str, Any]) -> dict[str, Any]:
        result = deepcopy(official_config)
        result.update(
            llm_name=self.generator.model,
            llm_base_url=self.generator.base_url,
            embedding_model_name=f"Transformers/{self.embedding_model}",
            retrieval_top_k=self.final_retrieval_top_k,
            qa_top_k=self.final_retrieval_top_k,
            max_new_tokens=self.hipporag_triple_max_tokens,
            openie_ner_max_tokens=self.hipporag_ner_max_tokens,
            openie_triple_max_tokens=self.hipporag_triple_max_tokens,
            temperature=self.temperature,
        )
        return result

    def lightmem_config(self, official_config: dict[str, Any]) -> dict[str, Any]:
        result = deepcopy(official_config)
        result.setdefault("memory_manager", {})["model_name"] = "vllm"
        manager_config = result.setdefault("memory_manager", {}).setdefault("configs", {})
        manager_config.update(
            model=self.generator.model,
            api_key=self.generator.api_key(),
            max_tokens=self.lightmem_generator_max_tokens,
            temperature=self.temperature,
            do_sample=False,
            vllm_base_url=self.generator.base_url,
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
        result.setdefault("llm", {})["provider"] = "vllm"
        llm_config = result.setdefault("llm", {}).setdefault("config", {})
        llm_config.update(
            model=self.generator.model,
            api_key=self.generator.api_key(),
            temperature=self.temperature,
            max_tokens=self.mem0_generator_max_tokens,
            vllm_base_url=self.generator.base_url,
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
    parser.add_argument("--phase", choices=["all", "retrieve", "retrieve-existing", "evaluate"], default="all")
    parser.add_argument(
        "--memory-input-dir",
        help="Existing baseline task directory containing generated indices; required by retrieve-existing.",
    )
    parser.add_argument("--retrieval-input", help="retrieval.jsonl produced by the retrieval phase.")
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-contexts", type=int)
    parser.add_argument("--group-start", type=int, default=0, help="First already-numbered memory group to process.")
    parser.add_argument("--group-stop", type=int, help="Exclusive last already-numbered memory group to process.")
    parser.add_argument("--max-queries", type=int, default=1000)
    parser.add_argument("--path", help="Official LoCoMo JSON path.")
    parser.add_argument("--data-root", help="HippoRAG 2 reproduce/dataset directory.")
    parser.add_argument("--official-config", help="JSON config retained from the selected official baseline.")
    parser.add_argument("--generator-model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--generator-base-url")
    parser.add_argument("--generator-api-key-env", default="VLLM_API_KEY")
    parser.add_argument("--evaluation-backbone", choices=EVALUATION_BACKBONES)
    parser.add_argument("--evaluation-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--evaluation-device-map", default="auto")
    parser.add_argument("--embedding-dimensions", type=int, default=1024)
    parser.add_argument("--answer-max-tokens", type=int, default=2048, help="Fallback only; each released benchmark protocol supplies its official per-question limit.")
    parser.add_argument("--lightmem-generator-max-tokens", type=int, default=16000)
    parser.add_argument("--mem0-generator-max-tokens", type=int, default=2000)
    parser.add_argument("--hipporag-ner-max-tokens", type=int, default=512)
    parser.add_argument("--hipporag-triple-max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _seed_everything(args.seed)
    config = _config_from_args(args)
    official_config = _read_config(args.official_config)
    retrieval_path = Path(args.retrieval_input) if args.retrieval_input else Path(args.output_dir) / "retrieval.jsonl"
    if args.phase == "retrieve":
        retrieve_groups(
            _load_groups(args),
            create_baseline=lambda group: _create_baseline(args.baseline, config, official_config, group, args.output_dir),
            top_k=config.final_retrieval_top_k,
            output_path=retrieval_path,
        )
        return
    if args.phase == "retrieve-existing":
        if args.baseline in {"bm25", "dense"}:
            raise SystemExit("retrieve-existing applies only to generated-memory baselines: lightmem, hipporag2, and mem0")
        if not args.memory_input_dir:
            raise SystemExit("--memory-input-dir is required for retrieve-existing")
        memory_input_dir = Path(args.memory_input_dir)
        output_dir = Path(args.output_dir)
        if memory_input_dir.resolve() == output_dir.resolve():
            raise SystemExit("--output-dir must differ from --memory-input-dir so the generated memory remains an input")
        groups = _load_groups(args)
        retrieve_existing_memory_groups(
            groups,
            create_baseline=lambda group: _create_existing_baseline(
                args.baseline,
                config,
                official_config,
                group,
                memory_input_dir,
            ),
            top_k=config.final_retrieval_top_k,
            output_path=retrieval_path,
        )
        return
    if not args.evaluation_backbone:
        raise SystemExit("--evaluation-backbone is required for all/evaluate phases")
    answer_model = HuggingFaceChatModel(
        args.evaluation_backbone,
        max_tokens=config.answer_max_tokens,
        dtype=args.evaluation_dtype,
        device_map=args.evaluation_device_map,
        seed=config.seed,
    )
    if args.phase == "evaluate":
        summary = evaluate_retrieval(
            retrieval_path,
            answer_model=answer_model,
            output_dir=args.output_dir,
            seed=config.seed,
        )
    else:
        summary = run_groups(
            _load_groups(args),
            create_baseline=lambda group: _create_baseline(args.baseline, config, official_config, group, args.output_dir),
            answer_model=answer_model,
            top_k=config.final_retrieval_top_k,
            output_dir=args.output_dir,
            seed=config.seed,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _seed_everything(seed: int) -> None:
    """Set the experiment seed before loading data or models."""
    import random

    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        generator=ModelEndpoint(args.generator_model, args.generator_base_url, args.generator_api_key_env),
        embedding_model=EMBEDDING_MODEL,
        embedding_dimensions=args.embedding_dimensions,
        input_chunk_size=args.chunk_size,
        final_retrieval_top_k=args.top_k,
        answer_max_tokens=args.answer_max_tokens,
        lightmem_generator_max_tokens=args.lightmem_generator_max_tokens,
        mem0_generator_max_tokens=args.mem0_generator_max_tokens,
        hipporag_ner_max_tokens=args.hipporag_ner_max_tokens,
        hipporag_triple_max_tokens=args.hipporag_triple_max_tokens,
        seed=args.seed,
        temperature=args.temperature,
    )


def _load_groups(args: argparse.Namespace):
    if args.group_start < 0 or (args.group_stop is not None and args.group_stop <= args.group_start):
        raise SystemExit("--group-start must be non-negative and --group-stop must be greater than it")
    if args.task == "LoCoMo":
        if not args.path:
            raise SystemExit("--path is required for LoCoMo")
        groups = locomo_groups(load_locomo(args.path))
    else:
        external = {"2WikiMultiHopQA": "2wikimultihopqa"}
        if args.task in external:
            if not args.data_root:
                raise SystemExit("--data-root is required for HippoRAG 2 datasets")
            groups = hipporag_groups(load_hipporag2_dataset(args.data_root, external[args.task], max_queries=args.max_queries))
        else:
            samples = load_memory_agent_bench(args.task, chunk_size=args.chunk_size, max_contexts=args.max_contexts)
            return memory_agent_bench_groups(
                samples[args.group_start : args.group_stop],
                start_index=args.group_start,
            )
    return islice(groups, args.group_start, args.group_stop)


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
        return LightMemBaseline(
            resolved,
            memory_timestamps=group.memory_timestamps,
            dialogue_turns=group.dialogue_turns,
            extraction_prompt=None if group.memory_timestamps else DOCUMENT_FACT_EXTRACTION_PROMPT,
        )
    if kind == "hipporag2":
        save_dir = Path(output_dir) / "hipporag_indices" / group.group_id
        return HippoRAG2Baseline(
            config.hipporag_config(_required_official_config(kind, official_config)),
            save_dir,
        )
    if kind == "mem0":
        resolved = config.mem0_config(_required_official_config(kind, official_config))
        resolved["vector_store"]["config"]["path"] = str(Path(output_dir) / "mem0_indices" / group.group_id)
        if not group.memory_timestamps:
            resolved["custom_instructions"] = DOCUMENT_FACT_EXTRACTION_INSTRUCTIONS
        return Mem0Baseline(
            resolved,
            group.group_id,
            dialogue_roles=tuple(turn.role for turn in group.dialogue_turns),
        )
    raise ValueError(f"unsupported baseline {kind!r}")


def _create_existing_baseline(
    kind: str,
    config: ExperimentConfig,
    official_config: dict[str, Any],
    group: MemoryGroup,
    memory_input_dir: str | Path,
) -> MemoryBaseline:
    """Open one generated memory store without running its build operation."""

    storage_names = {
        "lightmem": "lightmem_indices",
        "hipporag2": "hipporag_indices",
        "mem0": "mem0_indices",
    }
    storage_root = Path(memory_input_dir) / storage_names[kind] / group.group_id
    if not storage_root.is_dir() or not any(path.is_file() for path in storage_root.rglob("*")):
        raise FileNotFoundError(f"generated memory store is missing or empty: {storage_root}")
    if kind == "lightmem":
        resolved = config.lightmem_config(_required_official_config(kind, official_config))
        resolved["embedding_retriever"]["configs"].update(
            collection_name=group.group_id,
            path=str(storage_root),
            # LightMem's Qdrant wrapper treats on_disk=False as a request to
            # delete an existing local path.  Existing-memory retrieval must
            # preserve and open that path instead.
            on_disk=True,
        )
        return LightMemBaseline(
            resolved,
            memory_timestamps=group.memory_timestamps,
            dialogue_turns=group.dialogue_turns,
            extraction_prompt=None if group.memory_timestamps else DOCUMENT_FACT_EXTRACTION_PROMPT,
        )
    return _create_baseline(kind, config, official_config, group, str(memory_input_dir))


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

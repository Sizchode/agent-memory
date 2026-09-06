"""Variables shared across baselines for controlled comparisons."""

from dataclasses import dataclass
from copy import deepcopy
from typing import Any


@dataclass(frozen=True)
class ControlledConfig:
    llm_model: str
    embedding_model: str
    top_k: int
    chunk_size: int
    llm_base_url: str | None = None
    embedding_base_url: str | None = None
    temperature: float = 0.0
    max_output_tokens: int = 512

    def __post_init__(self) -> None:
        if not self.llm_model or not self.embedding_model:
            raise ValueError("llm_model and embedding_model are required")
        if self.top_k <= 0 or self.chunk_size <= 0 or self.max_output_tokens <= 0:
            raise ValueError("top_k, chunk_size, and max_output_tokens must be positive")


def control_hipporag(config: dict[str, Any], shared: ControlledConfig) -> dict[str, Any]:
    result = deepcopy(config)
    result.update(llm_name=shared.llm_model, embedding_model_name=shared.embedding_model, retrieval_top_k=shared.top_k, qa_top_k=shared.top_k, max_new_tokens=shared.max_output_tokens)
    if shared.llm_base_url is not None:
        result["llm_base_url"] = shared.llm_base_url
    if shared.embedding_base_url is not None:
        result["embedding_base_url"] = shared.embedding_base_url
    return result


def control_mem0(config: dict[str, Any], shared: ControlledConfig) -> dict[str, Any]:
    result = deepcopy(config)
    result.setdefault("llm", {}).setdefault("config", {})["model"] = shared.llm_model
    result.setdefault("embedder", {}).setdefault("config", {})["model"] = shared.embedding_model
    result["llm"]["config"].update(temperature=shared.temperature, max_tokens=shared.max_output_tokens)
    if shared.llm_base_url is not None:
        result["llm"]["config"]["openai_base_url"] = shared.llm_base_url
    return result


def control_lightmem(config: dict[str, Any], shared: ControlledConfig, embedding_dimensions: int) -> dict[str, Any]:
    if embedding_dimensions <= 0:
        raise ValueError("embedding_dimensions must be positive")
    result = deepcopy(config)
    manager = result.setdefault("memory_manager", {}).setdefault("configs", {})
    manager.update(model=shared.llm_model, max_tokens=shared.max_output_tokens, temperature=shared.temperature)
    embedder = result.setdefault("text_embedder", {}).setdefault("configs", {})
    embedder.update(model=shared.embedding_model, embedding_dims=embedding_dimensions)
    for name in ("embedding_retriever", "summary_retriever"):
        if name in result:
            result[name].setdefault("configs", {})["embedding_model_dims"] = embedding_dimensions
    return result

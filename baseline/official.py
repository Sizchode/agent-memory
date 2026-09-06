"""Thin adapters over pinned official baseline source trees."""

from pathlib import Path
import sys
from collections.abc import Sequence
from typing import Any

from .base import RetrievedItem

_ROOT = Path(__file__).resolve().parents[1]
_OFFICIAL_ALGORITHMS = _ROOT / "baseline_algorithms"


def _prepend(path: Path) -> None:
    resolved = str(path)
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


class LightMemBaseline:
    def __init__(self, official_config: dict[str, Any]) -> None:
        _prepend(_OFFICIAL_ALGORITHMS / "LightMem" / "src")
        from lightmem.memory.lightmem import LightMemory
        self._memory = LightMemory.from_config(official_config)

    def build(self, memory_items: Sequence[str]) -> None:
        for item in memory_items:
            self._memory.add_memory([{"role": "user", "content": item}], force_segment=True, force_extract=True)

    def retrieve(self, query: str, top_k: int) -> list[RetrievedItem]:
        return [RetrievedItem(text) for text in self._memory.retrieve(query, limit=top_k)]

    def close(self) -> None:
        close = getattr(self._memory, "close", None)
        if callable(close):
            close()


class HippoRAG2Baseline:
    def __init__(self, official_config: dict[str, Any], save_dir: str | Path) -> None:
        _prepend(_OFFICIAL_ALGORITHMS / "HippoRAG" / "src")
        from hipporag import HippoRAG
        from hipporag.utils.config_utils import BaseConfig
        self._memory = HippoRAG(global_config=BaseConfig(**official_config), save_dir=str(save_dir))

    def build(self, memory_items: Sequence[str]) -> None:
        self._memory.index(list(memory_items))

    def retrieve(self, query: str, top_k: int) -> list[RetrievedItem]:
        solution = self._memory.retrieve([query], num_to_retrieve=top_k)[0]
        scores = solution.doc_scores.tolist() if solution.doc_scores is not None else [None] * len(solution.docs)
        return [RetrievedItem(text, score) for text, score in zip(solution.docs, scores)]

    def close(self) -> None:
        self._memory.close()


class Mem0Baseline:
    def __init__(self, official_config: dict[str, Any], namespace: str) -> None:
        _prepend(_OFFICIAL_ALGORITHMS / "mem0")
        from mem0 import Memory
        self._memory = Memory.from_config(official_config)
        self._filters = {"user_id": namespace}

    def build(self, memory_items: Sequence[str]) -> None:
        for item in memory_items:
            self._memory.add([{"role": "user", "content": item}], user_id=self._filters["user_id"], infer=True)

    def retrieve(self, query: str, top_k: int) -> list[RetrievedItem]:
        response = self._memory.search(query, top_k=top_k, filters=self._filters)
        return [RetrievedItem(str(item["memory"]), float(item["score"]), item) for item in response["results"]]

    def close(self) -> None:
        return None


class MAGMABaseline:
    """Official MAGMA adapter; its algorithm builds from structured sessions."""

    def __init__(self, cache_dir: str | Path, llm_model: str, embedding_model: str, use_episodes: bool = False) -> None:
        _prepend(_OFFICIAL_ALGORITHMS / "MAGMA")
        from memory.memory_builder import MemoryBuilder
        self._builder = MemoryBuilder(cache_dir=str(cache_dir), llm_model=llm_model, use_episodes=use_episodes, embedding_model=embedding_model)
        self._query_engine = None

    def build_structured(self, locomo_sample: Any) -> None:
        from memory.query_engine import QueryEngine
        self._builder.build_memory(locomo_sample)
        self._query_engine = QueryEngine(self._builder.trg, self._builder.node_index, entity_session_map=self._builder.entity_session_map, entity_dia_map=self._builder.entity_dia_map)

    def build_locomo_file(self, path: str | Path, sample_index: int) -> None:
        from load_dataset import load_locomo_dataset
        samples = load_locomo_dataset(path)
        if not 0 <= sample_index < len(samples):
            raise IndexError(f"sample_index {sample_index} is outside the LoCoMo dataset")
        self.build_structured(samples[sample_index])

    def retrieve(self, query: str, top_k: int) -> list[RetrievedItem]:
        if self._query_engine is None:
            raise RuntimeError("build_structured must be called before retrieve")
        context, _ = self._query_engine.query(query, top_k=top_k)
        return [
            RetrievedItem(str(node.content), float(getattr(node, "similarity_score", 0.0)), {"node_id": node.node_id})
            for node in context.anchor_nodes[:top_k]
        ]

    def close(self) -> None:
        return None

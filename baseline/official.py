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

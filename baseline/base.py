"""Shared retrieval contract for controlled baseline evaluation."""

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class RetrievedItem:
    text: str
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryBaseline(Protocol):
    def build(self, memory_items: Sequence[str]) -> None: ...
    def retrieve(self, query: str, top_k: int) -> list[RetrievedItem]: ...
    def close(self) -> None: ...

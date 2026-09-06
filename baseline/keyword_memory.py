"""Small dependency-free retrieval baseline for the benchmark protocol."""

import re
from collections import Counter
from collections.abc import Iterable


class KeywordMemory:
    """Index chunks once and retrieve the highest-overlap chunk per query."""

    def __init__(self) -> None:
        self._chunks: tuple[str, ...] | None = None
        self._tokens: tuple[Counter[str], ...] | None = None

    def build(self, chunks: Iterable[str]) -> "KeywordMemory":
        normalized_chunks = tuple(chunks)
        if not normalized_chunks:
            raise ValueError("cannot build memory from zero chunks")
        self._chunks = normalized_chunks
        self._tokens = tuple(Counter(_tokenize(chunk)) for chunk in normalized_chunks)
        return self

    def retrieve(self, query: str) -> str:
        if self._chunks is None or self._tokens is None:
            raise RuntimeError("memory must be built before retrieval")
        query_tokens = Counter(_tokenize(query))
        scores = [sum((query_tokens & chunk_tokens).values()) for chunk_tokens in self._tokens]
        return self._chunks[max(range(len(scores)), key=scores.__getitem__)]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.casefold())

"""Dependency-free Okapi BM25 baseline."""

from collections import Counter
import math
import re
from collections.abc import Sequence

from .base import RetrievedItem


class BM25Baseline:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self._documents: tuple[str, ...] = ()
        self._tokens: tuple[Counter[str], ...] = ()
        self._idf: dict[str, float] = {}
        self._average_length = 0.0

    def build(self, memory_items: Sequence[str]) -> None:
        if not memory_items:
            raise ValueError("memory_items must not be empty")
        self._documents = tuple(memory_items)
        self._tokens = tuple(Counter(_tokenize(item)) for item in memory_items)
        self._average_length = sum(sum(row.values()) for row in self._tokens) / len(self._tokens)
        document_frequency = Counter(token for row in self._tokens for token in row)
        count = len(self._tokens)
        self._idf = {token: math.log(1 + (count - freq + 0.5) / (freq + 0.5)) for token, freq in document_frequency.items()}

    def retrieve(self, query: str, top_k: int) -> list[RetrievedItem]:
        if not self._documents:
            raise RuntimeError("build must be called before retrieve")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        scores = [self._score(_tokenize(query), row) for row in self._tokens]
        indices = sorted(range(len(scores)), key=lambda index: (-scores[index], index))[:top_k]
        return [RetrievedItem(self._documents[index], scores[index]) for index in indices]

    def _score(self, query: list[str], row: Counter[str]) -> float:
        length_norm = 1 - self.b + self.b * sum(row.values()) / self._average_length
        return sum(self._idf.get(token, 0.0) * row[token] * (self.k1 + 1) / (row[token] + self.k1 * length_norm) for token in query if row[token])

    def close(self) -> None:
        return None


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.casefold())

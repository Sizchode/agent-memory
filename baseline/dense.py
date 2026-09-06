"""Dense retrieval baseline with an injected, controlled embedding model."""

from collections.abc import Callable, Sequence
import math

from .base import RetrievedItem

EmbeddingFunction = Callable[[Sequence[str]], Sequence[Sequence[float]]]


class DenseRetrievalBaseline:
    def __init__(self, embed: EmbeddingFunction) -> None:
        self._embed = embed
        self._documents: tuple[str, ...] = ()
        self._vectors: tuple[tuple[float, ...], ...] = ()

    def build(self, memory_items: Sequence[str]) -> None:
        if not memory_items:
            raise ValueError("memory_items must not be empty")
        vectors = tuple(tuple(map(float, vector)) for vector in self._embed(memory_items))
        if len(vectors) != len(memory_items):
            raise ValueError("embedder returned a different number of vectors")
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1 or not next(iter(dimensions)):
            raise ValueError("embedding vectors must share a nonzero dimension")
        self._documents, self._vectors = tuple(memory_items), vectors

    def retrieve(self, query: str, top_k: int) -> list[RetrievedItem]:
        if not self._vectors:
            raise RuntimeError("build must be called before retrieve")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_vectors = self._embed([query])
        if len(query_vectors) != 1:
            raise ValueError("embedder must return exactly one query vector")
        scores = [_cosine(query_vectors[0], vector) for vector in self._vectors]
        indices = sorted(range(len(scores)), key=lambda index: (-scores[index], index))[:top_k]
        return [RetrievedItem(self._documents[index], scores[index]) for index in indices]

    def close(self) -> None:
        return None


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions do not match")
    denominator = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(x * x for x in right))
    return sum(x * y for x, y in zip(left, right, strict=True)) / denominator if denominator else 0.0

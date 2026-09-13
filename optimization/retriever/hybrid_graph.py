"""Fixed single-pass reciprocal-rank fusion over one graph's source corpus."""

from baseline.base import RetrievedItem
from baseline.bm25 import BM25Baseline


VARIANTS = ("original_graph_rrf", "canonical_graph_rrf",
            "original_graph_rrf_window", "canonical_graph_rrf_window")
RANK_CONSTANT = 60
RANK_WINDOW = 5


def fuse_rankings(graph_items, lexical_items, top_k, rank_constant=RANK_CONSTANT):
    if top_k < 1 or rank_constant < 0:
        raise ValueError("Invalid reciprocal-rank fusion parameters")
    candidates, scores, ranks = {}, {}, {}
    for name, items in (("graph", graph_items), ("lexical", lexical_items)):
        seen = set()
        for rank, item in enumerate(items, start=1):
            if item.text in seen:
                raise ValueError("A source occurs twice in one ranking")
            seen.add(item.text)
            if name == "lexical" and (item.score is None or item.score <= 0):
                continue
            candidates.setdefault(item.text, item)
            scores[item.text] = scores.get(item.text, 0.0) + 1.0 / (rank_constant + rank)
            ranks.setdefault(item.text, {})[name] = rank
    # Stable ties use graph order first, followed by previously unseen lexical sources.
    order = sorted(candidates, key=lambda text: -scores[text])[:top_k]
    return [RetrievedItem(text, scores[text], dict(candidates[text].metadata, fusion_ranks=ranks[text]))
            for text in order]


class HybridGraphMemory:
    def __init__(self, memory, ordered_keys, rank_constant=RANK_CONSTANT, rank_window=RANK_WINDOW):
        self.base = memory
        self._memory, self._generator = memory._memory, memory._generator
        rows = self._memory.chunk_embedding_store.get_all_id_to_rows()
        if len(set(ordered_keys)) != len(ordered_keys) or set(ordered_keys) != set(rows):
            raise ValueError("Lexical index must cover the same complete source corpus")
        self.lexical = BM25Baseline()
        self.lexical.build([rows[key]["content"] for key in ordered_keys])
        self.rank_constant, self.rank_window = rank_constant, rank_window

    def retrieve(self, query, top_k):
        if top_k > self.rank_window:
            raise ValueError("Requested output exceeds the fixed candidate window")
        graph_items = self.base.retrieve(query, self.rank_window)
        lexical_items = self.lexical.retrieve(query, self.rank_window)
        return fuse_rankings(graph_items, lexical_items, top_k, self.rank_constant)

    def efficiency_metrics(self):
        return self.base.efficiency_metrics()

    def close(self):
        self.lexical.close()
        self.base.close()

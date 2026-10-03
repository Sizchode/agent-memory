"""Current AMOR: weighted entity-source recommendation and ten-fact context.

Use load_amor_memory for the simplified method. load_optimized_memory remains
available for reproducing the older projected graph and component controls.
The input directory is an existing build_graph output for one source group.
"""

import json
from pathlib import Path

from optimization.graph_construction.weighted_membership import membership_graph, weighted_membership_graph
from optimization.retriever.hipporag import load_optimized_memory
from optimization.retriever.query_fact_context import QueryFactContext


class AMORMemory:
    """Recommend sources on the simplified graph, then render the QA context."""

    def __init__(self, memory):
        self.base = memory
        self._memory, self._generator = memory._memory, memory._generator
        self.contents = memory.contents
        self.context = QueryFactContext(self._memory, self.contents)

    def recommend_sources(self, query, top_k=5):
        return self.base.retrieve(query, top_k)

    def retrieve(self, query, top_k=5):
        return list(self.context.render(query, self.recommend_sources(query, top_k)))

    def efficiency_metrics(self):
        return self.base.efficiency_metrics()

    def close(self):
        self.base.close()


def load_amor_memory(config, artifact_directory, runtime):
    """Load the current weighted membership graph and its context renderer.

    Preserve the frozen graph's fact-derived edge weights and retained facts,
    remove connections outside supported entity-source membership, and retain
    the existing recognition, PPR, BM25 fusion, and ten-fact ordering.
    No experiment scripts, saved questions, or saved predictions are needed.
    """
    import igraph as ig

    directory = Path(artifact_directory)
    metadata = json.loads((directory / "graph.json").read_text())
    required = ("constructed_graph_file", "retained_fact_keys_file",
                "rank_fusion", "compiled_source_file", "source_graph")
    if any(key not in metadata for key in required):
        raise ValueError("AMOR requires the full retained-fact graph construction artifacts")
    memory = load_optimized_memory(config, directory, runtime)
    try:
        from hipporag.utils.misc_utils import text_processing

        original = ig.Graph.Read_Pickle(metadata["source_graph"])
        entities = memory._memory.entity_embedding_store.get_all_id_to_rows()
        entity_keys = {row["content"]: key for key, row in entities.items()}
        membership, _ = membership_graph(original, memory.contents, entity_keys, text_processing)
        memory._memory.graph = weighted_membership_graph(memory._memory.graph, membership)
        return AMORMemory(memory)
    except BaseException:
        memory.close()
        raise

"""Retrieve through explicit fact nodes using uniform fact personalization."""

from pathlib import Path
from time import perf_counter

import numpy as np

from optimization.retriever.hipporag import load_memory


class UniformFactSeeds:
    def __init__(self, hippo, facts):
        self.hippo, self.facts = hippo, facts

    def __call__(self, *, query, link_top_k, query_fact_scores, top_k_facts,
                 top_k_fact_indices, passage_node_weight):
        indices = sorted({self.facts[tuple(fact)] for fact in top_k_facts})
        if not indices:
            raise ValueError("Empty recognition must use the original dense fallback")
        reset = np.zeros(self.hippo.graph.vcount(), dtype=np.float64)
        reset[indices] = 1.0 / len(indices)
        start = perf_counter()
        result = self.hippo.run_ppr(reset, damping=self.hippo.global_config.damping)
        self.hippo.ppr_time += perf_counter() - start
        return result


def install_fact_graph(hippo, graph, *, keep_synonym_edges=False):
    """Install an existing incidence graph; optionally retain synonymy for ablation."""
    names = graph.vs["name"]
    original_names = hippo.graph.vs["name"]
    if names[:len(original_names)] != original_names or len(names) != len(set(names)):
        raise ValueError("Fact graph must preserve original node identities and order")
    weights = np.asarray(graph.es["weight"], dtype=np.float64)
    if graph.is_directed() or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Expected an undirected graph with finite nonnegative weights")
    result = graph.copy()
    if not keep_synonym_edges:
        remove = []
        for edge in result.es:
            if edge["edge_kind"] == "statement_incidence" or edge["passage_source"] is not None:
                continue
            if not edge["synonym_score"] or edge["weight"] != edge["synonym_score"]:
                raise ValueError("Unexpected inherited edge contribution in incidence graph")
            remove.append(edge.index)
        result.delete_edges(remove)
    facts = {tuple(vertex["statement"]): vertex.index for vertex in result.vs
             if vertex["statement"] is not None}
    if len(facts) != len(hippo.fact_node_keys):
        raise ValueError("Fact nodes must match the unchanged full fact index")
    if [result.vs[index]["name"] for index in hippo.passage_node_idxs] != hippo.passage_node_keys:
        raise ValueError("Passage identities changed")
    hippo.graph = result
    hippo.node_name_to_vertex_idx = {name: index for index, name in enumerate(names)}
    hippo.graph_search_with_fact_entities = UniformFactSeeds(hippo, facts)


def load_fact_memory(config, source, runtime, graph_path, *, keep_synonym_edges=False):
    import igraph as ig

    memory = load_memory(config, Path(source), Path(runtime))
    try:
        graph = ig.Graph.Read_Pickle(str(graph_path))
        install_fact_graph(memory._memory, graph, keep_synonym_edges=keep_synonym_edges)
    except BaseException:
        memory.close()
        raise
    return memory

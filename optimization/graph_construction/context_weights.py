"""Contextual synonym weights and symmetric degree normalization."""

import numpy as np


VARIANTS = ("relation_only", "balanced", "contextual", "contextual_balanced")


def construct_weights(graph, entity_sources, passage_keys, passage_vectors):
    """Return edge-weight variants without reading questions or answer labels.

    Context is the mean embedding of an entity's source passages. Only the
    synonym component is calibrated by context cosine; source-supported fact
    and passage connections remain present. Balanced variants apply symmetric
    strength normalization, with exponent 1/2, to the resulting adjacency.
    """
    names = graph.vs["name"]
    edges = np.asarray(graph.get_edgelist(), dtype=np.int64).reshape(-1, 2)
    original = np.asarray(graph.es["weight"], dtype=np.float64)
    if not np.isfinite(original).all() or np.any(original < 0):
        raise ValueError("Source graph has invalid edge weights")
    passage_index = {key: i for i, key in enumerate(passage_keys)}
    vectors = np.asarray(passage_vectors, dtype=np.float64)
    contexts = {}
    for name, sources in entity_sources.items():
        indices = [passage_index[key] for key in sorted(sources)]
        if indices:
            centroid = vectors[indices].mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                contexts[name] = centroid / norm

    relation_only = np.zeros(len(edges), dtype=np.float64)
    contextual = np.zeros(len(edges), dtype=np.float64)
    for i, edge in enumerate(graph.es):
        source, target = names[edge.source], names[edge.target]
        counts = edge["fact_source_counts"] or {}
        supported = max(float(sum(counts.values())),
                        1.0 if edge["passage_source"] is not None else 0.0)
        synonym = float(edge["synonym_score"] or 0.0)
        factor = 0.0
        if source in contexts and target in contexts:
            factor = float(np.clip(contexts[source] @ contexts[target], 0, 1))
        relation_only[i] = supported
        contextual[i] = max(supported, synonym * factor)

    def balanced(weights):
        strength = np.asarray(graph.strength(weights=weights.tolist(), mode="all"))
        denominator = np.sqrt(strength[edges[:, 0]] * strength[edges[:, 1]])
        return np.divide(weights, denominator, out=np.zeros_like(weights), where=denominator > 0)

    return {"relation_only": relation_only, "balanced": balanced(original),
            "contextual": contextual, "contextual_balanced": balanced(contextual)}

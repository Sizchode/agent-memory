"""Weight passage links by the fraction of their retained entity support."""

from collections import Counter
import numpy as np


VARIANTS = ("canonical_provenance", "canonical_provenance_balanced", "canonical_provenance_syn005")


def provenance_weights(graph, base_weights, documents, retained, entity_keys, normalize):
    original_support, retained_support = Counter(), Counter()
    for doc in documents:
        for triple in doc["extracted_triples"]:
            subject, _, obj = normalize(list(triple))
            for entity in {entity_keys[subject], entity_keys[obj]}:
                original_support[(doc["idx"], entity)] += 1
    for passage, (subject, _, obj) in retained:
        for entity in {entity_keys[subject], entity_keys[obj]}:
            retained_support[(passage, entity)] += 1
    if any(count > original_support[key] for key, count in retained_support.items()):
        raise ValueError("Retained support exceeds the original support")
    weights = np.array(base_weights, dtype=np.float64, copy=True)
    if weights.shape != (graph.ecount(),):
        raise ValueError("Source weights do not match graph edges")
    names = graph.vs["name"]
    for i, edge in enumerate(graph.es):
        passage = edge["passage_source"]
        if passage is None or weights[i] == 0:
            continue
        source, target = names[edge.source], names[edge.target]
        entity = target if source == passage else source
        key = (passage, entity)
        if original_support[key] == 0:
            raise ValueError("Positive source edge has no original entity support")
        weights[i] *= retained_support[key] / original_support[key]
    edges = np.asarray(graph.get_edgelist(), dtype=np.int64).reshape(-1, 2)
    strength = np.asarray(graph.strength(weights=weights.tolist(), mode="all"))
    denominator = np.sqrt(strength[edges[:, 0]] * strength[edges[:, 1]])
    balanced = np.divide(weights, denominator, out=np.zeros_like(weights), where=denominator > 0)
    synonyms = np.asarray([float(e["synonym_score"] or 0.0) for e in graph.es])
    return dict(canonical_provenance=weights, canonical_provenance_balanced=balanced,
                canonical_provenance_syn005=np.maximum(weights, 0.05 * synonyms))

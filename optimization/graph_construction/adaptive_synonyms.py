"""Discount synonym links using source assertions still represented in the graph."""

from collections import Counter
import numpy as np

from optimization.graph_construction.provenance_weights import provenance_weights


VARIANTS = ("adaptive_syn020", "adaptive_syn020_balanced", "adaptive_provenance_syn020")


def entity_compatibility(documents, retained, entity_keys, normalize, schema):
    retained_records = set(retained)
    statements, active = [], set()
    for doc in documents:
        for triple in doc["extracted_triples"]:
            normalized = tuple(normalize(list(triple)))
            subject, _, obj = normalized
            claim = (subject, normalize(schema[triple[1]]["canonical"]), obj)
            statements.append((subject, obj, claim))
            if (doc["idx"], normalized) in retained_records:
                active.add(claim)
    total, compatible = Counter(), Counter()
    for subject, obj, claim in statements:
        for entity in {entity_keys[subject], entity_keys[obj]}:
            total[entity] += 1
            if claim in active:
                compatible[entity] += 1
    return {entity: compatible[entity] / count for entity, count in total.items()}


def adaptive_weights(graph, base_weights, documents, retained, entity_keys, normalize, schema):
    compatibility = entity_compatibility(documents, retained, entity_keys, normalize, schema)
    names = graph.vs["name"]
    synonyms = np.zeros(graph.ecount(), dtype=np.float64)
    for i, edge in enumerate(graph.es):
        source, target = names[edge.source], names[edge.target]
        synonyms[i] = (0.2 * float(edge["synonym_score"] or 0.0)
                       * compatibility.get(source, 0.0) * compatibility.get(target, 0.0))
    weights = np.maximum(base_weights, synonyms)
    edges = np.asarray(graph.get_edgelist(), dtype=np.int64).reshape(-1, 2)
    strength = np.asarray(graph.strength(weights=weights.tolist(), mode="all"))
    denominator = np.sqrt(strength[edges[:, 0]] * strength[edges[:, 1]])
    balanced = np.divide(weights, denominator, out=np.zeros_like(weights), where=denominator > 0)
    local = provenance_weights(graph, base_weights, documents, retained, entity_keys, normalize)["canonical_provenance"]
    return dict(adaptive_syn020=weights, adaptive_syn020_balanced=balanced,
                adaptive_provenance_syn020=np.maximum(local, synonyms))

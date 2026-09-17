"""Consolidate graph support by source order, without discarding original text."""

from collections import Counter, defaultdict
import numpy as np


def retained_statements(documents, ordered_passage_keys, normalize):
    """Select normalized source-supported statements without changing source text."""
    positions = {key: i for i, key in enumerate(ordered_passage_keys)}
    if {doc["idx"] for doc in documents} != set(positions):
        raise ValueError("OpenIE and ordered source passages disagree")
    statements = []
    latest = {}
    for doc in documents:
        key = doc["idx"]
        for triple in doc["extracted_triples"]:
            if len(triple) != 3:
                raise ValueError("Malformed source triple")
            normalized = tuple(normalize(list(triple)))
            subject, relation, _ = normalized
            statements.append((key, normalized, (subject, relation)))
            slot = (subject, relation)
            latest[slot] = max(latest.get(slot, -1), positions[key])
    retained = []
    for key, triple, slot in statements:
        if positions[key] != latest[slot]:
            continue
        retained.append((key, triple))
    return retained, dict(
        source_statements=len(statements), retained_statement_support=len(retained),
        subject_relation_slots=len(latest), original_passages_preserved=len(positions))


def latest_relation_weights(graph, documents, ordered_passage_keys, entity_keys, normalize):
    """Keep source support from the last passage mentioning each subject/relation.

    Source order is an explicit indexing heuristic, not inferred event time.
    Multiple values in the same final passage remain unresolved and retained.
    The unchanged raw passage store still contains every historical statement.
    """
    retained, stats = retained_statements(documents, ordered_passage_keys, normalize)
    counts = Counter()
    passage_entities = defaultdict(set)
    for key, (subject, relation, obj) in retained:
        source, target = entity_keys[subject], entity_keys[obj]
        counts[tuple(sorted((source, target)))] += 1
        passage_entities[key].update((source, target))
    names = graph.vs["name"]
    weights = np.zeros(graph.ecount(), dtype=np.float64)
    for i, edge in enumerate(graph.es):
        source, target = names[edge.source], names[edge.target]
        passage = edge["passage_source"]
        if passage is not None:
            entity = target if source == passage else source
            weights[i] = float(entity in passage_entities[passage])
        else:
            weights[i] = counts[tuple(sorted((source, target)))]
    return weights, stats

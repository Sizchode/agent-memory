"""Consolidate graph support by source order, without discarding original text."""

from collections import Counter, defaultdict
import numpy as np


VARIANTS = ("latest_relation", "latest_relation_balanced")
SCHEMA_VARIANTS = ("schema_latest", "schema_latest_synonyms")
CANONICAL_VARIANTS = ("canonical_latest", "canonical_latest_balanced")


def retained_statements(documents, ordered_passage_keys, normalize, schema=None, consolidate_multiple=False):
    """Select source-supported statements, preserving the original triple wording."""
    positions = {key: i for i, key in enumerate(ordered_passage_keys)}
    if {doc["idx"] for doc in documents} != set(positions):
        raise ValueError("OpenIE and ordered source passages disagree")
    statements = []
    latest = {}
    discourse = 0
    for doc in documents:
        key = doc["idx"]
        for triple in doc["extracted_triples"]:
            if len(triple) != 3:
                raise ValueError("Malformed source triple")
            subject, relation, obj = normalize(list(triple))
            consolidate = True
            if schema is not None:
                record = schema[triple[1]]
                if record["role"] == "discourse":
                    discourse += 1
                    continue
                relation = normalize(record["canonical"])
                consolidate = record["cardinality"] == "single" or consolidate_multiple
            statements.append((key, tuple(normalize(list(triple))), (subject, relation), consolidate))
            slot = (subject, relation)
            latest[slot] = max(latest.get(slot, -1), positions[key])
    retained = []
    for key, triple, slot, consolidate in statements:
        if consolidate and positions[key] != latest[slot]:
            continue
        retained.append((key, triple))
    return retained, dict(
        source_statements=len(statements) + discourse, retained_statement_support=len(retained),
        discourse_statement_support_removed=discourse,
        subject_relation_slots=len(latest), original_passages_preserved=len(positions))


def latest_relation_weights(graph, documents, ordered_passage_keys, entity_keys, normalize, schema=None,
                            consolidate_multiple=False):
    """Keep source support from the last passage mentioning each subject/relation.

    Source order is an explicit indexing heuristic, not inferred event time.
    Multiple values in the same final passage remain unresolved and retained.
    The unchanged raw passage store still contains every historical statement.
    """
    retained, stats = retained_statements(documents, ordered_passage_keys, normalize, schema, consolidate_multiple)
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
    edges = np.asarray(graph.get_edgelist(), dtype=np.int64).reshape(-1, 2)
    strength = np.asarray(graph.strength(weights=weights.tolist(), mode="all"))
    denominator = np.sqrt(strength[edges[:, 0]] * strength[edges[:, 1]])
    balanced = np.divide(weights, denominator, out=np.zeros_like(weights), where=denominator > 0)
    variants = {"latest_relation": weights, "latest_relation_balanced": balanced}
    if schema is not None:
        synonyms = np.asarray([float(e["synonym_score"] or 0.0) for e in graph.es])
        variants = {"schema_latest": weights, "schema_latest_synonyms": np.maximum(weights, synonyms)}
    if consolidate_multiple:
        if schema is None:
            raise ValueError("Canonical consolidation requires a relation schema")
        variants = {"canonical_latest": weights, "canonical_latest_balanced": balanced}
    return variants, stats

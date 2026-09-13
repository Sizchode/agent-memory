"""Prune the fact index and entity provenance to retained graph support."""

from collections import defaultdict

from optimization.graph_construction.source_consolidation import retained_statements


VARIANTS = ("index_latest", "index_schema")


def construct_fact_index(documents, ordered_passage_keys, entity_keys, fact_rows, normalize, schema=None):
    retained, stats = retained_statements(documents, ordered_passage_keys, normalize, schema)
    facts = defaultdict(set)
    entities = defaultdict(set)
    for passage, triple in retained:
        facts[str(triple)].add(passage)
        entities[entity_keys[triple[0]]].add(passage)
        entities[entity_keys[triple[2]]].add(passage)
    available = {row["content"] for row in fact_rows.values()}
    if not set(facts).issubset(available):
        raise ValueError("Retained facts are missing from the source fact store")
    ids = [key for key, row in fact_rows.items() if row["content"] in facts]
    return dict(fact_ids=ids,
                entity_sources={key: sorted(value) for key, value in entities.items()},
                fact_sources={key: sorted(value) for key, value in facts.items()},
                original_fact_count=len(fact_rows), retained_fact_count=len(ids),
                source_statistics=stats)


def apply_fact_index(hippo, index):
    """Select existing rows and vectors only; never rewrite or delete source stores."""
    import numpy as np

    keys = index["fact_ids"]
    available = set(hippo.fact_embedding_store.get_all_ids())
    if len(set(keys)) != len(keys) or not set(keys).issubset(available):
        raise ValueError("Invalid retained fact IDs")
    entities = set(hippo.entity_node_keys)
    passages = set(hippo.passage_node_keys)
    if not set(index["entity_sources"]).issubset(entities):
        raise ValueError("Unknown retained entity")
    for sources in (*index["entity_sources"].values(), *index["fact_sources"].values()):
        if not set(sources).issubset(passages):
            raise ValueError("Unknown retained source passage")
    hippo.fact_node_keys = list(keys)
    hippo.fact_embeddings = np.asarray(hippo.fact_embedding_store.get_embeddings(keys))
    hippo.ent_node_to_chunk_ids = {key: set(value) for key, value in index["entity_sources"].items()}
    hippo.proc_triples_to_docs = {key: set(value) for key, value in index["fact_sources"].items()}

"""Represent extracted statements as incidence nodes with explicit source membership."""

from collections import defaultdict


def statement_incidence_graph(graph, documents, selected, entity_keys, normalize, *, refined):
    """Use binary incidence, retaining original passage connections.

    The input graph already carries the chosen refinement weights. Statement
    nodes are internal construction objects, projected out before retrieval.
    """
    if graph.is_directed():
        raise ValueError("This construction expects the existing undirected graph")
    names = graph.vs["name"]
    positions = {name: i for i, name in enumerate(names)}
    statements = {}
    for doc in documents:
        for triple in doc["extracted_triples"]:
            normalized = tuple(normalize(list(triple)))
            if len(normalized) != 3:
                raise ValueError("Malformed source triple")
            statements.setdefault(normalized, len(statements))
    sources = defaultdict(set)
    for source, triple in selected:
        if triple not in statements or source not in positions:
            raise ValueError("Selected statement is absent from source extraction")
        sources[triple].add(source)
    result = graph.copy()
    # Replace pair-level fact contributions, not passage membership or synonymy.
    residual = [float(edge["weight"]) if edge["passage_source"] is not None else
                (0.0 if refined else float(edge["synonym_score"] or 0.0)) for edge in graph.es]
    result.es["weight"] = residual
    result.delete_edges([i for i, weight in enumerate(residual) if weight == 0])
    statement_names = [f"statement-{index}" for index in statements.values()]
    if set(statement_names).intersection(positions):
        raise ValueError("Statement node IDs overlap existing node IDs")
    result.add_vertices(len(statements), attributes={"name": statement_names,
                        "statement": list(statements), "node_kind": ["statement"] * len(statements)})
    edges = []
    for triple, index in statements.items():
        if not sources[triple]:
            continue
        subject, _, obj = triple
        members = {positions[entity_keys[subject]], positions[entity_keys[obj]]}
        members.update(positions[source] for source in sources[triple])
        edges.extend((len(names) + index, member) for member in sorted(members))
    result.add_edges(edges, attributes={"weight": [1.0] * len(edges),
                    "edge_kind": ["statement_incidence"] * len(edges)})
    return result


def project_statement_graph(graph, original_vertices):
    """Apply Kumar et al. (2020), Eq. 3, retaining residual source connections."""
    import igraph as ig
    import numpy as np
    from scipy import sparse

    adjacency = graph.get_adjacency_sparse(attribute="weight").tocsr()
    incidence = adjacency[:original_vertices, original_vertices:]
    degrees = np.asarray(incidence.sum(axis=0), dtype=np.float64).ravel()
    if np.any(degrees == 1):
        raise ValueError("A loop-free fact transition requires at least two members")
    degrees = degrees - 1
    inverse = np.divide(1.0, degrees, out=np.zeros_like(degrees), where=degrees > 0)
    facts = incidence @ sparse.diags(inverse) @ incidence.T
    facts.setdiag(0)
    facts.eliminate_zeros()
    projected = adjacency[:original_vertices, :original_vertices] + facts
    result = ig.Graph.Weighted_Adjacency(projected, mode="undirected", loops="twice")
    for attribute in graph.vertex_attributes():
        result.vs[attribute] = graph.vs[attribute][:original_vertices]
    return result

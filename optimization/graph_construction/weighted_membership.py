"""Build AMOR's entity-to-source graph with retained fact-derived weights.

The source support and weighting are the existing September 29 simplified
method. Projection is used only to obtain weights; recommendation runs on
entity-to-source edges with the original node order.
"""

def membership_graph(original, contents, entity_keys, normalize):
    """Return exactly the residual binary source links used by construction."""
    from optimization.graph_construction.source_consolidation import statement_weights

    selected = [(key, tuple(normalize(list(triple)))) for key, content in contents.items()
                for triple in content["retained_triples"]]
    weights = statement_weights(original, selected, entity_keys)
    result = original.copy()
    result.es["weight"] = weights.tolist()
    result.delete_edges([edge.index for edge in result.es
                         if edge["passage_source"] is None or edge["weight"] == 0])
    assert all(weight == 1 for weight in result.es["weight"])
    assert result.vs["name"] == original.vs["name"]
    return result, selected


def weighted_membership_graph(projected, membership):
    """Keep only supported entity-to-source edges and preserve their weights."""
    if projected.is_directed() or membership.is_directed():
        raise ValueError("AMOR expects undirected graphs")
    if projected.vs["name"] != membership.vs["name"]:
        raise ValueError("Projection and membership node order differ")
    source_edges = set(membership.get_edgelist())
    if not source_edges.issubset(projected.get_edgelist()):
        raise ValueError("Projected graph is missing source membership edges")
    if any(weight != 1 for weight in membership.es["weight"]):
        raise ValueError("Membership must identify binary source connections")
    result = projected.copy()
    result.delete_edges([edge.index for edge in projected.es if edge.tuple not in source_edges])
    return result

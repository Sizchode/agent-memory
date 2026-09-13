"""A fixed, task-independent calibration grid for source-consolidated graphs."""

import numpy as np


CONFIGS = {
    "canonical_soft025": {"synonym_scale": 0.0, "strength_exponent": 0.25},
    "canonical_syn005": {"synonym_scale": 0.05, "strength_exponent": 0.0},
    "canonical_syn020": {"synonym_scale": 0.2, "strength_exponent": 0.0},
    "canonical_syn005_soft025": {"synonym_scale": 0.05, "strength_exponent": 0.25},
    "canonical_syn020_soft025": {"synonym_scale": 0.2, "strength_exponent": 0.25},
}


def calibrated_weights(graph, source_weights):
    source_weights = np.asarray(source_weights, dtype=np.float64)
    if source_weights.shape != (graph.ecount(),):
        raise ValueError("Source weights do not match the graph")
    if not np.isfinite(source_weights).all() or np.any(source_weights < 0):
        raise ValueError("Source weights must be finite and nonnegative")
    synonyms = np.asarray([float(e["synonym_score"] or 0.0) for e in graph.es])
    edges = np.asarray(graph.get_edgelist(), dtype=np.int64).reshape(-1, 2)
    variants = {}
    for name, config in CONFIGS.items():
        weights = np.maximum(source_weights, config["synonym_scale"] * synonyms)
        exponent = config["strength_exponent"]
        if exponent:
            strength = np.asarray(graph.strength(weights=weights.tolist(), mode="all"))
            denominator = (strength[edges[:, 0]] * strength[edges[:, 1]]) ** exponent
            weights = np.divide(weights, denominator, out=np.zeros_like(weights), where=denominator > 0)
        variants[name] = weights
    return variants

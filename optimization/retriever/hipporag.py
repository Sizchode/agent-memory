"""Reuse HippoRAG recognition and PPR with an independent writable LLM cache."""

from pathlib import Path
import json
import shutil

from baseline.official import HippoRAG2Baseline


def load_memory(config, source, runtime):
    memory = HippoRAG2Baseline(config, source)
    try:
        old_cache = Path(memory._generator.cache_file_name)
        destination = runtime / "llm_cache" / old_cache.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(old_cache, destination)
        memory._generator.cache_file_name = str(destination)
        memory._memory.prepare_retrieval_objects()
    except BaseException:
        memory.close()
        raise
    return memory


def load_optimized_memory(config, artifact_directory, runtime):
    """Load a frozen optimized graph for ordinary, potentially new questions.

    Query embedding, recognition and PPR are the existing HippoRAG methods.
    This interface does not read saved questions, query resets or predictions.
    A reachable generator is needed when recognition is not already cached.
    """
    import numpy as np

    artifact_directory = Path(artifact_directory)
    metadata_path = artifact_directory / "graph.json"
    if not metadata_path.exists():
        metadata_path = artifact_directory / "construction.json"
    metadata = json.loads(metadata_path.read_text())
    if (any(key in metadata for key in ("frozen_graph_file", "passage_index_directory", "retrieval_mode", "retained_fact_keys_file"))
            or metadata.get("pack_source_windows") or (artifact_directory / "fact_index.json").exists()):
        raise ValueError("Retired experimental representation; use the archived implementation")
    source_graph = Path(metadata["source_graph"])
    memory = load_memory(config, source_graph.parent.parent, Path(runtime))
    try:
        if "constructed_graph_file" in metadata:
            import igraph as ig
            constructed = ig.Graph.Read_Pickle(metadata["constructed_graph_file"])
            original_names = memory._memory.graph.vs["name"]
            if constructed.vs["name"][:len(original_names)] != original_names:
                raise ValueError("Constructed graph changed original node IDs or order")
            names = constructed.vs["name"]
            if len(names) != len(set(names)):
                raise ValueError("Constructed graph contains duplicate node IDs")
            memory._memory.graph = constructed
            memory._memory.node_name_to_vertex_idx = {name: i for i, name in enumerate(names)}
        weights = np.load(artifact_directory / "edge_weights.npy", allow_pickle=False)
        if weights.shape != (memory._memory.graph.ecount(),):
            raise ValueError("Optimized weights do not match the source graph edge count")
        if not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("Optimized weights must be finite and nonnegative")
        memory._memory.graph.es["weight"] = weights.tolist()
        if "rank_fusion" in metadata:
            from optimization.retriever.hybrid_graph import HybridGraphMemory
            fusion = metadata["rank_fusion"]
            keys = json.loads((artifact_directory / "lexical_source_keys.json").read_text())
            memory = HybridGraphMemory(memory, keys, fusion["rank_constant"], fusion["rank_window"])
        if "compiled_source_file" in metadata:
            from optimization.retriever.compiled_sources import CompiledSourceMemory
            memory = CompiledSourceMemory(memory, json.loads(Path(metadata["compiled_source_file"]).read_text()))
    except BaseException:
        memory.close()
        raise
    return memory


class CacheMissGuard:
    """Reject silent DPR fallback when the old recognition cache is insufficient."""

    def __init__(self, generator):
        self.misses = 0
        generator.openai_client.chat.completions.create = self.reject

    def reject(self, *args, **kwargs):
        self.misses += 1
        raise RuntimeError("Recognition cache miss: a live generator is required")

    def check(self):
        if self.misses:
            raise RuntimeError(f"Recognition cache missed {self.misses} calls; refusing fallback results")


class GenerationFailureGuard:
    """Surface provider failures swallowed by the upstream recognition filter."""

    def __init__(self, generator):
        self.failures = []
        call = generator.openai_client.chat.completions.create

        def checked(*args, **kwargs):
            try:
                return call(*args, **kwargs)
            except Exception as error:
                self.failures.append(type(error).__name__)
                raise

        generator.openai_client.chat.completions.create = checked

    def check(self):
        if self.failures:
            raise RuntimeError(f"Recognition provider failures: {self.failures}; refusing fallback results")

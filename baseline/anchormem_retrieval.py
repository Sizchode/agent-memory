"""Read existing AnchorMem stores: official retrieval or dense memory top-five."""

from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from baseline.base import RetrievedItem
from baseline.reference_memory import AnchorMemBuilder


class AnchorMemRetrieval:
    def __init__(self, source, runtime_dir, *, generator_model, embedding_model, seed=42):
        source = Path(source)
        self.builder = AnchorMemBuilder(
            output_dir=runtime_dir, generator_model=generator_model,
            generator_base_url="http://127.0.0.1:9/v1",
            embedding_model=embedding_model, seed=seed, cache_only=True,
        )
        self.memory = self.builder.memory
        from embedding_store import EmbeddingStore

        label = generator_model.replace("/", "_") + "_Transformers_" + embedding_model.replace("/", "_")
        for namespace in ("chunk", "fact", "event"):
            directory = source / label / f"{namespace}_embeddings"
            if not (directory / f"vdb_{namespace}.parquet").is_file():
                raise FileNotFoundError(directory)
            store = EmbeddingStore(self.memory.embedding_model, str(directory), 8, namespace)
            setattr(self.memory, f"{namespace}_embedding_store", store)

        payload = json.loads((source / ("fact_results_" + generator_model.replace("/", "_") + ".json")).read_text())
        self.memory.fact_to_chunk_ids = defaultdict(set)
        for doc in payload["docs"]:
            for fact in doc["extracted_facts"]:
                fact_id = self.memory.fact_embedding_store.text_to_hash_id[fact]
                self.memory.fact_to_chunk_ids[fact_id].add(doc["idx"])
        self.memory.fact_to_event_ids = defaultdict(set)
        for event in payload["events"]:
            for fact_id in event["source_facts"]:
                self.memory.fact_to_event_ids[fact_id].add(event["idx"])
        self.memory.prepare_retrieval_objects()

        self.items = []
        vectors = []
        for kind in ("fact", "event"):
            store = getattr(self.memory, f"{kind}_embedding_store")
            for ident, text, vector in zip(store.hash_ids, store.texts, store.embeddings, strict=True):
                self.items.append((text, {"memory_type": kind, "memory_id": ident}))
                vectors.append(vector)
        self.vectors = np.asarray(vectors, dtype=np.float64)
        if not self.items:
            raise ValueError("AnchorMem has no stored facts or events")
        self.norms = np.linalg.norm(self.vectors, axis=1)

    def retrieve_top5(self, query):
        # Same cosine and stable index tie-break as baseline/dense.py, using
        # the already stored vectors. Vectorization only avoids Python loops.
        vector = np.asarray(self.memory.embedding_model.batch_encode([query])[0], dtype=np.float64)
        denominators = self.norms * np.linalg.norm(vector)
        scores = np.divide(self.vectors @ vector, denominators,
                           out=np.zeros(len(self.items)), where=denominators != 0)
        order = np.argsort(-scores, kind="stable")[:5]
        return [RetrievedItem(self.items[i][0], float(scores[i]), self.items[i][1]) for i in order]

    def retrieve_official(self, query):
        # Keep upstream set expansion/order and all returned events unchanged.
        solution = self.memory.retrieve([query])[0]
        chunks = self.memory.chunk_embedding_store.text_to_hash_id
        events = self.memory.event_embedding_store.text_to_hash_id
        return [RetrievedItem(text, metadata={
            "memory_type": "chunk" if text in chunks else "event",
            "memory_id": chunks.get(text, events.get(text)),
        }) for text in solution.docs]

    def close(self):
        self.builder.close()

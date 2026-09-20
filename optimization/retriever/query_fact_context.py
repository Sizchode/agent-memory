"""Keep source text and append question-relevant retained facts.

The ten-fact default and similarity ordering follow the KAPING component
experiment; source retrieval, retained support, and embeddings stay unchanged.
"""

import json

import numpy as np

from baseline.base import RetrievedItem


class QueryFactContext:
    def __init__(self, hippo, contents, fact_limit=10):
        from hipporag.utils.misc_utils import text_processing

        if fact_limit < 1 or not hippo.global_config.embedding_return_as_normalized:
            raise ValueError("Positive fact limit and normalized embeddings are required")
        self.hippo, self.contents, self.fact_limit = hippo, contents, fact_limit
        self.normalize = text_processing
        self.text_to_key = {content["original_source_text"]: key for key, content in contents.items()}
        if len(self.text_to_key) != len(contents):
            raise ValueError("Source texts must have unique identities")
        self.fact_keys = hippo.fact_embedding_store.get_all_ids()
        self.fact_positions = {hippo.fact_embedding_store.get_row(key)["content"]: index
                               for index, key in enumerate(self.fact_keys)}
        self.vector_positions = {key: index for index, key in enumerate(hippo.fact_node_keys)}
        if len(self.fact_positions) != len(self.fact_keys):
            raise ValueError("Duplicate canonical facts in the source index")

    def _source(self, item, key):
        content = self.contents[key]
        original = content["original_source_text"]
        lines = [f"Source position: {content['source_position']}"]
        if content["timestamp"] is not None:
            lines.append(f"Source timestamp: {content['timestamp']}")
        lines.extend(["Original source record:", original])
        neighbors = content.get("window_sources", [])
        if neighbors:
            lines.append("Surrounding records with the same source timestamp:")
            lines.append("\n\n".join(f"Source position: {self.contents[neighbor]['source_position']}\nOriginal source record:\n"
                                    f"{self.contents[neighbor]['original_source_text']}" for neighbor in neighbors))
        return RetrievedItem("\n".join(lines), item.score,
            dict(item.metadata, context_representation="original_source_and_frozen_window"))

    def render(self, query, source_items):
        """Return QA context entries; the source retrieval budget is unchanged."""
        sources, eligible = [], []
        for item in source_items:
            key = self.text_to_key[item.metadata["original_source_text"]]
            if item.metadata.get("window_sources", []) != self.contents[key].get("window_sources", []):
                raise ValueError("Retrieved source and frozen window differ")
            eligible.append(key)
            eligible.extend(item.metadata.get("window_sources", []))
            sources.append(self._source(item, key))
        eligible = list(dict.fromkeys(eligible))
        assertions, support = {}, {}
        for key in eligible:
            for triple in self.contents[key]["retained_triples"]:
                normalized = str(tuple(self.normalize(list(triple))))
                index = self.fact_positions[normalized]
                if key not in self.hippo.proc_triples_to_docs[normalized]:
                    raise ValueError("Fact provenance differs from the source index")
                assertions.setdefault(index, triple)
                support.setdefault(index, []).append(key)
        indices = sorted(assertions)
        if not indices:
            return tuple(sources)
        self.hippo.get_query_embeddings([query])
        vector = self.hippo.query_to_embedding["triple"][query].reshape(-1)
        positions = [self.vector_positions[self.fact_keys[index]] for index in indices]
        scores = self.hippo.fact_embeddings[positions] @ vector
        if not np.isfinite(scores).all():
            raise ValueError("Non-finite fact similarity")
        # Preserve original index tie order; display more relevant facts last.
        selected = np.argsort(-scores, kind="stable")[:self.fact_limit][::-1]
        facts = []
        for position in selected:
            index = indices[position]
            keys = list(dict.fromkeys(support[index]))
            timestamps = list(dict.fromkeys(self.contents[key]["timestamp"] for key in keys
                                           if self.contents[key]["timestamp"] is not None))
            text = "(" + ", ".join(assertions[index]) + ")"
            if timestamps:
                text = "Source timestamps: " + json.dumps(timestamps, ensure_ascii=False) + "\n" + text
            facts.append(RetrievedItem(text, float(scores[position]), dict(
                fact_key=self.fact_keys[index], source_passages=keys, source_timestamps=timestamps,
                context_representation="retrieved_triples")))
        return tuple(sources + facts)

"""Return frozen source-node representations after ordinary graph retrieval."""

from baseline.base import RetrievedItem

def compiled_item(item, key, content):
    if item.text != content["original_source_text"]:
        raise ValueError("Compiled source does not match retrieved original text")
    metadata = dict(item.metadata, source_passage=key, original_source_text=item.text,
                    source_position=content["source_position"], timestamp=content["timestamp"],
                    facts_format=content.get("facts_format", "json"),
                    context_representation=content.get("context_representation", "retained_source_triples"))
    if "window_sources" in content:
        metadata["window_sources"] = content["window_sources"]
    return RetrievedItem(content["text"], item.score, metadata)


class CompiledSourceMemory:
    def __init__(self, memory, contents):
        self.base = memory
        self._memory = memory._memory
        self._generator = memory._generator
        rows = self._memory.chunk_embedding_store.get_all_id_to_rows()
        if set(rows) != set(contents):
            raise ValueError("Compiled context must cover the complete source store")
        for key, row in rows.items():
            if row["content"] != contents[key]["original_source_text"]:
                raise ValueError("Compiled source text differs from the source store")
        self.text_to_key = {row["content"]: key for key, row in rows.items()}
        self.contents = contents

    def retrieve(self, query, top_k):
        items = self.base.retrieve(query, top_k)
        return [compiled_item(item, self.text_to_key[item.text], self.contents[self.text_to_key[item.text]])
                for item in items]

    def efficiency_metrics(self):
        return self.base.efficiency_metrics()

    def close(self):
        self.base.close()

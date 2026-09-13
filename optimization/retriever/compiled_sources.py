"""Return frozen source-node representations after ordinary graph retrieval."""

from baseline.base import RetrievedItem

PACKING_VARIANTS = {"canonical_rrf_deduplicated": "json", "canonical_rrf_sentence_facts": "sentences"}


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


def pack_source_items(items, text_to_key, contents):
    centers = {text_to_key[item.text] for item in items}
    if len(centers) != len(items):
        raise ValueError("Packing requires unique central sources")
    assigned = set(centers)
    output = []
    for item in items:
        key = text_to_key[item.text]
        content = contents[key]
        neighbors = [neighbor for neighbor in content["window_sources"] if neighbor not in assigned]
        assigned.update(neighbors)
        packed = dict(content, text=content["window_base_text"], window_sources=neighbors)
        if neighbors:
            records = [f"Source position: {contents[neighbor]['source_position']}\nOriginal source record:\n"
                       f"{contents[neighbor]['original_source_text']}" for neighbor in neighbors]
            packed["text"] += "\nSurrounding records with the same source timestamp:\n" + "\n\n".join(records)
        result = compiled_item(item, key, packed)
        result.metadata["source_windows_deduplicated"] = True
        output.append(result)
    return output


class CompiledSourceMemory:
    def __init__(self, memory, contents, pack_windows=False):
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
        self.pack_windows = pack_windows
        if pack_windows and any("window_base_text" not in content or "window_sources" not in content
                                for content in contents.values()):
            raise ValueError("Window packing requires structured window contents")

    def retrieve(self, query, top_k):
        items = self.base.retrieve(query, top_k)
        if self.pack_windows:
            return pack_source_items(items, self.text_to_key, self.contents)
        return [compiled_item(item, self.text_to_key[item.text], self.contents[self.text_to_key[item.text]])
                for item in items]

    def efficiency_metrics(self):
        return self.base.efficiency_metrics()

    def close(self):
        self.base.close()

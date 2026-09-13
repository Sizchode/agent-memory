"""Serialize retained source assertions without adding or rewriting facts."""

import json


VARIANTS = ("original_graph_compiled", "canonical_graph_compiled")
SOURCE_VARIANTS = ("canonical_graph_with_source", "adaptive_graph_with_source")


def compile_sources(documents, ordered_keys, retained, timestamps, normalize, with_source=False, facts_format="json"):
    if facts_format not in ("json", "sentences"):
        raise ValueError("Unsupported source-fact representation")
    by_key = {doc["idx"]: doc for doc in documents}
    if set(by_key) != set(ordered_keys):
        raise ValueError("Source passages and source order differ")
    selected = set(retained)
    contents = {}
    for position, key in enumerate(ordered_keys):
        doc = by_key[key]
        facts, seen = [], set()
        for triple in doc["extracted_triples"]:
            normalized = tuple(normalize(list(triple)))
            if (key, normalized) in selected and normalized not in seen:
                facts.append(list(triple))
                seen.add(normalized)
        lines = [f"Source position: {position}"]
        timestamp = timestamps.get(key)
        if timestamp is not None:
            lines.append(f"Source timestamp: {timestamp}")
        if with_source:
            lines.extend(["Original source record:", doc["passage"],
                          "Graph-retained assertions (latest source per subject/relation):"])
        if facts_format == "json":
            lines.append("Facts: " + json.dumps(facts, ensure_ascii=False))
        else:
            lines.append("Facts:")
            lines.extend(" ".join(triple) + "." for triple in facts)
        contents[key] = dict(text="\n".join(lines), original_source_text=doc["passage"],
                             source_position=position, timestamp=timestamp, facts=len(facts),
                             retained_triples=facts, facts_format=facts_format,
                             context_representation="original_source_with_retained_triples" if with_source
                             else "retained_source_triples")
    return contents

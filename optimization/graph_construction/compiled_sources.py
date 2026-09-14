"""Serialize retained source assertions without adding or rewriting facts."""

import json


def compile_sources(documents, ordered_keys, retained, timestamps, normalize):
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
        lines.extend(["Original source record:", doc["passage"]])
        lines.append("Graph-retained assertions (latest source per subject/relation):")
        lines.append("Facts: " + json.dumps(facts, ensure_ascii=False))
        contents[key] = dict(text="\n".join(lines), original_source_text=doc["passage"],
                             source_position=position, timestamp=timestamp, facts=len(facts),
                             retained_triples=facts, facts_format="json",
                             context_representation="original_source_with_retained_triples")
    return contents

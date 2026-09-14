"""Freeze neighboring source context inside contiguous timestamp groups."""


def attach_source_windows(contents, ordered_keys, window_size=3):
    if window_size < 1:
        raise ValueError("Source window size must be positive")
    if len(set(ordered_keys)) != len(ordered_keys) or set(contents) != set(ordered_keys):
        raise ValueError("Source window requires the complete unique source order")
    output = {}
    for position, key in enumerate(ordered_keys):
        content = contents[key]
        timestamp = content["timestamp"]
        neighbors = []
        if timestamp is not None:
            for direction in (-1, 1):
                for distance in range(1, window_size + 1):
                    index = position + direction * distance
                    if index < 0 or index >= len(ordered_keys):
                        break
                    neighbor_key = ordered_keys[index]
                    if contents[neighbor_key]["timestamp"] != timestamp:
                        break
                    neighbors.append((index, neighbor_key))
        neighbors.sort()
        result = dict(content)
        result["window_base_text"] = content["text"]
        if neighbors:
            records = [f"Source position: {index}\nOriginal source record:\n"
                       f"{contents[neighbor_key]['original_source_text']}"
                       for index, neighbor_key in neighbors]
            result["text"] += "\nSurrounding records with the same source timestamp:\n" + "\n\n".join(records)
            result["context_representation"] = "original_source_with_retained_triples_and_window"
        result["window_sources"] = [neighbor_key for _, neighbor_key in neighbors]
        output[key] = result
    return output

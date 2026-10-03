"""Retrieve missing memory links with two short, evidence-grounded search queries."""

import argparse
import json
from pathlib import Path
import random
from time import perf_counter

from baseline.base import RetrievedItem
from experiments.run_anchormem import TASKS
from experiments.runner import _answer_prompt, _read_retrieval_records, evaluate_retrieval
from main import EMBEDDING_MODEL, _seed_everything
from utils.models import HuggingFaceChatModel, HuggingFaceEmbedder


QUERY_PROMPT = (
    "You formulate a search query for a memory store, not a final answer. Read the question, "
    "retrieved evidence, and task policy. Find the most important unresolved factual link needed "
    "to answer. Write ONE short search query about that link, using specific entities actually "
    "present in the question or evidence. For a chain, search the next unresolved relation rather "
    "than repeating the whole original question. If facts conflict, seek evidence that resolves "
    "their applicable date or explicit version policy. If no gap is apparent, query the relation "
    "most important to verify. Do not assume a missing answer from world knowledge. Output only "
    "the search query, without explanation or a final answer."
)
VARIANT = "memory_union_gap_queries2"
QUERY_REQUEST = (
    "The packet above describes the final task, which will be answered later. For THIS intermediate "
    "step, do not answer it. Return only one short memory-search query for its most important "
    "unresolved relation, using entities grounded in the question or retrieved records."
)


def expand_evidence(messages, items, records, vectors, norms, embed, model, usage, *, rounds, query_tokens):
    """This helper receives source evidence/messages, never labels or gold passages."""
    import numpy as np

    result = list(items)
    initial_count = len(result)
    for step in range(rounds):
        content = messages[-1]["content"]
        if len(result) > initial_count:
            content += "\n\nAdditional retrieved records:\n" + "\n\n".join(
                item.text for item in result[initial_count:]
            )
        content += "\n\n" + QUERY_REQUEST
        start = perf_counter()
        query = model.answer([
            {"role": "system", "content": QUERY_PROMPT},
            {"role": "user", "content": content},
        ], max_tokens=query_tokens, temperature=0.0)
        model_seconds = perf_counter() - start
        counts = dict(model.last_usage)
        start = perf_counter()
        vector = np.asarray(embed([query])[0], dtype=np.float64)
        denominator = norms * np.linalg.norm(vector)
        scores = np.divide(vectors @ vector, denominator, out=np.zeros(len(vectors)), where=denominator != 0)
        order = np.argsort(-scores, kind="stable")
        seen = {item.text for item in result}
        selected = []
        for index in order:
            index = int(index)
            text = records[index]["text"]
            if text in seen:
                continue
            seen.add(text)
            selected.append(index)
            result.append(RetrievedItem(text, float(scores[index]), {
                "memory_record_index": index, "members": records[index]["members"],
                "added_by": VARIANT, "search_round": step + 1, "search_query": query,
            }))
            if len(selected) == 5:
                break
        usage.append({"round": step + 1, "query": query, "query_prompt": QUERY_PROMPT,
                      "query_request": QUERY_REQUEST,
                      **counts, "model_seconds": model_seconds,
                      "embedding_and_ranking_seconds": perf_counter() - start,
                      "query_max_tokens": query_tokens, "query_temperature": 0.0,
                      "hit_output_limit": counts["output_tokens"] >= query_tokens,
                      "selected_indices": selected})
    return tuple(result)


class MeasuredFinalAnswer:
    def __init__(self, model, stream):
        self.model, self.stream = model, stream

    def answer(self, messages, **generation):
        start = perf_counter()
        response = self.model.answer(messages, **generation)
        self.stream.write(json.dumps({**self.model.last_usage, "seconds": perf_counter() - start,
                                      "generation_settings": generation}) + "\n")
        self.stream.flush()
        return response


def main():
    import numpy as np
    from dataclasses import asdict

    base = Path("/oscar/scratch/zliu328/agent-memory-outputs")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--union-root", type=Path, default=base / "memory_union_seed42_20260911")
    parser.add_argument("--flat-root", type=Path, default=base / "relation_memory_seed42_20260911")
    parser.add_argument("--flat-locomo-root", type=Path, default=base / "relation_memory_seed42_20260911_locomo_oom_retry")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    _seed_everything(args.seed)
    slug = args.task.replace(" ", "_")
    flat_root = args.flat_locomo_root if args.task == "LoCoMo" else args.flat_root
    memory_root = flat_root / "flat_triples_top5" / slug / "memory"
    source = args.union_root / "memory_union_all" / slug / "retrieval.jsonl"
    rows = list(_read_retrieval_records(source))
    raw = [json.loads(line) for line in source.open()]
    directory = args.output_root / VARIANT / slug
    if directory.exists():
        raise FileExistsError(directory)
    # Load both local models before creating output files; no memory rebuild or API calls.
    model = HuggingFaceChatModel(args.model, max_tokens=2048, dtype="bfloat16", device_map="cuda", seed=args.seed)
    model._ensure_loaded()
    embed = HuggingFaceEmbedder(EMBEDDING_MODEL)
    embed(["memory retrieval"])
    directory.mkdir(parents=True)
    (directory / "settings.json").write_text(json.dumps({
        "variant": VARIANT, "seed": args.seed, "model": args.model, "thinking": False,
        "query_rounds": 2, "query_max_tokens_per_round": 64, "added_facts_per_round": 5,
        "embedding_model": EMBEDDING_MODEL, "source_retrieval": str(source),
        "source_memory": str(memory_root), "preserve_all_base_evidence": True,
        "query_prompt": QUERY_PROMPT, "query_request": QUERY_REQUEST,
        "final_generation": "existing benchmark settings",
        "additional_memory_generation_calls": 0, "query_llm_calls_per_question": 2,
        "is_official_baseline": False, "is_training_free": True,
    }, indent=2) + "\n")
    choice_random = random.Random(args.seed)
    current_group = None
    with (directory / "retrieval.jsonl").open("w") as stream, (directory / "query_usage.jsonl").open("w") as usage_stream:
        for row, payload in zip(rows, raw, strict=True):
            if row.group_id != current_group:
                path = memory_root / row.group_id
                records = [json.loads(line) for line in (path / "records.jsonl").open()]
                vectors = np.asarray(np.load(path / "embeddings.npy"), dtype=np.float64)
                norms = np.linalg.norm(vectors, axis=1)
                current_group = row.group_id
            messages, _ = _answer_prompt(row.case, row.retrieved, choice_random)
            usage = []
            items = expand_evidence(messages, row.retrieved, records, vectors, norms, embed, model,
                                    usage, rounds=2, query_tokens=64)
            result = dict(payload)
            result.update(retrieved=[asdict(item) for item in items], top_k=len(items),
                          retrieval_protocol=VARIANT, retrieval_seconds=None)
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            usage_stream.write(json.dumps({"group_id": row.group_id, "case_id": row.case.case_id,
                                           "rounds": usage}, ensure_ascii=False) + "\n")
            usage_stream.flush()
    # The saved retrieval is consumed by the unchanged evaluator; query RNG cannot shift final QA RNG.
    _seed_everything(args.seed)
    output = directory / "evaluations" / args.model.replace("/", "_")
    output.mkdir(parents=True)
    with (output / "qa_usage.jsonl").open("w") as stream:
        summary = evaluate_retrieval(directory / "retrieval.jsonl", answer_model=MeasuredFinalAnswer(model, stream),
                                     output_dir=output, seed=args.seed, report_passage_metrics=False)
    print(json.dumps({"task": args.task, "variant": VARIANT, "summary": summary}), flush=True)


if __name__ == "__main__":
    main()

"""Rerun native CatRAG top-15; preserve its graph and audit all original top-5s.

Cache misses use the declared Qwen generator in-process through Transformers.
This is an explicitly recorded inference-backend change from the old vLLM
service, not a claim of bitwise replay. No answer-generation evaluation runs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from pathlib import Path
from statistics import mean
from types import SimpleNamespace


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def summarize(rows):
    if len(rows) != 1000 or len({r["case_id"] for r in rows}) != 1000:
        raise ValueError("The declared population must contain all 1,000 questions")
    summary = []
    for k in (5, 10, 15):
        pairs = [(set(row["gold"]), set(row["selected"][:k])) for row in rows]
        summary.append(dict(method="catrag", k=k, questions=len(rows),
            precision=100 * mean(len(g & r) / len(r) for g, r in pairs),
            recall=100 * mean(len(g & r) / len(g) for g, r in pairs),
            complete=100 * mean(g <= r for g, r in pairs)))
    return summary


class LocalCompletion:
    """Translate the native, fixed greedy Chat Completions request faithfully."""

    def __init__(self, model, seed, output):
        from utils.models import HuggingFaceChatModel
        self.model_id, self.seed, self.output = model, seed, output
        self.reader = HuggingFaceChatModel(model, max_tokens=2048,
            dtype="bfloat16", device_map="cuda", seed=seed)
        self.calls, self.failures = 0, []
        self.case_id = None

    def create(self, **request):
        allowed = {"model", "messages", "max_tokens", "n", "seed", "temperature", "extra_body"}
        try:
            if set(request) - allowed:
                raise ValueError(f"Unsupported request parameters: {set(request) - allowed}")
            if (request["model"] != self.model_id or request["seed"] != self.seed
                    or request["n"] != 1 or request["temperature"] != 0):
                raise ValueError("Request differs from the declared native greedy configuration")
            expected_body = {"chat_template_kwargs": {"enable_thinking": False}}
            if request.get("extra_body", expected_body) != expected_body:
                raise ValueError("Unexpected chat-template controls")
            text = self.reader.answer(request["messages"],
                max_tokens=request["max_tokens"], temperature=request["temperature"])
            self.calls += 1
            usage = self.reader.last_usage
            record = dict(case_id=self.case_id, request=request, response=text,
                usage=usage, generator_revision=self.reader._model.config._commit_hash,
                backend="Transformers greedy bfloat16, in process")
            with self.output.open("a") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            reason = "length" if usage["output_tokens"] >= request["max_tokens"] else "stop"
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=text), finish_reason=reason)],
                usage=SimpleNamespace(prompt_tokens=usage["input_tokens"],
                                      completion_tokens=usage["output_tokens"]))
        except Exception as error:
            self.failures.append(dict(case_id=self.case_id, error=repr(error)))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("Run inside the allocated GPU job")
    from baseline.catrag_memory import CatRAGMemory
    from experiments.mine_cases import manifest

    settings_path = args.memory / "adapter_settings.json"
    settings = json.loads(settings_path.read_text())
    entry = manifest(args.results_root)["one_shot", "Qwen_Qwen3.5-4B", "2WikiMultiHopQA", "catrag"]
    reference_path = Path(entry["directory"]) / "retrieval.jsonl"
    refs = read_rows(reference_path)
    if len(refs) != 1000 or len({r["case"]["case_id"] for r in refs}) != 1000:
        raise ValueError("Incomplete original retrieval population")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    design = dict(model=settings["generator_model"], embedding=settings["embedding_model"],
        seed=settings["seed"], questions=len(refs), depths=[5, 10, 15],
        memory=str(args.memory), reference=str(reference_path),
        reference_sha256=sha256(reference_path), settings_sha256=sha256(settings_path),
        backend="Native CatRAG retrieval; cached generation retained; cache misses use Transformers instead of original vLLM",
        rule="Recompute every depth from this run. Never splice old top-5 into new top-10/15.")
    (args.output / "design.json").write_text(json.dumps(design, indent=2) + "\n")
    rows = []
    with (args.output / "native_usage.jsonl").open("w") as usage:
        memory = CatRAGMemory(args.memory, usage, group_id="hipporag-2wikimultihopqa",
            generator_model=settings["generator_model"], embedding_model=settings["embedding_model"],
            generator_base_url="http://127.0.0.1:1/v1", seed=settings["seed"], load_existing=True)
        try:
            native = memory.native
            native.ent_node_to_fact_ids = {}
            native.prepare_retrieval_objects()
            native.query_to_embedding_store = str(args.output / "query_to_embedding.pkl")
            generator = native.llm_model
            original_cache = Path(generator.cache_file_name)
            private_cache = args.output / "generation_cache.sqlite"
            with sqlite3.connect(original_cache.as_uri() + "?mode=ro", uri=True) as source:
                with sqlite3.connect(private_cache) as target:
                    source.backup(target)
            generator.cache_file_name = str(private_cache)
            local = LocalCompletion(settings["generator_model"], settings["seed"],
                                    args.output / "new_generation.jsonl")
            generator.openai_client.chat.completions.create = local.create

            # Scheduling only: native cached inference and prompts are unchanged.
            def serial_batch(messages_list, max_completion_tokens=9192, **kwargs):
                return [generator.infer(messages, max_completion_tokens=max_completion_tokens, **kwargs)
                        for messages in messages_list]
            generator.batch_infer_async = serial_batch
            generator.batch_infer = serial_batch

            texts = native.chunk_embedding_store.get_all_id_to_rows()
            ids = {r["content"]: key for key, r in texts.items()}
            if len(ids) != len(texts):
                raise ValueError("Ambiguous source identity")
            for ref in refs:
                case = ref["case"]
                local.case_id = case["case_id"]
                items = memory.retrieve(case["question"], 15, case_id=local.case_id)
                if local.failures:
                    raise RuntimeError(f"Native code swallowed provider errors: {local.failures}")
                selected = [ids[r.text] for r in items]
                if len(selected) != len(set(selected)) or len(selected) != 15:
                    raise ValueError("Native retrieval did not return 15 distinct sources")
                row = dict(case_id=local.case_id, question=case["question"],
                    gold=[ids[t] for t in case["gold_passages"]], selected=selected,
                    original_top_five=[ids[r["text"]] for r in ref["retrieved"]])
                rows.append(row)
                with (args.output / "retrieval.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(json.dumps(dict(completed=len(rows), total=len(refs),
                    new_generator_calls=local.calls,
                    current_top5_matches=selected[:5] == row["original_top_five"])), flush=True)
            summary = summarize(rows)
            with (args.output / "depth_catrag.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
                writer.writeheader()
                writer.writerows(summary)
            differences = [r["case_id"] for r in rows if r["selected"][:5] != r["original_top_five"]]
            audit = dict(complete=True, questions=len(rows), new_generation_calls=local.calls,
                different_top5_order=differences,
                different_top5_sets=[r["case_id"] for r in rows
                    if set(r["selected"][:5]) != set(r["original_top_five"])],
                retrieval_sha256=sha256(args.output / "retrieval.jsonl"),
                protocol=design, summary=summary)
            (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
            print(json.dumps(audit), flush=True)
        finally:
            memory.close()


if __name__ == "__main__":
    main()

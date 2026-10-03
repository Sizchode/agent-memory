"""Build official reference memories on the retained benchmark tasks."""

import argparse
import json
import os
import shutil
from pathlib import Path
from time import perf_counter

from baseline.reference_memory import AnchorMemBuilder, StructMemBuilder
from main import build_parser, _config_from_args, _load_groups, _seed_everything
from utils.models import release_accelerator_memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", choices=["anchormem", "structmem"], required=True)
    parser.add_argument("--task", default="LoCoMo", choices=[
        "LoCoMo", "2WikiMultiHopQA", "SH-Doc QA", "MH-Doc QA",
        "FactConsolidation-SH", "FactConsolidation-MH",
    ])
    parser.add_argument("--path")
    parser.add_argument("--data-root")
    parser.add_argument("--recover-from", type=Path,
                        help="Existing LoCoMo build directory; recover events using cached responses only.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generator-base-url", required=True)
    parser.add_argument("--generator-model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-index", type=int,
                        help="One official conversation for Slurm sharding; omit to build all.")
    parser.add_argument("--input-protocol", choices=["official-locomo-all-turns", "benchmark-documents"], required=True,
                        help="Explicit selection; do not infer document timestamps or use gold evidence.")
    args = parser.parse_args()
    _seed_everything(args.seed)
    if not os.environ.get("VLLM_API_KEY"):
        raise SystemExit("VLLM_API_KEY must be set for the local server")
    # AnchorMem uses the OpenAI-compatible client environment variable.
    os.environ["OPENAI_API_KEY"] = os.environ["VLLM_API_KEY"]
    if (args.task == "LoCoMo") != (args.input_protocol == "official-locomo-all-turns"):
        parser.error("Select the LoCoMo formatter for LoCoMo and benchmark-documents for document tasks")
    if args.baseline == "structmem" and args.task != "LoCoMo":
        parser.error("StructMem is not approved for document tasks")
    if args.recover_from and (args.baseline != "anchormem" or args.task != "LoCoMo"):
        parser.error("Event recovery is for the existing AnchorMem LoCoMo build")
    data_args = []
    if args.path:
        data_args += ["--path", args.path]
    if args.data_root:
        data_args += ["--data-root", args.data_root]
    shared_args = build_parser().parse_args([
        "--task", args.task, "--baseline", "lightmem", "--output-dir", str(args.output_dir),
        "--generator-model", args.generator_model,
        "--generator-base-url", args.generator_base_url, "--seed", str(args.seed),
        "--chunk-size", "512", *data_args,
    ])
    shared = _config_from_args(shared_args)
    project = Path(__file__).resolve().parents[1]
    lightmem_config = json.loads((project / "experiments/configs/lightmem.json").read_text())
    resolved = shared.lightmem_config(lightmem_config)
    resolved["embedding_retriever"]["configs"]["embedding_model_dims"] = shared.embedding_dimensions
    raw = ({sample["sample_id"]: sample for sample in json.loads(Path(args.path).read_text())}
           if args.task == "LoCoMo" else {})
    groups = list(_load_groups(shared_args))
    if args.group_index is not None:
        if not 0 <= args.group_index < len(groups):
            raise SystemExit("--group-index is outside the released conversation list")
        groups = [groups[args.group_index]]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "settings.json").write_text(json.dumps({
        "baseline": args.baseline, "task": args.task, "input_protocol": args.input_protocol,
        "generator_model": args.generator_model, "embedding_model": shared.embedding_model,
        "seed": args.seed, "phase": "memory-build-only",
        "group_ids": [group.group_id for group in groups],
        "recover_from": str(args.recover_from) if args.recover_from else None,
    }, indent=2) + "\n")
    with (args.output_dir / "construction.jsonl").open("w") as stream:
        for group in groups:
            stream.write(json.dumps({"group_id": group.group_id, "status": "started"}) + "\n")
            stream.flush()
            builder = None
            started = perf_counter()
            try:
                destination = args.output_dir / group.group_id
                if args.baseline == "anchormem":
                    if args.recover_from:
                        source = args.recover_from / group.group_id
                        # Preserve original generation and embeddings. Only the
                        # incorrectly parsed event layer is rebuilt in a new copy.
                        shutil.copytree(source, destination,
                                        ignore=shutil.ignore_patterns("event_embeddings"))
                        fact_path = destination / ("fact_results_" + args.generator_model.replace("/", "_") + ".json")
                        payload = json.loads(fact_path.read_text())
                        payload["events"] = []
                        for doc in payload["docs"]:
                            doc["extracted_events"] = []
                        fact_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n")
                    builder = AnchorMemBuilder(
                        output_dir=destination, generator_model=args.generator_model,
                        generator_base_url=args.generator_base_url,
                        embedding_model=shared.embedding_model, seed=args.seed,
                        cache_only=bool(args.recover_from),
                    )
                    times = (builder.build_conversation(raw[group.native_sample.sample_id])
                             if args.task == "LoCoMo" else builder.build_documents(list(group.memory_items)))
                else:
                    builder = StructMemBuilder(resolved, group=group, output_dir=destination)
                    times = builder.build(group.memory_items)
                record = {"group_id": group.group_id, "status": "completed", **times}
            except BaseException as exc:
                record = {"group_id": group.group_id, "status": "failed",
                          "error": f"{type(exc).__name__}: {exc}"}
                raise
            finally:
                record["elapsed_seconds_including_initialization"] = perf_counter() - started
                record["generator_statistics"] = builder.efficiency_metrics() if builder else None
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                if builder:
                    builder.close()
                del builder
                release_accelerator_memory()


if __name__ == "__main__":
    main()

"""Copy generated LightMem stores, run official offline updates, then retrieve."""

import json
from pathlib import Path
import shutil

from main import (
    _config_from_args, _create_existing_baseline, _load_groups, _read_config,
    _seed_everything, build_parser,
)
from experiments.runner import retrieve_existing_memory_groups
from utils.models import release_accelerator_memory


def main():
    args = build_parser().parse_args()
    if args.baseline != "lightmem" or not args.memory_input_dir:
        raise SystemExit("Requires --baseline lightmem and --memory-input-dir")
    source = Path(args.memory_input_dir).resolve()
    output = Path(args.output_dir).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise SystemExit("Use a separate output directory outside the source memory tree")
    _seed_everything(args.seed)
    config = _config_from_args(args)
    official_config = _read_config(args.official_config)
    groups = list(_load_groups(args))
    output.mkdir(parents=True, exist_ok=True)
    if (output / "consolidation.jsonl").exists():
        raise SystemExit("Output already contains an offline run; use a new output directory")
    with (output / "consolidation.jsonl").open("w", encoding="utf-8") as progress, (
        output / "memory_changes.jsonl"
    ).open("w", encoding="utf-8") as changes:
        for group in groups:
            relative = Path("lightmem_indices") / group.group_id
            if not (source / relative).is_dir():
                raise FileNotFoundError(source / relative)
            shutil.copytree(source / relative, output / relative)
            progress.write(json.dumps({"group_id": group.group_id, "status": "started"}) + "\n")
            progress.flush()
            memory = _create_existing_baseline("lightmem", config, official_config, group, output)
            try:
                before = {r["id"]: r["payload"] for r in memory.memory_records()}
                print(f"Offline update: {group.group_id}, {len(before)} existing entries", flush=True)
                times = memory.consolidate()
                after = {r["id"]: r["payload"] for r in memory.memory_records()}
                deleted = updated = 0
                for identifier, payload in before.items():
                    new = after.get(identifier)
                    if new is None or new["memory"] != payload["memory"]:
                        deleted += new is None
                        updated += new is not None
                        changes.write(json.dumps({
                            "group_id": group.group_id, "memory_id": identifier,
                            "action": "delete" if new is None else "update",
                            "before": payload, "after": new,
                        }, ensure_ascii=False) + "\n")
                changes.flush()
                progress.write(json.dumps({
                    "group_id": group.group_id, "status": "completed",
                    "input_memory_entries": len(before), "output_memory_entries": len(after),
                    "deleted_entries": deleted, "updated_entries": updated,
                    **times, "generator_statistics": memory.efficiency_metrics(),
                }, ensure_ascii=False) + "\n")
                progress.flush()
            except BaseException as exc:
                progress.write(json.dumps({
                    "group_id": group.group_id, "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "generator_statistics": memory.efficiency_metrics(),
                }) + "\n")
                progress.flush()
                raise
            finally:
                memory.close()
                del memory
                release_accelerator_memory()
    # Existing runner and official cases/metrics remain unchanged. No build()
    # or add_memory() is called in either stage of this entry point.
    retrieve_existing_memory_groups(
        groups,
        create_baseline=lambda group: _create_existing_baseline(
            "lightmem", config, official_config, group, output,
        ),
        top_k=config.final_retrieval_top_k,
        output_path=output / "retrieval.jsonl",
    )


if __name__ == "__main__":
    main()

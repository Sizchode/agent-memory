"""Execute unchanged component science locally; persist only bounded verified archives."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from experiments.component_artifacts import artifact_paths, home_quota_guard, publish_tree, restore_tree, verify_archive
from experiments.encoder_robustness import digest, json_save
from experiments.final_component_ablation import TASKS, READERS, DEFAULT_RESULTS, REFERENCE, prepare, qa

BUDGET = 1024**3


def check_quota(additional):
    result = subprocess.run(["checkquota"], check=True, capture_output=True, text=True)
    audit = home_quota_guard(result.stdout, additional)
    print("Storage gate", json.dumps(audit), flush=True)


def run(args):
    if not args.output.resolve().is_relative_to(Path.home().resolve()):
        raise ValueError("This compact runner's quota guard requires a home-directory output")
    check_quota(0)
    args.output.mkdir(parents=True, exist_ok=True)
    json_save(args.output / "storage_protocol.json", dict(
        archive_budget_bytes=BUDGET, home_quota_reserve_gb=1,
        scheme="Lossless stage archives; per-file SHA256, atomic publication, shared budget lock; uncompressed files job-local",
        tasks=TASKS, readers=READERS, source_results=str(DEFAULT_RESULTS), reference=str(REFERENCE),
        code_sha256={name: digest(Path("experiments") / name) for name in (
            "component_artifacts.py", "compact_component_run.py", "final_component_ablation.py",
            "report_final_component_ablation.py")}))
    names = dict(prepare="prepare_" + str(args.task), qa="qa_" + str(args.model).replace("/", "_"), report="report")
    name = names[args.phase]
    archive, manifest = artifact_paths(args.output, name)
    if archive.exists() or manifest.exists():
        verify_archive(args.output, name)
        print("Verified existing stage archive", name, flush=True)
        return
    with TemporaryDirectory(prefix="amor-components-", dir=os.environ["TMPDIR"]) as temporary:
        root = Path(temporary)
        native = SimpleNamespace(output=root / "output", runtime_root=root / "runtime", results=DEFAULT_RESULTS,
            reference=REFERENCE, task=args.task, model=args.model, smoke=False)
        if args.phase == "prepare":
            prepare(native)
            includes = ["protocol.json", "groups", "inputs"]
        else:
            for task in TASKS:
                restore_tree(args.output, "prepare_" + task, native.output)
            if args.phase == "qa":
                native.smoke = True
                qa(native)
                native.smoke = False
                qa(native)
                includes = ["protocol.json", "qa", "qa_smoke"]
            else:
                for model in READERS:
                    restore_tree(args.output, "qa_" + model.replace("/", "_"), native.output)
                from experiments.report_final_component_ablation import report
                report(native.output)
                includes = ["protocol.json", "report"]
        result = publish_tree(native.output, includes, args.output, name, budget_bytes=BUDGET, quota_check=check_quota)
        # Full decompression/readback happens on the same local disk before this
        # stage is accepted, not just a compressed-file checksum.
        restore_tree(args.output, name, root / "readback")
        print("Stage complete", name, result["archive_bytes"], result["archive_sha256"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "qa", "report"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--model", choices=READERS)
    args = parser.parse_args()
    if args.phase == "prepare" and args.task is None:
        parser.error("prepare requires --task")
    if args.phase == "qa" and args.model is None:
        parser.error("qa requires --model")
    run(args)

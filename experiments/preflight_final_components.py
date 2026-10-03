"""Exercise real frozen retrieval in job-temporary storage, without QA or durable contexts."""
import argparse
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from experiments.final_component_ablation import prepare, DEFAULT_RESULTS, REFERENCE, TASKS, VARIANTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASKS, action="append")
    args = parser.parse_args()
    # Requiring TMPDIR prevents a caller from accidentally using persistent
    # experiment storage for this explicitly temporary integration check.
    temporary_root = Path(os.environ["TMPDIR"]).resolve(strict=True)
    print("Temporary integration storage", temporary_root, shutil.disk_usage(temporary_root), flush=True)
    for task in dict.fromkeys(args.task):
        with TemporaryDirectory(prefix="amor-component-preflight-", dir=temporary_root) as temporary:
            root = Path(temporary)
            prepare(SimpleNamespace(task=task, output=root / "output", runtime_root=root / "runtime",
                results=DEFAULT_RESULTS, reference=REFERENCE))
            marker = json.loads((root / "output/inputs" / (task + ".json")).read_text())
            if set(marker["retrieval_sha256"]) != {"reference", *VARIANTS}:
                raise ValueError("Integration check did not cover all component conditions")
            print(json.dumps(dict(integration_complete=True, task=task, marker=marker,
                temporary_bytes=sum(p.stat().st_size for p in root.rglob("*") if p.is_file()),
                note="Real complete task retrieval only; temporary artifacts removed; not final QA results")), flush=True)


if __name__ == "__main__":
    main()

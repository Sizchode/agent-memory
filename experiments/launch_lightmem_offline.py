"""Launch six LightMem offline tasks and their dependent three-backbone QA jobs."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess


def submit(arguments, project):
    response = subprocess.check_output(["sbatch", "--parsable", *arguments], cwd=project, text=True)
    return response.strip().split(";")[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-experiment", required=True)
    parser.add_argument("--experiment-id", default="lightmem_offline_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--output-root", type=Path, default=Path("/oscar/scratch/zliu328/agent-memory-outputs"))
    parser.add_argument("--gpu-resource", default="nvidia_b200")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    tasks = ["SH-Doc_QA", "MH-Doc_QA", "FactConsolidation-SH", "FactConsolidation-MH", "LoCoMo", "2WikiMultiHopQA"]
    source = args.output_root / args.memory_experiment / "lightmem"
    for task in tasks:
        if not (source / task / "lightmem_indices").is_dir():
            raise FileNotFoundError(source / task / "lightmem_indices")
    destination = args.output_root / args.experiment_id
    destination.mkdir(parents=True, exist_ok=False)
    script = str(project / "experiments/run_experiments.sh")
    common = (
        f"ALL,PROJECT_DIR={project},OUTPUT_ROOT={args.output_root},EXPERIMENT_ID={args.experiment_id},"
        "SEED=42,RETRIEVAL_TOP_K=5"
    )
    offline = submit([
        "--job-name=lightmem_offline", "--partition=gpu-he", f"--gres=gpu:{args.gpu_resource}:1",
        "--array=0-5%6", "--time=12:00:00", "--cpus-per-task=8", "--mem=64G",
        f"--export={common},MEMORY_INPUT_EXPERIMENT_ID={args.memory_experiment},LIGHTMEM_OFFLINE_UPDATE=1,"
        "GENERATOR_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507,GENERATOR_MAX_MODEL_LEN=32768,"
        "LIGHTMEM_GENERATOR_MAX_TOKENS=16000,OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1",
        script, "--retrieve-existing-worker",
    ], project)
    manifest = {"experiment_id": args.experiment_id, "memory_experiment": args.memory_experiment,
                "offline_job": offline, "qa_jobs": {}}
    path = destination / "launch.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    for index, task in enumerate(tasks):
        # The shared launcher orders five methods, six tasks, three models.
        # LightMem is method index 2: evaluator array indices 36 through 53.
        first = 36 + index * 3
        qa = submit([
            "--job-name=lightmem_offline_qa", "--partition=gpu", "--gres=gpu:l40s:1",
            f"--array={first}-{first + 2}%3", f"--dependency=afterok:{offline}_{index}",
            "--time=06:00:00", "--cpus-per-task=2", "--mem=32G",
            f"--export={common},RETRIEVAL_EXPERIMENT_ID={args.experiment_id},LIGHTMEM_OFFLINE_UPDATE=0",
            script, "--evaluate-worker",
        ], project)
        manifest["qa_jobs"][task] = qa
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

"""Own a recognition server within one Slurm allocation; never reuse other jobs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import urllib.error


def server_environment(executable, inherited=None):
    """Build the child environment for the selected service runtime."""
    environment = dict(os.environ if inherited is None else inherited)
    runtime_bin = str(Path(executable).absolute().parent)
    environment["PATH"] = os.pathsep.join(
        [runtime_bin] + list(filter(None, [environment.get("PATH", "")]))
    )
    return environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder", choices=("bge", "nv"), required=True)
    args = parser.parse_args()
    model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    port = 40000 + int(os.environ["SLURM_JOB_ID"]) % 10000
    endpoint = f"http://127.0.0.1:{port}/v1"
    directory = args.output / "jobs" / os.environ["SLURM_JOB_ID"]
    command = ["/oscar/scratch/zliu328/agent-memory-envs/vllm_cu129/bin/vllm", "serve", model,
        "--host", "127.0.0.1", "--port", str(port), "--dtype", "bfloat16",
        "--tensor-parallel-size", "1", "--gpu-memory-utilization", ".90",
        "--max-model-len", "32768", "--seed", "42", "--generation-config", "vllm",
        "--safetensors-load-strategy", "lazy", "--api-key", "local-graph-generator"]
    (directory / "server_command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (directory / "recognition_server.log").open("a") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env=server_environment(command[0]))
        try:
            deadline = time.monotonic() + 1800
            ready = False
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Recognition server exited: {process.returncode}")
                request = urllib.request.Request(endpoint + "/models",
                    headers={"Authorization": "Bearer local-graph-generator"})
                try:
                    with urllib.request.urlopen(request, timeout=5) as response:
                        models = json.load(response)
                    ready = any(item["id"] == model for item in models["data"])
                except (urllib.error.URLError, TimeoutError):
                    ready = False
                if ready:
                    break
                time.sleep(5)
            if not ready:
                raise TimeoutError("Recognition service did not become ready")
            base = [sys.executable, "-u", "-m", "experiments.encoder_robustness", "retrieve",
                "--output", str(args.output), "--encoder", args.encoder, "--endpoint", endpoint]
            subprocess.run(base + ["--smoke"], check=True)
            subprocess.run(base, check=True)
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()

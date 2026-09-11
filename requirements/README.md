# UV environment layout

The official implementations require incompatible dependency versions, so the
runner, LightMem, HippoRAG, Mem0, and the optional local Qwen server use five
isolated environments. This repository does not maintain generated lock files.

Create or synchronize every environment with:

```bash
ENV_ROOT=/oscar/scratch/zliu328/agent-memory-envs requirements/create_environments.sh
```

The runner environment hosts BM25, dense retrieval, dataset conversion,
metrics, and the local Hugging Face Evaluation Backbone. Each official method
runs in its own environment. The `qwen_generator` environment serves
`Qwen/Qwen3-30B-A3B-Instruct-2507` through vLLM as the shared Generator
Backbone. This checkpoint supports only non-thinking mode.

The final launcher uses one B200 with BF16 weights for each generated-memory
task. A two-L40S fallback is available by setting
`GENERATOR_TENSOR_PARALLEL_SIZE=2`; the Slurm GPU request and tensor-parallel
size must agree, for example:

```bash
sbatch --array=12-17,24-29%3 --partition=gpu-he --gres=gpu:l40s:2 \
  --export=ALL,PROJECT_DIR="$PWD",EXPERIMENT_ID=qwen30_seed42,GENERATOR_TENSOR_PARALLEL_SIZE=2 \
  experiments/run_experiments.sh --retrieve-worker
```

Retrieval writes `retrieval.jsonl`; evaluation reads that artifact in a
separate runner process, avoiding incompatible Transformers versions in one
environment.

Each environment is installed directly from its human-edited requirements and
the corresponding official editable source tree.

#!/bin/bash
#SBATCH --job-name=amor_findings_qa
#SBATCH --partition=gpu-he
#SBATCH --account=default
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/oscar/scratch/zliu328/agent-memory-outputs/two_findings_qa_%j.out
set -euo pipefail
cd /oscar/home/zliu328/agent-memory
export PYTHONPATH="$PWD:/oscar/scratch/zliu328/agent-memory-outputs/optimization_support_ablation_seed42_20260917/experiment_code/ircot"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NLTK_DATA=/oscar/scratch/zliu328/agent-memory-envs/nltk_data
export HF_HOME=/oscar/scratch/zliu328/hf_output
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENAI_API_KEY=local-cache-only
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42
export TORCHINDUCTOR_COMPILE_THREADS=1 MAX_JOBS=1
export VLLM_CACHE_ROOT="${TMPDIR:-/tmp}/amor-findings-${SLURM_JOB_ID}"
module load cuda/12.9.0-cinr
reader_model="$1"
shift
for experiment_output in "$@"; do
    /oscar/scratch/zliu328/agent-memory-envs/ircot/bin/python -u -m experiments.recommendation_findings qa \
      --results /oscar/scratch/zliu328/agent-memory-outputs/optimization_simplified_amor_seed42_20260929 \
      --output "$experiment_output" --model "$reader_model"
done

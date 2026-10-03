#!/bin/bash
#SBATCH --job-name=catrag_depth
#SBATCH --partition=gpu-he
#SBATCH --account=default
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/zliu328/agent-memory-outputs/catrag_depth_%j.out
set -euo pipefail
cd /oscar/home/zliu328/agent-memory
export PYTHONPATH="$PWD:/oscar/scratch/zliu328/agent-memory-outputs/optimization_support_ablation_seed42_20260917/experiment_code/ircot"
export HF_HOME=/oscar/scratch/zliu328/hf_output
export NLTK_DATA=/oscar/scratch/zliu328/agent-memory-envs/nltk_data
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42 OPENAI_API_KEY=local-inprocess
module load cuda/12.9.0-cinr
exec flock -n "/tmp/${USER}.agent-lock" \
  /oscar/scratch/zliu328/agent-memory-envs/ircot/bin/python -u -m experiments.complete_catrag_depth \
  --memory /oscar/scratch/zliu328/agent-memory-outputs/catrag_hypermem_top5_seed42_20260912_h100_batch8/catrag_top5/2WikiMultiHopQA/memory/hipporag-2wikimultihopqa \
  --results-root /oscar/scratch/zliu328/agent-memory-outputs \
  --output "/oscar/scratch/zliu328/agent-memory-outputs/catrag_depth_20261001_${SLURM_JOB_ID}"

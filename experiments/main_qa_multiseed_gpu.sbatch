#!/bin/bash
#SBATCH --partition=gpu-he
#SBATCH --account=default
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --job-name=amor-main-seeds-qa
set -euo pipefail
cd /users/zliu328/agent-memory
export PYTHONPATH="$PWD:/oscar/scratch/zliu328/agent-memory-outputs/optimization_support_ablation_seed42_20260917/experiment_code/ircot"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NLTK_DATA=/oscar/scratch/zliu328/agent-memory-envs/nltk_data
export HF_HOME=/oscar/scratch/zliu328/hf_output HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false TORCHINDUCTOR_COMPILE_THREADS=1 MAX_JOBS=1
export OPENAI_API_KEY=local-cache-only PYTHONHASHSEED="$3"
export VLLM_CACHE_ROOT="${TMPDIR:?}/amor-main-seed-${SLURM_JOB_ID}"
module load cuda/12.9.0-cinr
run_root=$1
mkdir -p "$run_root/jobs/$SLURM_JOB_ID"
scontrol show job "$SLURM_JOB_ID" > "$run_root/jobs/$SLURM_JOB_ID/slurm.txt"
nvidia-smi > "$run_root/jobs/$SLURM_JOB_ID/gpu.txt"
cp experiments/main_qa_multiseed.py experiments/main_qa_multiseed_gpu.sbatch optimization/ircot.py "$run_root/jobs/$SLURM_JOB_ID/"
/oscar/scratch/zliu328/agent-memory-envs/ircot/bin/python -u -m experiments.main_qa_multiseed qa --output "$run_root" --model "$2" --seed "$3"

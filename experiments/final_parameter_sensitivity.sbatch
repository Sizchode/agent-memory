#!/bin/bash
#SBATCH --partition=gpu-he
#SBATCH --account=default
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --job-name=amor-final-sensitivity
set -euo pipefail
cd /users/zliu328/agent-memory
export PYTHONPATH="$PWD:/oscar/scratch/zliu328/agent-memory-outputs/optimization_support_ablation_seed42_20260917/experiment_code/ircot"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NLTK_DATA=/oscar/scratch/zliu328/agent-memory-envs/nltk_data
export HF_HOME=/oscar/scratch/zliu328/hf_output
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENAI_API_KEY=local-cache-only
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42
export TORCHINDUCTOR_COMPILE_THREADS=1 MAX_JOBS=1
export VLLM_CACHE_ROOT="${TMPDIR:-/tmp}/amor-final-sensitivity-${SLURM_JOB_ID}"
module load cuda/12.9.0-cinr
phase=$1
run_root=$2
selector=$3
python_bin=/oscar/scratch/zliu328/agent-memory-envs/ircot/bin/python
mkdir -p "$run_root/jobs/$SLURM_JOB_ID"
scontrol show job "$SLURM_JOB_ID" > "$run_root/jobs/$SLURM_JOB_ID/slurm.txt"
nvidia-smi > "$run_root/jobs/$SLURM_JOB_ID/gpu.txt"
cp experiments/final_parameter_sensitivity.py experiments/final_parameter_sensitivity.sbatch experiments/encoder_robustness.py "$run_root/jobs/$SLURM_JOB_ID/"
case "$phase" in
  prepare)
    "$python_bin" -u -m experiments.final_parameter_sensitivity prepare --output "$run_root" --task "$selector"
    ;;
  qa)
    "$python_bin" -u -m experiments.final_parameter_sensitivity qa --output "$run_root" --model "$selector" --smoke
    "$python_bin" -u -m experiments.final_parameter_sensitivity qa --output "$run_root" --model "$selector"
    ;;
  *) exit 2 ;;
esac

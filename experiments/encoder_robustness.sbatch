#!/bin/bash
#SBATCH --partition=gpu-he
#SBATCH --account=default
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --job-name=amor-encoder
set -euo pipefail
cd /users/zliu328/agent-memory
export PYTHONPATH="$PWD:/oscar/scratch/zliu328/agent-memory-outputs/optimization_support_ablation_seed42_20260917/experiment_code/ircot"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NLTK_DATA=/oscar/scratch/zliu328/agent-memory-envs/nltk_data
export HF_HOME=/oscar/scratch/zliu328/hf_output
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42
export TORCHINDUCTOR_COMPILE_THREADS=1 MAX_JOBS=1
export OPENAI_API_KEY=local-graph-generator
export VLLM_CACHE_ROOT="${TMPDIR:-/tmp}/amor-encoder-${SLURM_JOB_ID}"
module load cuda/12.9.0-cinr
phase=$1
run_root=$2
shift 2
python_bin=/oscar/scratch/zliu328/agent-memory-envs/ircot/bin/python
mkdir -p "$run_root/jobs/$SLURM_JOB_ID"
scontrol show job "$SLURM_JOB_ID" > "$run_root/jobs/$SLURM_JOB_ID/slurm.txt"
nvidia-smi > "$run_root/jobs/$SLURM_JOB_ID/gpu.txt"
cp experiments/encoder_robustness.py experiments/encoder_robustness.sbatch experiments/encoder_robustness_service.py "$run_root/jobs/$SLURM_JOB_ID/"
case "$phase" in
  reference)
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    "$python_bin" -u -m experiments.encoder_robustness export --output "$run_root" "$@"
    "$python_bin" -u -m experiments.encoder_robustness retrieve --output "$run_root" --encoder qwen --smoke
    "$python_bin" -u -m experiments.encoder_robustness retrieve --output "$run_root" --encoder qwen
    ;;
  encode)
    encoder=$1
    case "$encoder" in
      bge) encoder_python=/oscar/scratch/zliu328/agent-memory-envs/runner/bin/python ;;
      nv)
        encoder_env=/oscar/scratch/zliu328/agent-memory-envs/encoder_robustness_nv_20261002
        if [ ! -e "$encoder_env/pyvenv.cfg" ]; then
          /users/zliu328/.local/bin/uv venv --python /oscar/scratch/zliu328/agent-memory-envs/ircot/bin/python "$encoder_env"
        fi
        /users/zliu328/.local/bin/uv pip install --python "$encoder_env/bin/python" -r experiments/encoder_robustness_nv_requirements.txt
        encoder_python="$encoder_env/bin/python"
        ;;
      *) exit 2 ;;
    esac
    /users/zliu328/.local/bin/uv pip freeze --python "$encoder_python" > "$run_root/jobs/$SLURM_JOB_ID/packages.txt"
    "$encoder_python" -u -m experiments.encoder_robustness encode --output "$run_root" --encoder "$encoder" --smoke
    "$encoder_python" -u -m experiments.encoder_robustness encode --output "$run_root" --encoder "$encoder"
    ;;
  retrieve)
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    "$python_bin" -u -m experiments.encoder_robustness_service --output "$run_root" --encoder "$1"
    ;;
  qa)
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    "$python_bin" -u -m experiments.encoder_robustness qa --output "$run_root" --encoder "$1" --model "$2" --smoke
    "$python_bin" -u -m experiments.encoder_robustness qa --output "$run_root" --encoder "$1" --model "$2"
    ;;
  *) exit 2 ;;
esac

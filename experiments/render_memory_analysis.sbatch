#!/bin/bash
#SBATCH --job-name=render_memory_analysis
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/oscar/scratch/zliu328/agent-memory-outputs/render_memory_analysis_%j.out
set -euo pipefail
cd /oscar/home/zliu328/agent-memory
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/memory-analysis-render-${SLURM_JOB_ID}"
export PYTHONPATH="$PWD"
exec 9>"/tmp/${USER}.agent-lock"
flock -n 9
/oscar/scratch/zliu328/agent-memory-envs/ircot/bin/python -m experiments.plot_memory_findings \
 --results /oscar/scratch/zliu328/agent-memory-outputs/optimization_simplified_amor_seed42_20260929 \
 --document /oscar/home/zliu328/agent-memory/docs/memory_analysis
/oscar/scratch/zliu328/agent-memory-envs/ircot/bin/python -m experiments.plot_memory_recommendation \
 --results /oscar/scratch/zliu328/agent-memory-outputs/optimization_simplified_amor_seed42_20260929 \
 --document /oscar/home/zliu328/agent-memory/docs/memory_analysis
module load texlive/20240312-xnyd
cd docs/memory_analysis
pdflatex -interaction=nonstopmode -halt-on-error preview.tex
bibtex preview
pdflatex -interaction=nonstopmode -halt-on-error preview.tex
pdflatex -interaction=nonstopmode -halt-on-error preview.tex
pdfinfo preview.pdf
pdftoppm -scale-to 1500 -png preview.pdf preview_page >/dev/null 2>&1
if [[ -f data/fact_weighting_answer_verification.json ]]; then
 pdflatex -interaction=nonstopmode -halt-on-error supporting.tex
 pdflatex -interaction=nonstopmode -halt-on-error supporting.tex
 pdfinfo supporting.pdf
fi

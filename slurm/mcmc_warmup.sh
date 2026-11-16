#!/bin/bash
# Compile and cache the Numba kernels once, so the array tasks do not all
# compile at the same time. Submit from the repository root:
#   warmup=$(sbatch --parsable slurm/mcmc_warmup.sh)
#
# ---- PLACEHOLDERS: fill in for Adroit before submitting --------------------------
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<PARTITION>
# -----------------------------------------------------------------------------------
#SBATCH --job-name=skb-mcmc-warmup
#SBATCH --time=00:15:00
#SBATCH --mem=2G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm/logs/%x-%j.out

set -euo pipefail

# ---- PLACEHOLDER: environment activation (same as mcmc_array.sh) -------------------
# module purge
# module load anaconda3/<VERSION>
# source "$SLURM_SUBMIT_DIR/.venv/bin/activate"
# -----------------------------------------------------------------------------------

export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTHONPATH="$SLURM_SUBMIT_DIR/src"

cd "$SLURM_SUBMIT_DIR/src"
python -u -c "
from Inversion_Workflow.MCMC_Functions.MCMC_sampler import CONFIG, warm_up_numba
warm_up_numba(CONFIG)
"

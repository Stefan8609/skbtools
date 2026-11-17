#!/bin/bash
# Compile and cache the Numba kernels once, so the array tasks do not all
# compile at the same time. Submit from the repository root:
#   warmup=$(sbatch --parsable slurm/mcmc_warmup.sh)
#SBATCH --job-name=skb-mcmc-warmup
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sk8609@princeton.edu

module purge
module load anaconda3/2024.10
conda activate geodesy

set -euo pipefail

export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTHONPATH="$SLURM_SUBMIT_DIR/src"

cd "$SLURM_SUBMIT_DIR/src"
python -u -c "
from Inversion_Workflow.MCMC_Functions.MCMC_sampler import CONFIG, warm_up_numba
warm_up_numba(CONFIG)
"
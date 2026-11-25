#!/bin/bash
# Bermuda MCMC: one chain per Slurm array task.
#
# The array index sets each chain's seed, start point and output file
# (Bermuda_Data/Output/MCMC_Chains/<MCMC_RUN>/chain_<index>.npz). MCMC_RUN picks
# a variant from RUN_VARIANTS in src/Inversion_Workflow/MCMC_Functions/MCMC_sampler.py
# (iid_fixed, iid, ar1, ar1_drift); everything else is that file's CONFIG.
# Resubmitting the same command resumes every unfinished chain from its checkpoint.
#
# Submit from the repository root, after the warm-up job (README, "Running on Adroit"):
#   sbatch --dependency=afterok:<warmup job id> --export=ALL,MCMC_RUN=ar1 slurm/mcmc_array.sh
#
# --time: set from a timing test (about 220k iterations per chain, plus margin).
#SBATCH --time=12:00:00
#SBATCH --mem=4G
#SBATCH --job-name=skb-mcmc
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sk8609@princeton.edu

module purge
module load anaconda3/2024.10
conda activate geodesy

set -euo pipefail

# One thread per chain: the parallelism is across array tasks.
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg
export MCMC_RUN="${MCMC_RUN:-ar1}"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src"

echo "run $MCMC_RUN, chain $SLURM_ARRAY_TASK_ID, host $(hostname), commit $(git -C "$SLURM_SUBMIT_DIR" rev-parse --short HEAD)"
cd "$SLURM_SUBMIT_DIR/src"
python -u -m Inversion_Workflow.MCMC_Functions.MCMC_sampler

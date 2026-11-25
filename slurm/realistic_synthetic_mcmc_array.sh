#!/bin/bash
# Final-model MCMC chains on realistic synthetic surveys: the check on the
# Laplace summaries (slurm/submit_realistic_synthetic.sh submits them).
#
# Array task i runs chain i % N_CHAINS of seed FIRST_SEED + i // N_CHAINS, with
# N_SURVEYS and N_CHAINS from the config block of
# src/Inversion_Workflow/Synthetic/Realistic_Synthetic_MCMC.py (3 surveys x 2
# chains = tasks 0-5). SYN_PRESET (random | observed, default random) picks the
# ensemble preset, SYN_TRACK (bermuda | hourglass, default bermuda) the ship
# track, SYN_RUN a variant of RUN_VARIANTS in MCMC_sampler.py (default
# ar1_common30) and SYN_ONLY one error-budget group. Seed s is the same survey on
# every track and variant, so results pair by seed. Chains go to
# Bermuda_Data/Output/MCMC_Synthetic/mcmc_<run>_<preset>_<track>[_only_<group>]/.
# Resubmitting resumes unfinished chains from their checkpoints.
#
# --time: the same model on the real data takes 3.2-3.3 h per chain (440k
# iterations, one core).
#SBATCH --time=04:00:00
#SBATCH --mem=4G
#SBATCH --job-name=skb-synmcmc
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

export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTHONPATH="$SLURM_SUBMIT_DIR/src"

echo "preset ${SYN_PRESET:-random}, track ${SYN_TRACK:-bermuda}, run ${SYN_RUN:-ar1_common30}, only ${SYN_ONLY:-all}, task $SLURM_ARRAY_TASK_ID, summary ${SYN_MCMC_SUMMARY:-0}, host $(hostname), commit $(git -C "$SLURM_SUBMIT_DIR" rev-parse --short HEAD)"
cd "$SLURM_SUBMIT_DIR/src"
python -u -m Inversion_Workflow.Synthetic.Realistic_Synthetic_MCMC

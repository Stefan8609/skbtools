#!/bin/bash
# Final-model MCMC on realistic synthetic surveys (Paper I revision): accuracy and
# coverage of the model we publish, by track and by survey leg.
#
# Array task i runs chain i % N_CHAINS of seed FIRST_SEED + i // N_CHAINS, with
# PRESET, N_SURVEYS and N_CHAINS from the config block of
# src/Inversion_Workflow/Synthetic/Realistic_Synthetic_MCMC.py (defaults: "random",
# 20 surveys x 2 chains = tasks 0-39). SYN_TRACK (bermuda | circle | grid | cross,
# default bermuda) picks the ship track and SYN_RUN a variant of RUN_VARIANTS in
# MCMC_sampler.py (default ar1_common30; e.g. ar1_common30_legs, ar1_common30_dog3).
# SYN_ONLY (error budget) switches one source group on alone over the measurement
# floor (timing noise and real dropouts), or "floor" for the floor alone; unset runs
# the full preset.
# Seed s is the same survey (ocean, ship errors, prior centres) on every track and
# variant, so results pair by seed. Chains go to
# Bermuda_Data/Output/MCMC_Synthetic/mcmc_<run>_<preset>_<track>[_only_<group>][_<amplitudes>]/;
# AMPLITUDE_SET (guess | models_low | models_high) is set in the config block.
# Resubmitting resumes unfinished chains from their checkpoints.
#
# Submit from the repository root, after the Numba warm-up:
#   warmup=$(sbatch --parsable slurm/mcmc_warmup.sh)
#   sbatch --dependency=afterok:$warmup --array=0-39 \
#       --export=ALL,SYN_TRACK=bermuda,SYN_RUN=ar1_common30 \
#       slurm/realistic_synthetic_mcmc_array.sh
# then, when every chain of that run is done, the summary CSV (a few minutes):
#   sbatch --array=0 --time=00:30:00 \
#       --export=ALL,SYN_MCMC_SUMMARY=1,SYN_TRACK=bermuda,SYN_RUN=ar1_common30 \
#       slurm/realistic_synthetic_mcmc_array.sh
#
# --time: about 9 s per 1000 iterations on one core (M-series laptop, single
# thread), so 440k iterations take 1-1.5 h; 4 h leaves margin for slower nodes.
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

echo "track ${SYN_TRACK:-bermuda}, run ${SYN_RUN:-ar1_common30}, only ${SYN_ONLY:-all}, task $SLURM_ARRAY_TASK_ID, summary ${SYN_MCMC_SUMMARY:-0}, host $(hostname), commit $(git -C "$SLURM_SUBMIT_DIR" rev-parse --short HEAD)"
cd "$SLURM_SUBMIT_DIR/src"
python -u -m Inversion_Workflow.Synthetic.Realistic_Synthetic_MCMC

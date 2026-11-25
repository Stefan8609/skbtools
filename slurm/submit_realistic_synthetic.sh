#!/bin/bash
# Run the realistic synthetics, from the repository root, one phase at a time:
#   bash slurm/submit_realistic_synthetic.sh gauss_newton   Slurm, about 120 core-h
#   bash slurm/submit_realistic_synthetic.sh chains         Slurm, 18 chains of 3.3 h
#   bash slurm/submit_realistic_synthetic.sh laplace        this shell, about 7 core-h
#   bash slurm/submit_realistic_synthetic.sh summary        this shell, minutes
# "laplace" writes every final-model summary from the sampler's Laplace
# approximation. "chains" samples three surveys of three runs to check it, and
# "summary" writes their CSVs once they have finished. The last two phases need
# the project's Python active and no cluster.
set -euo pipefail
phase=${1:?phase: gauss_newton, chains, laplace or summary}

# Gauss-Newton ensembles: experiment, array tasks, seeds per task, amplitude set.
gauss_newton() {
    sbatch --array=0-$(($2 - 1)) \
        --export=ALL,RS_EXPERIMENT=$1,RS_SEEDS_PER_TASK=$3,RS_AMPLITUDES=${4:-guess} \
        slurm/realistic_synthetic_array.sh
}
if [[ $phase == gauss_newton ]]; then
    for amplitudes in guess models_low models_high; do
        gauss_newton coverage 10 100 $amplitudes  # 1000 surveys
        gauss_newton budget 2 50 $amplitudes      # 100 surveys
    done
    gauss_newton geometry 6 50   # 300 surveys
    gauss_newton halfsplit 4 50  # 200 surveys
    gauss_newton currents 1 10
fi

# Final model: run settings, then "check" for the runs that also get chains
# (N_SURVEYS x N_CHAINS = 6 array tasks each, after the Numba warm-up).
if [[ $phase == chains ]]; then
    warmup=$(sbatch --parsable slurm/mcmc_warmup.sh)
fi
final_model() {
    if [[ $phase == laplace ]]; then
        (cd src && env "$1" SYN_LAPLACE=1 PYTHONPATH=. \
            python -m Inversion_Workflow.Synthetic.Realistic_Synthetic_MCMC)
    elif [[ ${2:-} == check && $phase == summary ]]; then
        (cd src && env "$1" SYN_MCMC_SUMMARY=1 PYTHONPATH=. \
            python -m Inversion_Workflow.Synthetic.Realistic_Synthetic_MCMC)
    elif [[ ${2:-} == check && $phase == chains ]]; then
        sbatch --dependency=afterok:$warmup --array=0-5 --export=ALL,$1 \
            slurm/realistic_synthetic_mcmc_array.sh
    fi
}
# Accuracy and coverage of the published model, by track.
final_model SYN_TRACK=bermuda check
final_model SYN_TRACK=hourglass check
# The same with the dc(t) we estimated from the survey.
final_model SYN_PRESET=observed
# Do the synthetics reproduce the real subset behaviour? Each pairs by seed with
# the full bermuda run above.
final_model SYN_RUN=ar1_common30_legs check
final_model SYN_RUN=ar1_common30_half1
final_model SYN_RUN=ar1_common30_half2
# Error budget: the larger sources, one at a time over the measurement floor.
for only in floor esv_profile dc_series dc_gradient current_steady gnss_common; do
    final_model SYN_ONLY=$only
done

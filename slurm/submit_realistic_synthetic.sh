#!/bin/bash
# Submit the realistic-synthetic runs, from the repository root, one phase at a
# time. The cluster allows one user 100 queued jobs per QOS, so the 360 chains go
# in four batches: submit the next when squeue shows the last one has left.
#   bash slurm/submit_realistic_synthetic.sh gauss_newton
#   bash slurm/submit_realistic_synthetic.sh chains1    (then chains2, 3, 4)
# When every chain has finished, write the MCMC summary CSVs. That needs no
# cluster: run it wherever the chains are, with the project's Python active:
#   bash slurm/submit_realistic_synthetic.sh summary
# Gauss-Newton ensembles take about 120 core-h in all; each full-survey MCMC
# chain takes 3.2-3.3 h (12 runs, up to about 1200 core-h).
set -euo pipefail
phase=${1:?phase: gauss_newton, chains1, chains2, chains3, chains4 or summary}

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

# Final-model MCMC: batch, settings, then the number of surveys (default 20; two
# chains each, so twice as many array tasks). The first batch compiles the Numba
# kernels before its chains start.
dependency=""
if [[ $phase == chains1 ]]; then
    dependency="--dependency=afterok:$(sbatch --parsable slurm/mcmc_warmup.sh)"
fi
mcmc() {
    if [[ $phase == summary ]]; then
        (cd src && env "$2" SYN_MCMC_SUMMARY=1 PYTHONPATH=. \
            python -m Inversion_Workflow.Synthetic.Realistic_Synthetic_MCMC)
    elif [[ $phase == chains$1 ]]; then
        sbatch $dependency --array=0-$((2 * ${3:-20} - 1)) --export=ALL,$2 \
            slurm/realistic_synthetic_mcmc_array.sh
    fi
}
# Accuracy and coverage of the published model, by track.
mcmc 1 SYN_TRACK=bermuda
mcmc 1 SYN_TRACK=hourglass
# The same with the dc(t) we estimated from the survey.
mcmc 2 SYN_PRESET=observed
# Do the synthetics reproduce the real subset behaviour? Each pairs by seed with
# the full bermuda run above.
mcmc 2 SYN_RUN=ar1_common30_legs
mcmc 3 SYN_RUN=ar1_common30_half1
mcmc 3 SYN_RUN=ar1_common30_half2
# Error budget: the larger sources, one at a time over the measurement floor, on
# 10 surveys (error sizes, not coverage).
mcmc 2 SYN_ONLY=esv_profile 10
mcmc 3 SYN_ONLY=dc_series 10
mcmc 4 SYN_ONLY=floor 10
mcmc 4 SYN_ONLY=dc_gradient 10
mcmc 4 SYN_ONLY=current_steady 10
mcmc 4 SYN_ONLY=gnss_common 10

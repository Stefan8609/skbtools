#!/bin/bash
# Submit every realistic-synthetic run of the Paper I revision, from the
# repository root. The cluster caps the jobs one user may have queued, so each
# submission is retried every 10 minutes until it fits: run it where it survives
# a logout,
#   nohup bash slurm/submit_realistic_synthetic.sh > slurm/logs/submit.log 2>&1 &
# and, when every chain has finished, write the MCMC summary CSVs. That needs no
# cluster: run it wherever the chains are (a few minutes per run), with the
# project's Python active:
#   bash slurm/submit_realistic_synthetic.sh summary
# Gauss-Newton ensembles take about 120 core-h in all; each MCMC chain takes
# 1.5-2 h on the 17-h survey (12 runs, 360 chains, about 650 core-h).
set -euo pipefail
phase=${1:-runs}

submit() {
    until job=$(sbatch --parsable "$@" 2>&1); do
        if [[ $job != *QOSMaxSubmitJobPerUserLimit* ]]; then
            echo "$job" >&2
            exit 1
        fi
        sleep 600
    done
    echo "submitted $job: $*"
}

# Gauss-Newton ensembles: experiment, array tasks, seeds per task, amplitude set.
gauss_newton() {
    submit --array=0-$(($2 - 1)) \
        --export=ALL,RS_EXPERIMENT=$1,RS_SEEDS_PER_TASK=$3,RS_AMPLITUDES=${4:-guess} \
        slurm/realistic_synthetic_array.sh
}
if [[ $phase == runs ]]; then
    for amplitudes in guess models_low models_high; do
        gauss_newton coverage 10 100 $amplitudes  # 1000 surveys: R3 M2, R1 3a
        gauss_newton budget 2 50 $amplitudes      # 100 surveys: R1 2g, R3 M8, M1
    done
    gauss_newton geometry 6 50   # 300 surveys: R1 3e
    gauss_newton halfsplit 4 50  # 200 surveys: R4 5
    gauss_newton currents 1 10   # R2 2
    # Compile the Numba kernels before any chain starts.
    submit --wait slurm/mcmc_warmup.sh
fi

# Final-model MCMC: settings, then the number of surveys (default 20; two chains
# each, so twice as many array tasks). The "summary" phase writes the CSV of the
# finished seeds instead, in this shell.
mcmc() {
    if [[ $phase == summary ]]; then
        (cd src && env "$1" SYN_MCMC_SUMMARY=1 PYTHONPATH=. \
            python -m Inversion_Workflow.Synthetic.Realistic_Synthetic_MCMC)
    else
        submit --array=0-$((2 * ${2:-20} - 1)) --export=ALL,$1 \
            slurm/realistic_synthetic_mcmc_array.sh
    fi
}
# Accuracy and coverage of the published model, by track (R3 M2, M3, R1 3e).
for track in bermuda hourglass; do
    mcmc SYN_TRACK=$track
done
# The same with the dc(t) we estimated from the survey.
mcmc SYN_PRESET=observed
# Adequacy: do the synthetics reproduce the real subset behaviour (common
# horizontal shift, stable baselines, dt-U ridge)? Each pairs by seed with the
# full bermuda run above (R4 5).
for variant in legs half1 half2; do
    mcmc SYN_RUN=ar1_common30_$variant
done
# Error budget: the larger sources, one at a time over the measurement floor
# (R1 2g, R3 M8), on 10 surveys: it needs error sizes, not coverage. The
# Gauss-Newton budget above covers every source.
for only in floor esv_profile dc_series dc_gradient current_steady gnss_common; do
    mcmc SYN_ONLY=$only 10
done

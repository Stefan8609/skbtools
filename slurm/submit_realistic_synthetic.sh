#!/bin/bash
# Submit every realistic-synthetic run of the Paper I revision, from the
# repository root:
#   bash slurm/submit_realistic_synthetic.sh
# Gauss-Newton ensembles take about 100 core-h in all; each MCMC run is 40 chains
# of 1-1.5 h (12 runs, about 550 core-h). Comment out what is already done.
set -euo pipefail

# Gauss-Newton ensembles: experiment, array tasks, seeds per task, amplitude set.
gauss_newton() {
    sbatch --array=0-$(($2 - 1)) \
        --export=ALL,RS_EXPERIMENT=$1,RS_SEEDS_PER_TASK=$3,RS_AMPLITUDES=${4:-guess} \
        slurm/realistic_synthetic_array.sh
}
for amplitudes in guess models_low models_high; do
    gauss_newton coverage 20 50 $amplitudes  # 1000 surveys: R3 M2, R1 3a
    gauss_newton budget 5 20 $amplitudes     # 100 surveys: R1 2g, R3 M8, M1
done
gauss_newton geometry 15 20   # 300 surveys: R1 3e
gauss_newton halfsplit 10 20  # 200 surveys: R4 5
gauss_newton currents 1 10    # R2 2

# Final-model MCMC: settings, then the number of surveys (default 20; two chains
# each, so twice as many array tasks), then the summary CSV of the finished seeds.
warmup=$(sbatch --parsable slurm/mcmc_warmup.sh)
mcmc() {
    chains=$(sbatch --parsable --dependency=afterok:$warmup \
        --array=0-$((2 * ${2:-20} - 1)) \
        --export=ALL,$1 slurm/realistic_synthetic_mcmc_array.sh)
    sbatch --dependency=afterok:$chains --array=0 --time=00:30:00 \
        --export=ALL,SYN_MCMC_SUMMARY=1,$1 slurm/realistic_synthetic_mcmc_array.sh
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

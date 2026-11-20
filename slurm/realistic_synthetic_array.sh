#!/bin/bash
# Realistic synthetic ensembles (Paper I revision): one block of seeds per array task.
#
# RS_EXPERIMENT picks an experiment of experiment_cells() in
# src/Inversion_Workflow/Synthetic/Realistic_Synthetic_Ensemble.py (budget, coverage,
# geometry, currents, isolation, sanity). Task i runs seeds
# [i * RS_SEEDS_PER_TASK, (i + 1) * RS_SEEDS_PER_TASK) of every cell on
# RS_WORKERS cores and writes
# revision/analysis/realistic_synthetic/<experiment>_seed<first>.csv plus one npz
# per realization (task 0 writes <experiment>.csv). Merge and summarize afterwards,
# from the repository root:
#   PYTHONPATH=src python -c "from Inversion_Workflow.Synthetic.Realistic_Synthetic_Ensemble \
#       import merge_task_csvs, summary_table; print(summary_table(merge_task_csvs('geometry')))"
# which writes <experiment>_all.csv and <experiment>_all_summary.csv.
#
# Submit from the repository root, e.g. 1000 coverage realizations as 20 tasks:
#   sbatch --array=0-19 --export=ALL,RS_EXPERIMENT=coverage,RS_SEEDS_PER_TASK=50 \
#       slurm/realistic_synthetic_array.sh
# or 300 geometry surveys (14 cells, cross included; about 32 core-h) as 15 tasks:
#   sbatch --array=0-14 --export=ALL,RS_EXPERIMENT=geometry,RS_SEEDS_PER_TASK=20 \
#       slurm/realistic_synthetic_array.sh
#
# RS_AMPLITUDES (guess | models_low | models_high) picks the amplitudes of the current,
# M2 current and ESV gradient (AMPLITUDES in Realistic_Synthetic_Ensemble.py); output
# files and cell names carry the label unless it is "guess". Add
# RS_AMPLITUDES=models_high to --export to run the high set.
#
# --time: about 3 s per DOG and realization with the lever fixed and up to 15 s
# with it free (offset search at downsample 10), on one core.
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --job-name=skb-realsyn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sk8609@princeton.edu

module purge
module load anaconda3/2024.10
conda activate geodesy

set -euo pipefail

export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg
export RS_EXPERIMENT="${RS_EXPERIMENT:-coverage}"
export RS_SEEDS_PER_TASK="${RS_SEEDS_PER_TASK:-50}"
export RS_AMPLITUDES="${RS_AMPLITUDES:-guess}"
export RS_WORKERS="${SLURM_CPUS_PER_TASK:-10}"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src"

echo "experiment $RS_EXPERIMENT, amplitudes $RS_AMPLITUDES, task $SLURM_ARRAY_TASK_ID, host $(hostname), commit $(git -C "$SLURM_SUBMIT_DIR" rev-parse --short HEAD)"
cd "$SLURM_SUBMIT_DIR/src"
python -u -m Inversion_Workflow.Synthetic.Realistic_Synthetic_Ensemble

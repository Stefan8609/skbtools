#!/bin/bash
# Submit the Paper I revision runs on Adroit (run on the LOGIN node, from the repo root).
#
#   bash slurm/submit_revision_runs.sh [group ...]        # default: repeat oldtable synthetic
#   DRY_RUN=1 bash slurm/submit_revision_runs.sh repeat   # print the sbatch commands only
#
# Groups
#   repeat     Repeatability on the real data (R4 5), all with the final model
#              (common 30-min dc(t), AR(1)):
#                ar1_common30_dog{1,3,4}          per-DOG lever arm (FIG-8)
#                ar1_common30_half{1,2}           time halves, lever free
#                ar1_common30_half{1,2}_pinlever  time halves, lever pinned at the final value
#                ar1_common30_alt15{a,b}          geometry-balanced split (alternating 15-min blocks)
#              ar1_dog{1,3,4} (4-split model) are left out; add them to REPEAT_RUNS if wanted.
#   oldtable   Reruns of the chains sampled on the old ESV table (sha 1bc0deb1...):
#                ar1_drift ar1_splits1 (used in FIG-5). With ALL_OLD=1 also the ESV-split
#                tests ar1_splits8/16 and ar1_white_splits1/8/16.
#              Old chains are moved to Bermuda_Data/Output/MCMC_Chains_old_table/<sha8>/<run>/
#              first; the sampler now refuses to resume a chain written from other inputs.
#   synthetic  Full-size realistic synthetics (coverage 1000, budget 200, geometry 300,
#              halfsplit 300 realisations; about 140 core-h) and final-model MCMC on 20
#              synthetic surveys x 2 chains, followed by its summary job.
#
# Every MCMC array task is one chain (8 chains per run). Resubmitting resumes from
# checkpoints; finished chains exit at once.
set -euo pipefail

CANONICAL_ESV_SHA="d1369cae36d29b090221e312369e3f6520ced7d313062a05c1d268a6af01209b"
TABLE="Bermuda_Data/ESV_Tables/global_table_esv_extended.mat"
CHAINS="Bermuda_Data/Output/MCMC_Chains"
GROUPS_TO_RUN=("${@:-repeat oldtable synthetic}")
GROUPS_TO_RUN=(${GROUPS_TO_RUN[*]})

REPEAT_RUNS=(
  ar1_common30_dog1 ar1_common30_dog3 ar1_common30_dog4
  ar1_common30_half1 ar1_common30_half2
  ar1_common30_half1_pinlever ar1_common30_half2_pinlever
  ar1_common30_alt15a ar1_common30_alt15b
)
OLD_TABLE_RUNS=(ar1_drift ar1_splits1)
if [[ "${ALL_OLD:-0}" == 1 ]]; then
  OLD_TABLE_RUNS+=(ar1_splits8 ar1_splits16 ar1_white_splits1 ar1_white_splits8 ar1_white_splits16)
fi

submit() {  # echo, and run unless DRY_RUN; prints the job id
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    echo "DRY: sbatch --parsable $*" >&2
    echo "DRYRUN"
  else
    sbatch --parsable "$@"
  fi
}

# --- checks -------------------------------------------------------------------
[[ -f slurm/mcmc_array.sh ]] || { echo "run from the repository root" >&2; exit 1; }
mkdir -p slurm/logs
sha=$(sha256sum "$TABLE" | cut -d' ' -f1)
if [[ "$sha" != "$CANONICAL_ESV_SHA" ]]; then
  echo "ESV table on this machine has sha ${sha:0:8}, expected ${CANONICAL_ESV_SHA:0:8}." >&2
  echo "rsync the canonical table from your laptop first (README, 'Running on Adroit')." >&2
  exit 1
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "WARNING: uncommitted changes; chains will record git_dirty=True." >&2
fi
echo "commit $(git rev-parse --short HEAD), ESV table ${sha:0:8}, groups: ${GROUPS_TO_RUN[*]}"

# --- Numba warm-up (once, shared by every MCMC array) ---------------------------
warmup=$(submit slurm/mcmc_warmup.sh)
dep="--dependency=afterok:$warmup"
echo "warm-up job $warmup"

submit_mcmc() {
  local run=$1
  local id
  id=$(submit $dep --export=ALL,MCMC_RUN="$run" --job-name="mcmc-$run" slurm/mcmc_array.sh)
  echo "  $run -> job $id"
}

for group in "${GROUPS_TO_RUN[@]}"; do
  case "$group" in
    repeat)
      echo "repeat:"
      for run in "${REPEAT_RUNS[@]}"; do submit_mcmc "$run"; done
      ;;
    oldtable)
      echo "oldtable:"
      for run in "${OLD_TABLE_RUNS[@]}"; do
        if [[ -d "$CHAINS/$run" ]]; then
          old=$(python - "$CHAINS/$run" "$CANONICAL_ESV_SHA" <<'EOF'
import glob, json, sys
import numpy as np
shas = set()
for path in glob.glob(f"{sys.argv[1]}/chain_*.npz"):
    with np.load(path) as saved:
        files = json.loads(str(saved["metadata_json"])).get("input_files", {})
    shas.add(files.get("esv_table", {}).get("sha256", "unknown"))
other = sorted(s for s in shas if s != sys.argv[2])
print(other[0][:8] if other else "")
EOF
)
          if [[ -n "$old" ]]; then
            dest="Bermuda_Data/Output/MCMC_Chains_old_table/$old/$run"
            if [[ "${DRY_RUN:-0}" == 1 ]]; then
              echo "  DRY: mv $CHAINS/$run $dest"
            else
              mkdir -p "$(dirname "$dest")"
              [[ -e "$dest" ]] && { echo "$dest exists; resolve by hand" >&2; exit 1; }
              mv "$CHAINS/$run" "$dest"
              echo "  archived $run (table $old) -> $dest"
            fi
          fi
        fi
        submit_mcmc "$run"
      done
      ;;
    synthetic)
      echo "synthetic:"
      # experiment : array range : seeds per task (seeds = tasks x per task)
      for spec in coverage:0-19:50 budget:0-3:50 geometry:0-5:50 halfsplit:0-5:50; do
        IFS=: read -r exp range per <<<"$spec"
        id=$(submit --array="$range" --job-name="realsyn-$exp" \
          --export=ALL,RS_EXPERIMENT="$exp",RS_SEEDS_PER_TASK="$per" \
          slurm/realistic_synthetic_array.sh)
        echo "  realistic synthetic $exp ($range x $per seeds) -> job $id"
      done
      syn=$(submit $dep --array=0-39 slurm/realistic_synthetic_mcmc_array.sh)
      echo "  synthetic-survey MCMC (20 surveys x 2 chains) -> job $syn"
      summary=$(submit --dependency=afterok:"$syn" --array=0 --time=00:30:00 \
        --export=ALL,SYN_MCMC_SUMMARY=1 slurm/realistic_synthetic_mcmc_array.sh)
      echo "  synthetic MCMC summary -> job $summary (runs after $syn)"
      ;;
    *)
      echo "unknown group $group (repeat, oldtable, synthetic)" >&2
      exit 1
      ;;
  esac
done

cat <<'EOF'

After the arrays finish:
  - realistic synthetics write one CSV per task under revision/analysis/realistic_synthetic/;
    concatenate per experiment before plotting.
  - MCMC chains are in Bermuda_Data/Output/MCMC_Chains/<run>/; run the diagnostics and
    posterior summary for each run, then rsync the run folders back to the laptop.
  - Chains that hit the 12 h limit: rerun this script (or the single sbatch line); they resume.
EOF

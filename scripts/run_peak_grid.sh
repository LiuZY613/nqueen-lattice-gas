#!/bin/bash
#SBATCH --job-name=nq_peak
#SBATCH --partition=home
#SBATCH --array=0-100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=peak_%A_%a.log
#SBATCH --error=peak_%A_%a.err

# Re-run the 101-point peak grid for N=8,16,32,64,100,128.
# Each array element owns one temperature and runs all six sizes serially.

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
RANK=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
T=$(awk -v rank="$RANK" 'BEGIN { printf "%.3f", (200 + rank) / 1000 }')
BIN="$ROOT/build/mc_canonical_peak"
OUTROOT="${RUN_ROOT:-$ROOT/runs/peak_101pt}"

mkdir -p "$ROOT/build" "$OUTROOT"
if [ ! -x "$BIN" ]; then
    echo "Missing $BIN. Compile once with: gcc -O3 -Wall -o $BIN $ROOT/src/mc_canonical.c -lm" >&2
    exit 2
fi

NMEAS=${NMEAS:-100000000}
THERM=${THERM:-2000000}
NBIN=${NBIN:-200}
BASE_SEED=${BASE_SEED:-20260323}

for N in 8 16 32 64 100 128; do
    OUTDIR="$OUTROOT/N${N}"
    mkdir -p "$OUTDIR"
    SEED=$((BASE_SEED + N * 10000 + RANK))
    "$BIN" -L "$N" -N "$N" -T "$T" \
        -therm "$THERM" -nmeas "$NMEAS" -nbin "$NBIN" \
        -seed "$SEED" -max_lag 2000 -acf_interval 10 \
        > "$OUTDIR/result_N${N}_T${T}.dat"
done

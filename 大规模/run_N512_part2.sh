#!/bin/bash
#SBATCH --job-name=nqueen_N512b
#SBATCH --partition=home
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --output=N512b_%j.log
#SBATCH --error=N512b_%j.err

# ============================================================
#  N=512 Part 2 — Upper peak + mid-T + high-T
#  Complements job 11821 which handles low-T + lower peak
# ============================================================

set -e
cd ~/private/homefile/nqueen模拟/task2_N等于L

echo "=========================================="
echo "  N=512 Part 2 — $(date)"
echo "=========================================="

gcc -O3 -o mc_canonical_b mc_canonical.c -lm -Wall
echo "Compiled."

L=512
N=512
NBIN=200
BASE_SEED=20260324
MAX_PARALLEL=30

OUTDIR="N512_results"
mkdir -p "$OUTDIR"

run_temp() {
    local T=$1
    local NMEAS=$2
    local THERM=$3
    local SEED=$4
    local OUTFILE="$OUTDIR/result_T${T}.dat"

    # Skip if already completed (non-empty file)
    if [ -s "$OUTFILE" ]; then
        echo "  [SKIP] T=$T already done"
        return 0
    fi

    # Use a temp file to avoid conflict with job 1's empty files
    local TMPFILE="$OUTDIR/.tmp_b_T${T}.dat"
    ./mc_canonical_b -L $L -N $N -T "$T" \
        -therm "$THERM" -nmeas "$NMEAS" \
        -nbin $NBIN -seed "$SEED" \
        -max_lag 5000 -acf_interval 10 \
        > "$TMPFILE" 2>/dev/null

    # Only overwrite if we got results and original is still empty
    if [ -s "$TMPFILE" ]; then
        mv -f "$TMPFILE" "$OUTFILE"
        echo "  [DONE] T=$T  nmeas=$NMEAS  $(date +%H:%M:%S)"
    else
        rm -f "$TMPFILE"
    fi
}

# ============================================================
# Upper peak: T = 0.2700 to 0.5000 (step 0.005) — 47 points
# ============================================================
TEMPS_PEAK2=$(awk 'BEGIN{for(i=270;i<=500;i+=5) printf "%.4f ", i/1000}')
NMEAS_PEAK=100000000
THERM_PEAK=2000000

# Mid-T: 26 points
TEMPS_MID="0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00 1.10 1.20 1.30 1.40 1.50 1.60 1.70 1.80 1.90 2.00 2.50 3.00 3.50 4.00 4.50 5.00"
NMEAS_MID=10000000
THERM_MID=500000

# High-T: 15 points
TEMPS_HIGH="6.0 7.0 8.0 9.0 10.0 15.0 20.0 30.0 40.0 60.0 80.0 100.0 200.0 300.0 400.0"
NMEAS_HIGH=1000000
THERM_HIGH=100000

echo "Peak2:  $(echo $TEMPS_PEAK2 | wc -w) points"
echo "Mid-T:  $(echo $TEMPS_MID | wc -w) points"
echo "High-T: $(echo $TEMPS_HIGH | wc -w) points"
echo ""

SEED_IDX=0
RUNNING=0

echo "=== Upper Peak ($(date)) ==="
for T in $TEMPS_PEAK2; do
    SEED=$((BASE_SEED * 1000 + SEED_IDX))
    run_temp "$T" "$NMEAS_PEAK" "$THERM_PEAK" "$SEED" &
    SEED_IDX=$((SEED_IDX + 1))
    RUNNING=$((RUNNING + 1))
    if [ $RUNNING -ge $MAX_PARALLEL ]; then
        wait -n
        RUNNING=$((RUNNING - 1))
    fi
done

echo "=== Mid-T ($(date)) ==="
for T in $TEMPS_MID; do
    SEED=$((BASE_SEED * 1000 + SEED_IDX))
    run_temp "$T" "$NMEAS_MID" "$THERM_MID" "$SEED" &
    SEED_IDX=$((SEED_IDX + 1))
    RUNNING=$((RUNNING + 1))
    if [ $RUNNING -ge $MAX_PARALLEL ]; then
        wait -n
        RUNNING=$((RUNNING - 1))
    fi
done

echo "=== High-T ($(date)) ==="
for T in $TEMPS_HIGH; do
    SEED=$((BASE_SEED * 1000 + SEED_IDX))
    run_temp "$T" "$NMEAS_HIGH" "$THERM_HIGH" "$SEED" &
    SEED_IDX=$((SEED_IDX + 1))
    RUNNING=$((RUNNING + 1))
    if [ $RUNNING -ge $MAX_PARALLEL ]; then
        wait -n
        RUNNING=$((RUNNING - 1))
    fi
done

wait
echo ""
echo "=== Part 2 COMPLETE — $(date) ==="
echo "Non-empty results: $(cd $OUTDIR && for f in *.dat; do [ -s \"\$f\" ] && echo 1; done | wc -l)"

#!/bin/bash
#SBATCH --job-name=nqueen_N512
#SBATCH --partition=home
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --output=N512_%j.log
#SBATCH --error=N512_%j.err

# ============================================================
#  N=512 Queen Lattice Gas — Thermodynamic Integration
#  140 temperature points, max 30 parallel processes
# ============================================================

set -e
cd ~/private/homefile/nqueen模拟/task2_N等于L

echo "=========================================="
echo "  N=512 Large-Scale Simulation — $(date)"
echo "=========================================="

# Compile
echo "[0] Compiling mc_canonical (O3)..."
gcc -O3 -o mc_canonical mc_canonical.c -lm -Wall
echo "    Done."

# Global parameters
L=512
N=512
NBIN=200
BASE_SEED=20260323
MAX_PARALLEL=30

# Output directory
OUTDIR="N512_results"
mkdir -p "$OUTDIR"

# ============================================================
# Temperature grid generation
# ============================================================

# Region 1: Low-T (5 points), nmeas=10^7, therm=10^6
TEMPS_LOW="0.0500 0.0625 0.0750 0.0875 0.1000"
NMEAS_LOW=10000000
THERM_LOW=1000000

# Region 2: Peak (80 points, T=0.105 to 0.500, step 0.005), nmeas=10^8, therm=2M
NMEAS_PEAK=100000000
THERM_PEAK=2000000

# Region 3: Mid-T (26 points), nmeas=10^7, therm=500K
TEMPS_MID="0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00 1.10 1.20 1.30 1.40 1.50 1.60 1.70 1.80 1.90 2.00 2.50 3.00 3.50 4.00 4.50 5.00"
NMEAS_MID=10000000
THERM_MID=500000

# Region 4: High-T (15 points), nmeas=10^6, therm=100K
TEMPS_HIGH="6.0 7.0 8.0 9.0 10.0 15.0 20.0 30.0 40.0 60.0 80.0 100.0 200.0 300.0 400.0"
NMEAS_HIGH=1000000
THERM_HIGH=100000

# Generate peak temperatures: 0.105, 0.110, ..., 0.500
TEMPS_PEAK=$(awk 'BEGIN{for(i=105;i<=500;i+=5) printf "%.4f ", i/1000}')

echo "Low-T:  $(echo $TEMPS_LOW | wc -w) points, nmeas=$NMEAS_LOW"
echo "Peak:   $(echo $TEMPS_PEAK | wc -w) points, nmeas=$NMEAS_PEAK"
echo "Mid-T:  $(echo $TEMPS_MID | wc -w) points, nmeas=$NMEAS_MID"
echo "High-T: $(echo $TEMPS_HIGH | wc -w) points, nmeas=$NMEAS_HIGH"
echo ""

# ============================================================
# Run function
# ============================================================
run_temp() {
    local T=$1
    local NMEAS=$2
    local THERM=$3
    local SEED=$4
    local OUTFILE="$OUTDIR/result_T${T}.dat"

    if [ -f "$OUTFILE" ]; then
        echo "  [SKIP] T=$T already exists"
        return 0
    fi

    ./mc_canonical -L $L -N $N -T "$T" \
        -therm "$THERM" -nmeas "$NMEAS" \
        -nbin $NBIN -seed "$SEED" \
        -max_lag 5000 -acf_interval 10 \
        > "$OUTFILE" 2>/dev/null

    echo "  [DONE] T=$T  nmeas=$NMEAS  $(date +%H:%M:%S)"
}

# ============================================================
# Execute all regions with parallel limit
# ============================================================
SEED_IDX=0
RUNNING=0

echo "=== Region 1: Low-T ($(date)) ==="
for T in $TEMPS_LOW; do
    SEED=$((BASE_SEED * 1000 + SEED_IDX))
    run_temp "$T" "$NMEAS_LOW" "$THERM_LOW" "$SEED" &
    SEED_IDX=$((SEED_IDX + 1))
    RUNNING=$((RUNNING + 1))
    if [ $RUNNING -ge $MAX_PARALLEL ]; then
        wait -n
        RUNNING=$((RUNNING - 1))
    fi
done

echo "=== Region 2: Peak ($(date)) ==="
for T in $TEMPS_PEAK; do
    SEED=$((BASE_SEED * 1000 + SEED_IDX))
    run_temp "$T" "$NMEAS_PEAK" "$THERM_PEAK" "$SEED" &
    SEED_IDX=$((SEED_IDX + 1))
    RUNNING=$((RUNNING + 1))
    if [ $RUNNING -ge $MAX_PARALLEL ]; then
        wait -n
        RUNNING=$((RUNNING - 1))
    fi
done

echo "=== Region 3: Mid-T ($(date)) ==="
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

echo "=== Region 4: High-T ($(date)) ==="
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

# Wait for all remaining
wait
echo ""
echo "=== All simulations finished: $(date) ==="

# ============================================================
# Merge results into single sorted file
# ============================================================
echo ""
echo "[MERGE] Creating data_N512.dat ..."

MERGED="data_N512.dat"
echo "# L=512 N=512  Large-scale thermodynamic integration" > "$MERGED"
echo "# Generated: $(date)" >> "$MERGED"
echo "# T  E/N  err_E/N  Cv/N  err_Cv/N  accept_rate  E_total  tau_int" >> "$MERGED"

for f in "$OUTDIR"/result_T*.dat; do
    [ -f "$f" ] && cat "$f"
done | sort -g -k1,1 >> "$MERGED"

NPTS=$(grep -v '^#' "$MERGED" | wc -l)
echo "  Merged $NPTS temperature points into $MERGED"
echo ""
echo "=========================================="
echo "  N=512 COMPLETE — $(date)"
echo "=========================================="

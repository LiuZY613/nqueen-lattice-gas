#!/bin/bash
#SBATCH --job-name=nq_1024
#SBATCH --partition=home
#SBATCH --ntasks=280
#SBATCH --nodes=3-6
#SBATCH --cpus-per-task=1
#SBATCH --output=run_1024_%j.log
#SBATCH --error=run_1024_%j.err

# ============================================================
#  N=1024 × 280 temperatures
#  280核并行，完成后合并
# ============================================================

set -e
WORKDIR=~/private/homefile/nqueen模拟/task2_N等于L
cd "$WORKDIR"

echo "=========================================="
echo "  N=1024 × 280 temperatures — $(date)"
echo "=========================================="

# Compile
gcc -O3 -o mc_canonical_1024 mc_canonical.c -lm -Wall
echo "Compiled."

# Parameters
N=1024
NMEAS=100000000
THERM=2000000
NBIN=200
BASE_SEED=20260324
BIN="$WORKDIR/mc_canonical_1024"

# Directories
RUNDIR="$WORKDIR/run_280pt_1024"
RESULTSDIR="$RUNDIR/results"
DATADIR="$RUNDIR/data"
mkdir -p "$RESULTSDIR" "$DATADIR"

# Generate temperature list (same 280 points)
TEMP_LIST="$RUNDIR/temp_list.txt"
awk 'BEGIN{
  for(i=50;i<=100;i+=5) printf "%.4f\n",i/1000
  for(i=1025;i<=5000;i+=25) printf "%.4f\n",i/10000
  for(i=525;i<=1000;i+=25) printf "%.4f\n",i/1000
  for(i=105;i<=200;i+=5) printf "%.4f\n",i/100
  for(i=210;i<=300;i+=10) printf "%.4f\n",i/100
  for(i=310;i<=500;i+=10) printf "%.4f\n",i/100
  for(i=55;i<=100;i+=5) printf "%.4f\n",i/10
  for(i=11;i<=20;i++) printf "%.4f\n",i
  split("25 30 35 40 50 60 70 80 90 100",a," ");for(i=1;i<=10;i++) printf "%.4f\n",a[i]
  split("120 150 200 250 300 350 400 450 500",b," ");for(i=1;i<=9;i++) printf "%.4f\n",b[i]
}' > "$TEMP_LIST"

NTEMPS=$(wc -l < "$TEMP_LIST")
echo "Temperature points: $NTEMPS"

# Copy worker script
cp "$WORKDIR/worker.sh" "$RUNDIR/worker.sh"
chmod +x "$RUNDIR/worker.sh"

echo ""
echo "=== N=$N — Start $(date) ==="

SIZEDIR="$RESULTSDIR/N${N}"
mkdir -p "$SIZEDIR"

srun --ntasks=280 bash "$RUNDIR/worker.sh" \
    "$N" "$TEMP_LIST" "$SIZEDIR" "$NMEAS" "$THERM" "$NBIN" "$BASE_SEED" "$BIN"

# Merge
MERGED="$DATADIR/data_N${N}.dat"
{
    echo "# L=$N N=$N  280-point thermodynamic integration"
    echo "# nmeas=$NMEAS therm=$THERM nbin=$NBIN"
    echo "# Generated: $(date)"
    echo "# T  E/N  err_E/N  Cv/N  err_Cv/N  accept_rate  E_total  tau_int"
    for f in "$SIZEDIR"/result_N${N}_T*.dat; do
        [ -s "$f" ] && cat "$f"
    done | sort -g -k1,1
} > "$MERGED"

NPTS=$(grep -vc '^#' "$MERGED" || true)
echo "  N=$N done: $NPTS points → $MERGED — $(date)"

echo ""
echo "=========================================="
echo "  N=1024 COMPLETE — $(date)"
echo "=========================================="

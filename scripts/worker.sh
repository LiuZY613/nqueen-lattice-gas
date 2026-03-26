#!/bin/bash
# worker.sh — 单个 (N, T) 模拟任务
# 由 srun 调用，SLURM_PROCID 自动设置为 0~279

N=$1
TEMP_LIST=$2
OUTDIR=$3
NMEAS=$4
THERM=$5
NBIN=$6
BASE_SEED=$7
BIN=$8

RANK=${SLURM_PROCID}
T=$(sed -n "$((RANK+1))p" "$TEMP_LIST")

if [ -z "$T" ]; then
    exit 1
fi

SEED=$((BASE_SEED + N * 1000 + RANK))
OUTFILE="${OUTDIR}/result_N${N}_T${T}.dat"

"$BIN" -L "$N" -N "$N" -T "$T" \
    -therm "$THERM" -nmeas "$NMEAS" \
    -nbin "$NBIN" -seed "$SEED" \
    -max_lag 5000 -acf_interval 10 \
    > "$OUTFILE" 2>/dev/null

#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-/path/to/HiGGs-GBD/HiGGs-main}"
PY=/path/to/conda/envs/higgs/bin/python
cd "$ROOT"
RUNNER="$ROOT/tools/diagnostics/planar_beta_smoke_20261026.py"
test -f "$RUNNER"
GPU="$(nvidia-smi -i 2,3 \
    --query-gpu=index,memory.free \
    --format=csv,noheader,nounits |
    awk -F, '$2+0>=8192 {print $1, $2}' |
    sort -k2,2nr | awk 'NR==1 {print $1}')"
if [ -z "$GPU" ]; then
    echo "GPU 2/3 have less than 8 GiB free; not starting."
    exit 1
fi
JOB="planar_beta_mean_hog_smoke_$(date +%Y%m%d_%H%M%S_%N)"
OUT="$ROOT/outputs/$JOB"
LOG="$ROOT/logs/diagnostics/${JOB}_gpu${GPU}.log"
mkdir -p "$ROOT/logs/diagnostics"
nohup env CUDA_VISIBLE_DEVICES="$GPU" MPLBACKEND=Agg WANDB_MODE=disabled \
    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PY" -u -B "$RUNNER" "$ROOT" "$OUT" all \
    >"$LOG" 2>&1 </dev/null &
PID=$!
printf '%s\n' "$PID" >"${LOG}.pid"
printf '%s\n' "$OUT" >"${LOG}.out"
printf 'MODE=THREE_STAGE_SMOKE_ONLY\nGPU=%s\nPID=%s\nLOG=%s\nOUTPUT=%s\n' \
    "$GPU" "$PID" "$LOG" "$OUT"


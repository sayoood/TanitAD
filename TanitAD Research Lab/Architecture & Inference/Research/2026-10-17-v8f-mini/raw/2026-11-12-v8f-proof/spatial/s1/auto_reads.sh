#!/usr/bin/env bash
# auto_reads.sh [max_hours]  -- dev-box side, script polling (no model turns): for every HIER seed whose state files have arrived in spatial/s1states/
# (pulled and verified by pod/pull_arms.sh), run the two pre-registered trained-state reads of note N8 once:
#   s1_read.py run       (train probe set, in-sample)      -> RESULT_S1_<ARM>.json / VERDICT_S1_<ARM>.md
#   s1_eval_read.py run  (eval bank, out-of-sample)        -> RESULT_S1EVAL_<ARM>.json / VERDICT_S1EVAL_<ARM>.md
# One read at a time, 2 threads, CPU only; waits while free commit is under 6 GB. Ends when six HIER seeds are read, or after max_hours (default 40).
# Tools and rules are fixed (S1_DEFS.md, note N8); this script chooses nothing. The lead reads the controls block of every verdict first.
MAXH=${1:-40}
W=/f/TanitAD-scratch/v8proof/spatial; S1=$W/s1; ST=$W/s1states
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8
ts() { date '+%Y-%m-%d %H:%M:%S'; }
free_gb() { powershell -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB)" 2>/dev/null | tr -d '\r'; }
t0=$(date +%s)
cd "$S1" || exit 1
while true; do
  done_n=0
  for rec in "$ST"/HIER_s*_train.npz.json; do
    [ -f "$rec" ] || continue
    arm=$(basename "$rec" _train.npz.json)
    [ -f "$ST/${arm}_eval.npz.json" ] || continue
    ck=$(grep -o '"ckpt_sha256": "[0-9a-f]*"' "$rec" | cut -d'"' -f4)
    if [ ! -f "RESULT_S1EVAL_$arm.json" ]; then
      f=$(free_gb); if [ "${f:-0}" -lt 6 ]; then echo "[$(ts)] $arm: free commit ${f} GB, waiting"; continue; fi
      "$PY" s1_eval_read.py run --states "$ST/${arm}_eval.npz" --tag "$arm" --expect-dim 128 --ckpt-sha256 "$ck" > "auto_s1eval_$arm.log" 2>&1
      echo "[$(ts)] $arm: eval-bank read rc $?"
    fi
    if [ ! -f "RESULT_S1_$arm.json" ]; then
      f=$(free_gb); if [ "${f:-0}" -lt 6 ]; then echo "[$(ts)] $arm: free commit ${f} GB, waiting"; continue; fi
      "$PY" s1_read.py run --states "$ST/${arm}_train.npz" --tag "$arm" --expect-dim 128 --expect-state mean --ckpt-sha256 "$ck" > "auto_s1_$arm.log" 2>&1
      echo "[$(ts)] $arm: train-probe read rc $?"
    fi
    [ -f "RESULT_S1_$arm.json" ] && [ -f "RESULT_S1EVAL_$arm.json" ] && done_n=$((done_n + 1))
  done
  if [ "$done_n" -ge 6 ]; then echo "[$(ts)] six HIER seeds read: $(ls RESULT_S1_HIER_s*.json | tr '\n' ' ')"; exit 0; fi
  if [ $(( ($(date +%s) - t0) / 3600 )) -ge "$MAXH" ]; then echo "[$(ts)] TIMEOUT after $MAXH h; seeds read: $done_n"; exit 3; fi
  sleep 600
done

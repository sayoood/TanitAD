#!/bin/bash
# rehearse_cs.sh -- pod-side runner of rehearse_cs.py (card ADD113, while the rollout queue runs). CPU only, 2 threads. Banked, already open runs only.
# Usage: bash rehearse_cs.sh <kit dir> <rehearsal dir>
set -u
K=${1:?kit dir}; RH=${2:?rehearsal dir}
V=/workspace/v8proof; W=$V/integ10k; P=$V/podread; RD=$P/root/integ10k; PL=$RD/pull; CODE=$P/code
export HEADWAY_Q3DIR=$CODE/q3 HIERREP_STAGEB=$CODE/stageB TANITEVAL_ROOT=$CODE/tev HEADWAY_JOIN=$P/data/join_eval141.npz HEADWAY_BANK=$P/data/frames.npz SR_ESTREAD_DIR=$RD
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$RH"
[ -e "$RH/REHEARSE_CONTROLS.json" ] && { echo "ZZ REHEARSE STOP output exists"; exit 3; }
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
if [ "$mm" != max ]; then free_gb=$(( (mm - an) / 1073741824 )); [ "$free_gb" -ge 20 ] || { echo "ZZ REHEARSE STOP memory free ${free_gb} GB"; exit 3; }; fi
for x in I10RQ50KF5P30Hb20gGEc I10RQ50Hb20 I10RLcHb20 I10RQ50KF5Hb20 I10RLcKF5Hb20 I10RQ50KF5P60Hb20 I10RLcKF5P60Hb20 I10RQ50KF5P30Hb20 I10RLcKF5P30Hb20 I10R0Hb20; do
  for s in 0 1; do
    d=$W/t1/${x}_s$s; [ -f "$d/T1_DONE" ] || { echo "ZZ REHEARSE STOP missing run $d"; exit 3; }
    [ -e "$PL/$(basename "$d")" ] || ln -s "$d" "$PL/$(basename "$d")"
  done
done
date -u +%Y-%m-%dT%H:%M:%SZ > "$RH/REHEARSE_START.txt"
( cd "$RD" && /workspace/venv/bin/python "$RH/rehearse_cs.py" "$K" "$RH/REHEARSE_CONTROLS.json" > "$RH/rehearse.log" 2>&1 ); rc=$?
echo "ZZ REHEARSE DONE rc $rc $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RH/REHEARSE_DONE.txt"

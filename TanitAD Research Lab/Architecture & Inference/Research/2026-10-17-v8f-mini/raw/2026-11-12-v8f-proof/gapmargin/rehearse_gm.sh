#!/bin/bash
# rehearse_gm.sh -- pod-side runner of rehearse_gm.py for the reader of card ADD117 (while its rollout queue runs). CPU only, 2 threads.
# Banked, already open runs only (the stand-ins are the banked candidate runs). Deletes, moves and stops nothing.
# Usage: bash rehearse_gm.sh <rehearsal dir holding rehearse_gm.py>
set -u
RH=${1:?rehearsal dir}
V=/workspace/v8proof; W=$V/integ10k; P=$V/podread; RD=$P/root/integ10k; PL=$RD/pull; CODE=$P/code
export HEADWAY_Q3DIR=$CODE/q3 HIERREP_STAGEB=$CODE/stageB TANITEVAL_ROOT=$CODE/tev HEADWAY_JOIN=$P/data/join_eval141.npz HEADWAY_BANK=$P/data/frames.npz SR_ESTREAD_DIR=$RD
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
[ -e "$RH/REHEARSE_START.txt" ] && { echo "ZZ REHEARSE STOP already started"; exit 3; }
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
if [ "$mm" != max ]; then free_gb=$(( (mm - an) / 1073741824 )); [ "$free_gb" -ge 20 ] || { echo "ZZ REHEARSE STOP memory free ${free_gb} GB"; exit 3; }; fi
for x in I10RLcKF5Hb20 I10RQ50KF5Hb20gGEc I10RQ50KF5Hb20 I10RLcHb20 I10RQ50Hb20gGEc I10RQ50Hb20 I10R0Hb20; do
  for s in 0 1; do
    d=$W/t1/${x}_s$s; [ -f "$d/T1_DONE" ] || { echo "ZZ REHEARSE STOP missing run $d"; exit 3; }
    [ -e "$PL/$(basename "$d")" ] || ln -s "$d" "$PL/$(basename "$d")"
  done
done
date -u +%Y-%m-%dT%H:%M:%SZ > "$RH/REHEARSE_START.txt"
mkdir -p "$RH/r117"
( cd "$RD" && /workspace/venv/bin/python "$RH/rehearse_gm.py" "$V/gapmargin/kit_read/read_gapmargin10k.py" "$RH/r117/REHEARSE_CONTROLS.json" > "$RH/r117/rehearse.log" 2>&1 ); rc7=$?
echo "ZZ REHEARSE DONE rc117 $rc7 $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RH/REHEARSE_DONE.txt"

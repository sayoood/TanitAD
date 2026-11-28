#!/bin/bash
# rehearse_pa.sh -- pod-side runner of rehearse_pa.py for the readers of cards ADD114 and ADD115 (while the rollout queue of card ADD114 runs). CPU only, 2 threads.
# Banked, already open runs only. Deletes, moves and stops nothing.
# Usage: bash rehearse_pa.sh <rehearsal dir holding rehearse_pa.py>
set -u
RH=${1:?rehearsal dir}
V=/workspace/v8proof; W=$V/integ10k; P=$V/podread; RD=$P/root/integ10k; PL=$RD/pull; CODE=$P/code
export HEADWAY_Q3DIR=$CODE/q3 HIERREP_STAGEB=$CODE/stageB TANITEVAL_ROOT=$CODE/tev HEADWAY_JOIN=$P/data/join_eval141.npz HEADWAY_BANK=$P/data/frames.npz SR_ESTREAD_DIR=$RD
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
[ -e "$RH/REHEARSE_START.txt" ] && { echo "ZZ REHEARSE STOP already started"; exit 3; }
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
if [ "$mm" != max ]; then free_gb=$(( (mm - an) / 1073741824 )); [ "$free_gb" -ge 20 ] || { echo "ZZ REHEARSE STOP memory free ${free_gb} GB"; exit 3; }; fi
for x in I10RLcKF5Hb20 I10RQ50KF5Hb20gGEc I10RQ50KF5Hb20 I10RLcKF5P30Hb20 I10RQ50KF5P30Hb20aLA I10RQ50KF5P30Hb20gGEc I10RQ50KF5P30Hb20 I10RLcHb20 I10RQ50Hb20gGEc I10RQ50Hb20 I10R0Hb20; do
  for s in 0 1; do
    d=$W/t1/${x}_s$s; [ -f "$d/T1_DONE" ] || { echo "ZZ REHEARSE STOP missing run $d"; exit 3; }
    [ -e "$PL/$(basename "$d")" ] || ln -s "$d" "$PL/$(basename "$d")"
  done
done
date -u +%Y-%m-%dT%H:%M:%SZ > "$RH/REHEARSE_START.txt"
mkdir -p "$RH/r115" "$RH/r114"
( cd "$RD" && /workspace/venv/bin/python "$RH/rehearse_pa.py" "$V/presattrc/kit_read/read_presattrc10k.py" "$V/presattrc" "$V/presattrc/PRESATTRC_AFTER_START.txt" "$RH/r115/REHEARSE_CONTROLS.json" \
    '{"CONV": {"LA": "RQ50Hb20gGEc", "PL": "RLcHb20"}}' > "$RH/r115/rehearse.log" 2>&1 ); rc5=$?
( cd "$RD" && /workspace/venv/bin/python "$RH/rehearse_pa.py" "$V/presattr/kit_read/read_presattr10k.py" "$V/presattr" "$V/presattr/PRESATTR10K_START.txt" "$RH/r114/REHEARSE_CONTROLS.json" \
    '{"K0": {"LA": "RQ50KF5Hb20gGEc", "PL": "RLcKF5Hb20"}, "K30": {"PL": "RLcKF5P30Hb20"}}' > "$RH/r114/rehearse.log" 2>&1 ); rc4=$?
echo "ZZ REHEARSE DONE rc115 $rc5 rc114 $rc4 $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RH/REHEARSE_DONE.txt"

#!/bin/bash
# SFT-4 launch-gate SMOKE, memory-light (2026-10-04). The first smoke loaded the full 177,836-set bank beside SFT-1 and
# the OOM killer took SFT-1 (RESULT_SFT1.md). This one:
#  - builds a bank of only the sets that carry PDM / teacher-lane labels (smoke_bank_subset.py, nice 19);
#  - uses ONE DataLoader worker;
#  - is the OOM killer's FIRST choice (oom_score_adj 1000, inherited by the trainer);
#  - launches only when the cgroup has >= 14 GB of anon headroom.
# Same scorer_finetune.py and flags as sft4_pod.sh MODE=smoke; only --onpolicy and --workers are overridden (last wins).
set -u
source /workspace/teacher_env.sh
PY="$DRIVERL_EVAL_PYTHON"
echo 1000 > /proc/self/oom_score_adj
D=/workspace/data/refe_sft2
SB=/workspace/data/refe_sft4_smoke_sets
OUT=/workspace/data/refe_sft4
T=${SMOKE_TAG:-smoke2}            # SMOKE_TAG=smoke2_g3mut with REFE_SFT4_MUTATE_G3=1 = the G3 deliberate regression
L=$OUT/${T}_bank.log
cd /workspace/refe-sft4/smoke
nice -n 19 "$PY" smoke_bank_subset.py --sets /workspace/data/refe_navtrain/onpolicy/sets --labels $D/train_pdm \
  --labels $D/train_teacher_lane --max-per-dir 1500 --out $SB > $L 2>&1 || { echo "ZZSMOKE2_BANK_FAIL" >> $L; exit 1; }
MAX=$(cat /sys/fs/cgroup/memory.max)
for i in $(seq 1 240); do
  anon=$(awk '$1=="anon"{print $2}' /sys/fs/cgroup/memory.stat)
  [ $((MAX - anon)) -ge 14000000000 ] && break
  sleep 60
done
echo "launch: anon $anon of $MAX at $(date -u +%T)" >> $L
cd /workspace/refe-sft4/run
COMMON="--ckpt /workspace/data/refe_runs/vitl16_navtrain10_grow_tau0.3/model_final.pt --targets /workspace/data/refe_navtrain/train_grow --images /workspace/data/navtrain_pixels --calib /workspace/data/refe_navtrain/train_grow/calib_table.json --onpolicy /workspace/data/refe_navtrain/onpolicy/sets --heldout /workspace/data/refe_heldout/sets --heldout-logs /workspace/data/refe_heldout/heldout_logs.txt --workers 3 --lr 1e-4 --b-mode expreward --lam 1.0 --tau-s 0.1 --lane-head --lane-weight 1.0 --teacher-lane-train $D/train_teacher_lane --teacher-lane-heldout $D/heldout_teacher_lane --lane-labels-heldout $D/heldout_lane --pdm-labels-heldout $D/heldout_pdm"
export OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
"$PY" scorer_finetune.py $COMMON --pdm-labels-train $D/train_pdm --require-pdm --onpolicy $SB --workers 1 \
  --out $OUT/$T --preflight > $OUT/$T.log 2>&1 &
TP=$!
peak=0
while kill -0 $TP 2>/dev/null; do
  r=$(awk '/VmRSS/{print $2}' /proc/$TP/status 2>/dev/null); [ -n "$r" ] && [ "$r" -gt "$peak" ] && peak=$r
  sleep 5
done
wait $TP; rc=$?
echo "ZZEXIT $rc peak_main_rss_kb $peak" >> $OUT/$T.log

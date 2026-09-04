#!/bin/sh
# E-REFC-V4-RIG: the 5-arm tiny-rig panel. SEQUENTIAL (one GPU, and torch
# spawns ~113 threads per process -- concurrent arms make NO progress).
set -u
W=/c/Users/Admin/run_refcv4
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
DR="C:\\Users\\Admin\\tanitad-data\\physicalai\\_epcache"
export PYTHONPATH="C:\\Users\\Admin\\run_refcv4\\repo\\stack;C:\\Users\\Admin\\run_refcv4\\repo\\taniteval;C:\\Users\\Admin\\run_refcv4\\repo"
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=6
cd "$W/repo/stack" || exit 1

# HELD CONSTANT across every arm: corpus, episodes, steps, batch, lr, warmup,
# seed, size, arm, mode. Only the ego/echo levers move.
COMMON="--arm hier --size tiny --data-root $DR --episodes 48 --steps 2000 \
--batch 12 --lr 1e-4 --warmup 250 --seed 0 --log-every 50 --save-every 2000 \
--device cuda"

run () {
  name=$1; shift
  if [ -f "$W/arms/$name/summary.json" ]; then
    echo "ZZSKIP-$name-ZZ"; return 0
  fi
  echo "ZZSTART-$name-ZZ"
  # shellcheck disable=SC2086
  $PY scripts/refc_v3_train.py $COMMON --out "$W/arms/$name" "$@" \
      > "$W/arms/$name.log" 2>&1
  rc=$?
  echo "ZZDONE-$name-rc$rc-ZZ"
}

mkdir -p "$W/arms"
run A_v3
run B_v4_noguard --ego-state-inject --ego-dropout 0.0
run C_v4_drop    --ego-state-inject --ego-dropout 0.5
run D_v4_full    --ego-state-inject --ego-dropout 0.5 --echo-base
run E_regress    --ego-state-inject --ego-dropout 0.5 --echo-base --ablate-frames
echo "ZZPANEL-COMPLETE-ZZ"

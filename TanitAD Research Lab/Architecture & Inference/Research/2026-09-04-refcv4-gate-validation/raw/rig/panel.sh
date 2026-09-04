#!/bin/sh
# E-REFCV4-COMBINED-RIG — the 5-arm tiny-rig LADDER for the LAUNCHED refcv4
# lever set.  SEQUENTIAL (one GPU; torch spawns ~113 threads/process and
# concurrent arms make NO progress — MEASURED 2026-07-27).
#
# Each rung adds exactly ONE lever group over the rung above, so a regression is
# attributable.  E_regress is the DELIBERATE-REGRESSION arm (image-blind): if the
# echo gate does not FAIL it, a PASS on D_full means nothing.
#
# ⚠️ NOT isolated in this panel: the tactical-aux /2 fix is UNCONDITIONAL in this
# trainer (refc_v3_train.py `(LAT_WEIGHT / 2.0)`), so every arm including A_v3
# carries it.  It is HELD CONSTANT, not tested.
set -u
W=/c/Users/Admin/run_refcv4v
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
DR="C:\\Users\\Admin\\tanitad-data\\physicalai\\_epcache"
ANC="C:\\Users\\Admin\\run_refcv4v\\anchors_refcv4.pt"
export PYTHONPATH="C:\\Users\\Admin\\run_refcv4v\\repo\\stack;C:\\Users\\Admin\\run_refcv4v\\repo\\taniteval;C:\\Users\\Admin\\run_refcv4v\\repo"
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=6
cd "$W/repo/stack" || exit 1

STEPS=${STEPS:-2000}
ARMS=${ARMS_DIR:-$W/arms}

# HELD CONSTANT across every arm: corpus, episodes, steps, batch, lr, warmup,
# seed, size, arm, mode, device.
COMMON="--arm hier --size tiny --data-root $DR --episodes 48 --steps $STEPS \
--batch 12 --lr 1e-4 --warmup 250 --seed 0 --log-every 50 --save-every $STEPS \
--device cuda"

run () {
  name=$1; shift
  if [ -f "$ARMS/$name/summary.json" ]; then
    echo "ZZSKIP-$name-ZZ"; return 0
  fi
  echo "ZZSTART-$name-ZZ"
  # shellcheck disable=SC2086
  $PY scripts/refc_v3_train.py $COMMON --out "$ARMS/$name" "$@" \
      > "$ARMS/$name.log" 2>&1
  rc=$?
  echo "ZZDONE-$name-rc$rc-ZZ"
}

mkdir -p "$ARMS"
run A_v3
run B_ego     --ego-state-inject --ego-dropout 0.5
run C_anch    --ego-state-inject --ego-dropout 0.5 --anchors "$ANC" --sel-accel-max 2.0
run D_full    --ego-state-inject --ego-dropout 0.5 --anchors "$ANC" --sel-accel-max 2.0 --goal-str
run E_regress --ego-state-inject --ego-dropout 0.5 --anchors "$ANC" --sel-accel-max 2.0 --goal-str --ablate-frames
echo "ZZPANEL-COMPLETE-ZZ"

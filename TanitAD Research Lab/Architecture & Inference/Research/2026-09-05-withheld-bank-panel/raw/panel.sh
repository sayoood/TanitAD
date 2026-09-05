#!/bin/sh
# H-EGO-LIT-4: the 5-arm withheld-bank panel on the v7-tiny rig (+ the
# deliberate-regression GATE CONTROL last). SEQUENTIAL: one GPU, and torch
# spawns ~113 threads per process -- concurrent arms make NO progress.
# The rig is the H-ECHO-8 rig verbatim (`.../2026-09-03-refc-v4-design/raw/panel.sh`)
# plus refcv4b's SHIPPED levers (its config.json argv), minus what the epcache rig
# cannot carry (--v2-cache/--v7-labels/--nav-from-v7/--goal-str/--u8-batches).
set -u
W=/c/Users/Admin/run_wbank
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
DR="C:\\Users\\Admin\\tanitad-data\\physicalai\\_epcache"
ANC="C:\\Users\\Admin\\run_wbank\\refc_anchors_6s_v0cond_alat_117.pt"
export PYTHONPATH="C:\\Users\\Admin\\refcv4b_suite\\stack;C:\\Users\\Admin\\tanitad-wt\\taniteval"
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export OMP_NUM_THREADS=6
cd /c/Users/Admin/refcv4b_suite/stack || exit 1

# HELD CONSTANT across every arm: corpus, episodes, steps, batch, lr, warmup,
# seed, size, arm, mode, vocabulary (alat-117, md5 297f6f1d...), units, reach
# band, ego inject (+X15), NO echo-base. Only the named lever moves per arm.
COMMON="--arm hier --size tiny --data-root $DR --episodes 48 --steps 2000 \
--batch 12 --lr 1e-4 --warmup 250 --seed 0 --log-every 50 --save-every 2000 \
--device cuda \
--anchors $ANC --n-anchors 117 --anchor-v0-conditioned --anchor-control-units alat \
--sel-accel-max 2.0 --ego-state-inject"

run () {
  name=$1; shift
  if [ -f "$W/arms/$name/summary.json" ]; then
    echo "ZZSKIP-$name-ZZ"; return 0
  fi
  echo "ZZSTART-$name-$(date +%T)-ZZ"
  # shellcheck disable=SC2086
  $PY scripts/refc_v3_train.py $COMMON --out "$W/arms/$name" "$@" \
      > "$W/arms/$name.log" 2>&1
  rc=$?
  echo "ZZDONE-$name-rc$rc-$(date +%T)-ZZ"
  # verify by CONTENT, never by exit code (skill section 5)
  if [ ! -s "$W/arms/$name/ckpt.pt" ] || [ ! -s "$W/arms/$name/metrics.jsonl" ]; then
    echo "ZZBAD-$name-NO-ARTIFACT-ZZ"
  fi
}

mkdir -p "$W/arms"
run A0_fixed  --ego-dropout 0.5  --withheld-bank fixed
# N is READ FROM A0's LOG per the pre-registration (SPEC.md: 5-row running
# mean of withheld_speed_mae < 2.5 m/s, else 1/3 of the budget). A1 and A2
# share it, so A1 vs A2 is ONE variable (the speed SOURCE).
N=$($PY "$W/warmup_from_log.py" "$W/arms/A0_fixed/metrics.jsonl" 2000)
echo "ZZWARMUP-N=$N-ZZ"
echo "$N" > "$W/arms/WARMUP_N.txt"
run A1_pred   --ego-dropout 0.5  --withheld-bank pred   --withheld-bank-warmup "$N"
run A2_random --ego-dropout 0.5  --withheld-bank random --withheld-bank-warmup "$N"
run A3_drop25 --ego-dropout 0.25 --withheld-bank fixed
run A4_none   --ego-dropout 0.5  --withheld-bank none
# the deliberate-regression GATE CONTROL (skill section 2): A0 with the image
# ablated -- an echo BY CONSTRUCTION. If the gate does not FAIL it, no PASS above
# means anything. Never a model.
run A5_regress --ego-dropout 0.5 --withheld-bank fixed --ablate-frames
echo "ZZPANEL-COMPLETE-$(date +%T)-ZZ"

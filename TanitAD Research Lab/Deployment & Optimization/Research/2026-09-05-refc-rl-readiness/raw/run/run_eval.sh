#!/bin/sh
# STAGE 4-5 of the re-scoped run: the SECONDARY endpoint (four families, T1) per
# arm, then the paired read against the BANKED base over the shared ha0 floor,
# then fan safety on each arm's own emitted dump.
#
# ⛔ SECONDARY, and run only after every arm's PRIMARY (fan-safety) readout is
# banked — the priority order exists so a killed run still yields the endpoint
# the PI asked about.
# ⛔ ONE AT A TIME on the shared 4060.
set -u
PY="/c/Users/Admin/venvs/tanitad/Scripts/python.exe"
REPO="/c/Users/Admin/refcv4b_repo"
OUT="/c/Users/Admin/rl_rescope/run"
BASE_DUMP="/c/Users/Admin/_wp56/dump/refcv3_40284_dump"
EVAL_EPS="/c/Users/Admin/run_refcv3_ol/data/eval"
EVAL_LABELS="/c/Users/Admin/run_refcv3_ol/data/s2_labels_v7.2_eval.jsonl.gz"
EVAL_LEAD="/c/Users/Admin/tanitad-wt/_lead_b1/b1_eval_lead_block.npz"

export TANITAD_REPO="$REPO" PYTHONIOENCODING=utf-8 PYTHONUTF8=1 OMP_NUM_THREADS=6
export PYTHONPATH="$REPO/stack;$REPO"

STRIDE=5; NBOOT=2000; SEED=0
ARMS="${ARMS:-rl reg_echo ctrl0}"
mkdir -p "$OUT/eval"

for arm in $ARMS; do
  CK="$OUT/$arm/ckpt/ckpt_after.pt"
  if [ ! -s "$CK" ]; then echo "ZZNOCKPT-$arm-ZZ"; continue; fi
  if [ ! -s "$OUT/eval/refcv3-40284-rlmin-$arm.json" ]; then
    echo "ZZEVALSTART-$arm-$(date -u +%H:%M:%S)Z-ZZ"
    "$PY" -u "$REPO/taniteval/tools/openloop_suite.py" --ckpt "$CK" \
       --config "$OUT/$arm/ckpt/config.json" \
       --episodes "$EVAL_EPS" --labels "$EVAL_LABELS" --lead-block "$EVAL_LEAD" \
       --nav-source v72 --grid 2s --action-units steer --with-oracle-sel \
       --window-stride "$STRIDE" --n-boot "$NBOOT" --seed "$SEED" --expect-step 40284 \
       --dump-dir "$OUT/eval/${arm}_dump" --out-dir "$OUT/eval" \
       --tag "refcv3-40284-rlmin-$arm" \
       --tiers os=T1,os_navshuf=T1,os_navzero=T1,oracle_sel=T0 \
       --corpus "physicalai B1 v7.2 EVAL split, 141 clips" \
       --parity-status "NON-PARITY (B1, v2_parity false)" \
       --train-eval-disjoint "ASSERTED by rl_refcv3_min.py GATE 3 (fit n eval = 0)" \
       --strict > "$OUT/eval_$arm.log" 2>&1
    echo "ZZEVALDONE-$arm-rc$?-$(date -u +%H:%M:%S)Z-ZZ"
  else
    echo "ZZEVALSKIP-$arm-ZZ"
  fi

  # the SECONDARY endpoint: paired against the banked base over the shared floor
  if [ -d "$OUT/eval/${arm}_dump" ]; then
    "$PY" -u "$REPO/taniteval/tools/paired_openloop.py" \
       --a-dump "$BASE_DUMP" --a-name base --a-arm os \
       --b-dump "$OUT/eval/${arm}_dump" --b-name "$arm" --b-arm os \
       --floor ha0 --n-boot "$NBOOT" --seed "$SEED" \
       --out "$OUT/paired_${arm}_vs_base.json" --md "$OUT/paired_${arm}_vs_base.md" \
       >> "$OUT/eval_$arm.log" 2>&1
    echo "ZZPAIRED-$arm-rc$?-ZZ"
    # the PRIMARY endpoint on the arm's OWN emitted dump (selected paths), so the
    # fan-safety story is told on the same artifact the families are read from
    "$PY" -u "$REPO/taniteval/tools/fan_safety.py" --dump "$OUT/eval/${arm}_dump" \
       --lead-block "$EVAL_LEAD" --n-boot "$NBOOT" --seed "$SEED" \
       --out "$OUT/eval/fan_safety_${arm}.json" >> "$OUT/eval_$arm.log" 2>&1
    echo "ZZFANSAFE-$arm-rc$?-ZZ"
  fi
done
echo "ZZEVAL-COMPLETE-$(date -u +%H:%M:%S)Z-ZZ"

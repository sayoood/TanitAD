#!/bin/sh
# The VETO-ONLY arms of `H-VETO-FAN-1`, SEQUENTIAL, in PRIORITY ORDER so a killed
# run still yields value (AGENT_OPERATING_STANDARD: bank incrementally).
#
#   1. s0/ctrl_null   veto OFF, zero reward, 200  -> G1: veto_rate_mean must be EXACTLY 0.0
#   2. s0/veto200     veto ON,  zero reward, 200  -> reproduces the measured ctrl_const gain
#   3. s1/veto200     the REPLICATE (seed 1)      -> the noise floor at that dose
#   4. s0/veto2k      veto ON,  zero reward, 2000 -> the dose the brief asks for
#   5. s1/veto2k      the REPLICATE (seed 1)      -> the noise floor at that dose
#
# ONE ARM AT A TIME. torch spawns ~113 threads per process and concurrent arms make
# NO progress (MEASURED: 7 arms at 0-6 % sm for 50 min; one arm 232 s).
# The training pod `tanitad-refcv3` and Thor are never touched.
set -u
PY="/c/Users/Admin/venvs/tanitad/Scripts/python.exe"
REPO="/c/Users/Admin/refcv4b_repo"
W="/c/Users/Admin/rl_refcv3_min"
OUT="/c/Users/Admin/veto_run/run"
DRV="$REPO/stack/scripts/rl_refcv3_min.py"

export TANITAD_REPO="$REPO" PYTHONIOENCODING=utf-8 PYTHONUTF8=1 OMP_NUM_THREADS=6
export PYTHONPATH="$REPO/stack;$REPO"
export LAUNCH_APPROVED=1

CKPT="$W/base/ckpt_step40284_frozen.pt"
CONFIG="$W/base/config.json"
FIT="$W/fit120"
FIT_LABELS="/c/Users/Admin/tanitad-wt/_s2build/release/v72/s2_labels_v7.2_train.jsonl.gz"
FIT_LEAD="$W/fit120_lead_block.npz"
EVAL_EPS="/c/Users/Admin/run_refcv3_ol/data/eval"
EVAL_LABELS="/c/Users/Admin/run_refcv3_ol/data/s2_labels_v7.2_eval.jsonl.gz"
EVAL_LEAD="/c/Users/Admin/tanitad-wt/_lead_b1/b1_eval_lead_block.npz"

for f in "$CKPT" "$CONFIG" "$FIT_LABELS" "$FIT_LEAD" "$EVAL_LABELS" "$EVAL_LEAD"; do
  [ -s "$f" ] || { echo "ZZMISSING-$f-ZZ"; exit 2; }
done
mkdir -p "$OUT"

# "seed:arm" pairs, in priority order
PLAN="${PLAN:-0:ctrl_null 0:veto200 1:veto200 0:veto2k 1:veto2k}"

for item in $PLAN; do
  sd=${item%%:*}; arm=${item##*:}
  d="$OUT/s$sd"
  if [ -s "$d/$arm/arm_summary.json" ]; then echo "ZZSKIP-s$sd-$arm-ZZ"; continue; fi
  mkdir -p "$d"
  echo "ZZSTART-s$sd-$arm-$(date -u +%H:%M:%S)Z-ZZ"
  "$PY" -u "$DRV" --mode arm --lead-mode track --arm "$arm" \
      --ckpt "$CKPT" --config "$CONFIG" --expect-step 40284 \
      --episodes "$FIT" --labels "$FIT_LABELS" --lead-block "$FIT_LEAD" \
      --eval-episodes "$EVAL_EPS" --eval-labels "$EVAL_LABELS" \
      --eval-lead-block "$EVAL_LEAD" \
      --out-dir "$d" --device cuda --batch 2 --group 4 --noise 0.1 \
      --seed "$sd" --lru 6 --readout-windows 120 \
      > "$d/$arm.log" 2>&1
  rc=$?
  echo "ZZDONE-s$sd-$arm-rc$rc-$(date -u +%H:%M:%S)Z-ZZ"
  # verify by CONTENT, never by exit code
  if [ ! -s "$d/$arm/arm_summary.json" ]; then
    echo "ZZBAD-s$sd-$arm-NO-SUMMARY-ZZ"; tail -25 "$d/$arm.log"; exit 1
  fi
done
echo "ZZARMS-COMPLETE-$(date -u +%H:%M:%S)Z-ZZ"

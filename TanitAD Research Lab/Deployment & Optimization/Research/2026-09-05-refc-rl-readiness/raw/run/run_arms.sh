#!/bin/sh
# The re-scoped H-RL-MIN-1 arms, SEQUENTIAL, in PRIORITY ORDER so a killed run
# still yields value (AGENT_OPERATING_STANDARD: bank incrementally).
#
#   1. ctrl0       lr=0, 200 steps   -> validity gate V1 (weights hash-identical)
#   2. ctrl_const  const reward, 200 -> validity gate V4 (no-information value)
#   3. rl          the ONE variable, 2000
#   4. reg_echo    the deliberate regression, 2000 -> echo gate V2 (G-FAN must fire)
#
# ⛔ ONE ARM AT A TIME. torch spawns ~113 threads per process and concurrent arms
# make NO progress (MEASURED: 7 arms at 0-6 % sm for 50 min; one arm 232 s).
# ⛔ Every arm writes into its own dir and is banked the moment it exists.
set -u
PY="/c/Users/Admin/venvs/tanitad/Scripts/python.exe"
REPO="/c/Users/Admin/refcv4b_repo"
W="/c/Users/Admin/rl_refcv3_min"
OUT="/c/Users/Admin/rl_rescope/run"
DRV="$REPO/stack/scripts/rl_refcv3_min.py"

export TANITAD_REPO="$REPO" PYTHONIOENCODING=utf-8 PYTHONUTF8=1 OMP_NUM_THREADS=6
export PYTHONPATH="$REPO/stack;$REPO"
export LAUNCH_APPROVED=1          # the Master Mind approves this run; the PI asked for it

CKPT="$W/base/ckpt_step40284_frozen.pt"
CONFIG="$W/base/config.json"
FIT="$W/fit120"
# ⚠️ the v7.2 TRAIN labels live in the release tree, NOT beside the eval blob.
# Verified present (1,999,886 B) before any arm — a missing label file is a
# crash after the model has loaded, i.e. after minutes of paid start-up.
FIT_LABELS="/c/Users/Admin/tanitad-wt/_s2build/release/v72/s2_labels_v7.2_train.jsonl.gz"
FIT_LEAD="$W/fit120_lead_block.npz"
EVAL_EPS="/c/Users/Admin/run_refcv3_ol/data/eval"
EVAL_LABELS="/c/Users/Admin/run_refcv3_ol/data/s2_labels_v7.2_eval.jsonl.gz"
EVAL_LEAD="/c/Users/Admin/tanitad-wt/_lead_b1/b1_eval_lead_block.npz"

mkdir -p "$OUT"
ARMS="${ARMS:-ctrl0 ctrl_const rl reg_echo}"

for arm in $ARMS; do
  if [ -s "$OUT/$arm/arm_summary.json" ]; then
    echo "ZZSKIP-$arm-ZZ"; continue
  fi
  echo "ZZSTART-$arm-$(date -u +%H:%M:%S)Z-ZZ"
  "$PY" -u "$DRV" --mode arm --lead-mode track --arm "$arm" \
      --ckpt "$CKPT" --config "$CONFIG" --expect-step 40284 \
      --episodes "$FIT" --labels "$FIT_LABELS" --lead-block "$FIT_LEAD" \
      --eval-episodes "$EVAL_EPS" --eval-labels "$EVAL_LABELS" \
      --eval-lead-block "$EVAL_LEAD" \
      --out-dir "$OUT" --device cuda --batch 2 --group 4 --noise 0.1 \
      --seed 0 --lru 6 --readout-windows 120 \
      > "$OUT/$arm.log" 2>&1
  rc=$?
  echo "ZZDONE-$arm-rc$rc-$(date -u +%H:%M:%S)Z-ZZ"
  # verify by CONTENT, never by exit code
  if [ ! -s "$OUT/$arm/arm_summary.json" ]; then
    echo "ZZBAD-$arm-NO-SUMMARY-ZZ"
    tail -20 "$OUT/$arm.log"
    # a failed arm must not silently poison the rest: stop and let a human read it
    exit 1
  fi
done
echo "ZZARMS-COMPLETE-$(date -u +%H:%M:%S)Z-ZZ"

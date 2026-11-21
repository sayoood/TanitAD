#!/bin/sh
# The COLLISION-LEVER arms of the RL-generator-collisions WP, SEQUENTIAL, in PRIORITY
# ORDER so a killed run still yields value (AGENT_OPERATING_STANDARD: bank incrementally).
#
#   1. s0/coll200    collision weight 1.0, every other weight 0.0, veto OFF, 200 steps
#                    -> THE LEVER. one-variable against s0/ctrl_null (already banked).
#   2. s1/coll200    the REPLICATE (seed 1) -> the run-to-run noise floor at that dose
#   3. s0/ctrl0      lr = 0 -> the ONLY admissible ZERO-LEVER floor (a zeroed *loss*
#                    still moves every tensor because AdamW normalises by grad scale)
#   4. s1/ctrl_null  the zero-INFORMATION replicate (s0 is already banked)
#
# ONE ARM AT A TIME. torch spawns ~113 threads per process and concurrent arms make NO
# progress. The training pod `tanitad-refcv3` and Thor are NEVER touched.
#
# Markers are OPAQUE (ZZ...ZZ) and DISJOINT from anything the commands contain, so a
# client-side filter cannot match the PTY's echo of its own command line.
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

# VERIFY-GATE: refuse to launch against a clone that does not carry the arm. A chain
# that runs stale code is worse than one that refuses (pods have burned days on this).
grep -q '"coll200"' "$DRV" || { echo "ZZSTALE-DRIVER-NO-COLL200-ZZ"; exit 3; }
grep -q '"ctrl_null"' "$DRV" || { echo "ZZUNREADABLE-DRIVER-ZZ"; exit 3; }

mkdir -p "$OUT"
PLAN="${PLAN:-0:coll200 1:coll200 0:ctrl0 1:ctrl_null}"

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
    echo "ZZBAD-s$sd-$arm-NO-SUMMARY-ZZ"; tail -30 "$d/$arm.log"; exit 1
  fi
done
echo "ZZARMS-COMPLETE-$(date -u +%H:%M:%S)Z-ZZ"

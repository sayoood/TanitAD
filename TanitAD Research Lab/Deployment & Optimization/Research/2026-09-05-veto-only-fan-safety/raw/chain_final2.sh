#!/bin/sh
# REPLACES chain_final.sh (killed by explicit PID 29164, confirmed gone).
# Runs the CORRECTED re-rank probe -- the shrink sweep now interpolates the real
# `anchor_bank` (refc.py:1793) rather than `fan - out["offset"]`, which was the fan minus
# only the CLASSIFIER-pass offset and therefore an intermediate that exists nowhere in the
# decode -- on BOTH the base and the veto200_s0 checkpoint.
#   base        : does the 0-training kinematic gate help refcv3 AS SHIPPED?
#   veto200_s0  : do the two levers COMPOSE or CANCEL?  (the deployment question)
set -u
PY="/c/Users/Admin/venvs/tanitad/Scripts/python.exe"
REPO="/c/Users/Admin/refcv4b_repo"
W="/c/Users/Admin/rl_refcv3_min"
export TANITAD_REPO="$REPO" PYTHONIOENCODING=utf-8 PYTHONUTF8=1 OMP_NUM_THREADS=6
export PYTHONPATH="$REPO/stack;$REPO"
EVAL_EPS=/c/Users/Admin/run_refcv3_ol/data/eval
EVAL_LABELS=/c/Users/Admin/run_refcv3_ol/data/s2_labels_v7.2_eval.jsonl.gz
EVAL_LEAD=/c/Users/Admin/tanitad-wt/_lead_b1/b1_eval_lead_block.npz

while ! grep -q "ZZEVAL-COMPLETE" /c/Users/Admin/veto_run/t1_chain.log 2>/dev/null; do
  grep -q "ZZT1-ABORT" /c/Users/Admin/veto_run/t1_chain.log 2>/dev/null && { echo "ZZFINAL-ABORT-ZZ"; exit 1; }
  sleep 30
done
echo "ZZFINAL-T1-DONE-$(date -u +%H:%M:%S)Z-ZZ"

run_probe() {
  tag="$1"; ck="$2"; cfg="$3"
  [ -s "$ck" ] || { echo "ZZFINAL-NOCKPT-$tag-ZZ"; return 1; }
  [ -s "/c/Users/Admin/veto_run/raw/fan_rerank_${tag}.json" ] && { echo "ZZSKIP-rerank-$tag-ZZ"; return 0; }
  echo "ZZSTART-rerank-$tag-$(date -u +%H:%M:%S)Z-ZZ"
  "$PY" -u "$REPO/stack/scripts/rl_fan_rerank_probe.py" \
    --ckpt "$ck" --config "$cfg" --expect-step 40284 \
    --episodes "$EVAL_EPS" --labels "$EVAL_LABELS" --lead-block "$EVAL_LEAD" \
    --windows 480 --batch 4 --lru 6 --device cuda --n-boot 4000 --seed 11 \
    --out "/c/Users/Admin/veto_run/raw/fan_rerank_${tag}.json" \
    > "/c/Users/Admin/veto_run/raw/fan_rerank_${tag}.log" 2>&1
  echo "ZZDONE-rerank-$tag-rc$?-$(date -u +%H:%M:%S)Z-ZZ"
}

run_probe base "$W/base/ckpt_step40284_frozen.pt" "$W/base/config.json"
run_probe veto200s0 "/c/Users/Admin/veto_run/run/s0/veto200/ckpt/ckpt_after.pt" \
                    "/c/Users/Admin/veto_run/run/s0/veto200/ckpt/config.json"
echo "ZZFINAL-COMPLETE-$(date -u +%H:%M:%S)Z-ZZ"

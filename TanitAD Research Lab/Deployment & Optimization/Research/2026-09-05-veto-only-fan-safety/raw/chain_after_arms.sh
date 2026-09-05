#!/bin/sh
# Sequential continuation, so the 4060 is never asked to run two torch processes at once
# (7 concurrent arms once sat at 0-6 % sm for 50 minutes).
#   1. wait for the arm plan to finish
#   2. ctrl_null at 2,000 steps -> the ZERO-INFORMATION floor at the MATCHED DOSE for veto2k
#      (the 200-step null cannot floor a 2,000-step lever: nuisance drift accumulates)
#   3. the 0-training selection-rule probe (the product P1 pointed at)
set -u
PY="/c/Users/Admin/venvs/tanitad/Scripts/python.exe"
REPO="/c/Users/Admin/refcv4b_repo"
W="/c/Users/Admin/rl_refcv3_min"
OUT="/c/Users/Admin/veto_run/run"
export TANITAD_REPO="$REPO" PYTHONIOENCODING=utf-8 PYTHONUTF8=1 OMP_NUM_THREADS=6
export PYTHONPATH="$REPO/stack;$REPO" LAUNCH_APPROVED=1

while ! grep -q "ZZARMS-COMPLETE" /c/Users/Admin/veto_run/arms_run.log 2>/dev/null; do
  if grep -q "ZZBAD-" /c/Users/Admin/veto_run/arms_run.log 2>/dev/null; then
    echo "ZZCHAIN-ABORT-arms-failed-ZZ"; exit 1
  fi
  sleep 20
done
echo "ZZCHAIN-ARMS-DONE-$(date -u +%H:%M:%S)Z-ZZ"

if [ ! -s "$OUT/n2k/ctrl_null/arm_summary.json" ]; then
  echo "ZZSTART-n2k-ctrl_null-$(date -u +%H:%M:%S)Z-ZZ"
  mkdir -p "$OUT/n2k"
  "$PY" -u "$REPO/stack/scripts/rl_refcv3_min.py" --mode arm --lead-mode track --arm ctrl_null \
    --ckpt "$W/base/ckpt_step40284_frozen.pt" --config "$W/base/config.json" --expect-step 40284 \
    --episodes "$W/fit120" \
    --labels /c/Users/Admin/tanitad-wt/_s2build/release/v72/s2_labels_v7.2_train.jsonl.gz \
    --lead-block "$W/fit120_lead_block.npz" \
    --eval-episodes /c/Users/Admin/run_refcv3_ol/data/eval \
    --eval-labels /c/Users/Admin/run_refcv3_ol/data/s2_labels_v7.2_eval.jsonl.gz \
    --eval-lead-block /c/Users/Admin/tanitad-wt/_lead_b1/b1_eval_lead_block.npz \
    --out-dir "$OUT/n2k" --device cuda --batch 2 --group 4 --noise 0.1 --seed 0 --lru 6 \
    --readout-windows 120 --steps 2000 > "$OUT/n2k/ctrl_null.log" 2>&1
  echo "ZZDONE-n2k-ctrl_null-rc$?-$(date -u +%H:%M:%S)Z-ZZ"
  [ -s "$OUT/n2k/ctrl_null/arm_summary.json" ] || { echo "ZZBAD-n2k-NO-SUMMARY-ZZ"; tail -20 "$OUT/n2k/ctrl_null.log"; }
fi

if [ ! -s /c/Users/Admin/veto_run/raw/fan_rerank.json ]; then
  echo "ZZSTART-rerank-$(date -u +%H:%M:%S)Z-ZZ"
  "$PY" -u "$REPO/stack/scripts/rl_fan_rerank_probe.py" \
    --ckpt "$W/base/ckpt_step40284_frozen.pt" --config "$W/base/config.json" --expect-step 40284 \
    --episodes /c/Users/Admin/run_refcv3_ol/data/eval \
    --labels /c/Users/Admin/run_refcv3_ol/data/s2_labels_v7.2_eval.jsonl.gz \
    --lead-block /c/Users/Admin/tanitad-wt/_lead_b1/b1_eval_lead_block.npz \
    --windows 480 --batch 4 --lru 6 --device cuda --n-boot 4000 --seed 11 \
    --out /c/Users/Admin/veto_run/raw/fan_rerank.json \
    > /c/Users/Admin/veto_run/raw/fan_rerank.log 2>&1
  echo "ZZDONE-rerank-rc$?-$(date -u +%H:%M:%S)Z-ZZ"
fi
echo "ZZCHAIN-COMPLETE-$(date -u +%H:%M:%S)Z-ZZ"

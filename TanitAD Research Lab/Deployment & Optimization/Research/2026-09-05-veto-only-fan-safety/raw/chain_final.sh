#!/bin/sh
# The SHIPPING COMBINATION, measured: the 0-training kinematic gate applied to the fan the
# VETO'd checkpoint emits. The base-model probe answers "does the gate help refcv3 as
# shipped"; this answers "do the two levers compose or cancel", which is the question a
# deployment decision actually needs. ~2 min on the 4060, queued so nothing overlaps.
set -u
PY="/c/Users/Admin/venvs/tanitad/Scripts/python.exe"
REPO="/c/Users/Admin/refcv4b_repo"
W="/c/Users/Admin/rl_refcv3_min"
export TANITAD_REPO="$REPO" PYTHONIOENCODING=utf-8 PYTHONUTF8=1 OMP_NUM_THREADS=6
export PYTHONPATH="$REPO/stack;$REPO"
while ! grep -q "ZZEVAL-COMPLETE" /c/Users/Admin/veto_run/t1_chain.log 2>/dev/null; do
  grep -q "ZZT1-ABORT" /c/Users/Admin/veto_run/t1_chain.log 2>/dev/null && { echo "ZZFINAL-ABORT-ZZ"; exit 1; }
  sleep 30
done
echo "ZZFINAL-T1-DONE-$(date -u +%H:%M:%S)Z-ZZ"
CK="/c/Users/Admin/veto_run/run/s0/veto200/ckpt/ckpt_after.pt"
[ -s "$CK" ] || { echo "ZZFINAL-NOCKPT-ZZ"; exit 1; }
"$PY" -u "$REPO/stack/scripts/rl_fan_rerank_probe.py" \
  --ckpt "$CK" --config "/c/Users/Admin/veto_run/run/s0/veto200/ckpt/config.json" \
  --expect-step 40284 \
  --episodes /c/Users/Admin/run_refcv3_ol/data/eval \
  --labels /c/Users/Admin/run_refcv3_ol/data/s2_labels_v7.2_eval.jsonl.gz \
  --lead-block /c/Users/Admin/tanitad-wt/_lead_b1/b1_eval_lead_block.npz \
  --windows 480 --batch 4 --lru 6 --device cuda --n-boot 4000 --seed 11 \
  --out /c/Users/Admin/veto_run/raw/fan_rerank_veto200s0.json \
  > /c/Users/Admin/veto_run/raw/fan_rerank_veto200s0.log 2>&1
echo "ZZDONE-rerank-veto200s0-rc$?-$(date -u +%H:%M:%S)Z-ZZ"
echo "ZZFINAL-COMPLETE-$(date -u +%H:%M:%S)Z-ZZ"

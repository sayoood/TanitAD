#!/usr/bin/env bash
# L1a - THE DECIDING ONE-VARIABLE ARM. W_KAPPA, the curvature penalty.
# Against arm A (`cos_argmax`, rec_cos_argmax.json) this changes EXACTLY ONE
# thing: --cost-weights from the zeroed triple (0,0,64.297) to the SHIPPED
# (0.02, 0.05, 0.10). Same metric (cos), same plan seed (0), same episodes,
# same stride -> paired window-for-window.
# Both outcomes were committed in advance in P4_RESULT.md (commit abded60).
set -u
M="C:/Users/Admin/tanitad-wt"
OUT="C:/Users/Admin/refav1_margin/p4out"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
export PYTHONPATH="$M/stack;$M/taniteval"
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=6
mkdir -p "$OUT"
LBL=C:/Users/Admin/navcomp/data/s2_labels_v7.2_eval.jsonl.gz
CACHE=C:/Users/Admin/refav1_margin/p4/fp8
EPS=C:/Users/Admin/refav1_margin/p4/eps
echo "ZZL1A-START-$(date -u +%FT%TZ)ZZ"
"$PY" "$M/taniteval/tools/refav1_arm.py" \
  --ckpt C:/Users/Admin/refav1_eval_slice/ckpt_ep3/ckpt.pt \
  --config C:/Users/Admin/refav1_eval_slice/ckpt_ep3/config.json \
  --cache "$CACHE" --episodes "$EPS" --labels "$LBL" --nav "$LBL" \
  --device cuda --episodes-n 0 --window-stride 16 --no-navshuf \
  --no-lead-block --cost-metric cos \
  --cost-weights 0.02,0.05,0.1 \
  --plan-seed 0 \
  --dump-dir "$OUT/dump_cos_wk" --out "$OUT/rec_cos_wk.json" \
  --arm "refav1-21109-p4-cos_shippedweights" >> "$OUT/cos_wk.log" 2>&1
rc=$?
echo "ZZL1A-EXIT-${rc}-$(date -u +%FT%TZ)ZZ"

#!/usr/bin/env bash
# L2 lane (re-prioritised 2026-09-05T17:25Z): `ccosh` -- ccos WITH THE HOLD
# BRANCH -- at the SAME weights as the banked `ccos_argmax` (0, 0, 64.297) so
# the METRIC is the only variable. Runs as soon as the shipped-weights `cos`
# arm frees its GPU slot. Polls for the ABSENCE of explicit PIDs.
set -u
. "$(dirname "$0")/wk_lib.sh"
wait_pids "25848,29932"
tag=ccosh_w000
echo "ZZARM-${tag}-START-$(date -u +%FT%TZ)ZZ"
"$PY" "$M/taniteval/tools/refav1_arm.py" \
  --ckpt C:/Users/Admin/refav1_eval_slice/ckpt_ep3/ckpt.pt \
  --config C:/Users/Admin/refav1_eval_slice/ckpt_ep3/config.json \
  --cache "$CACHE" --episodes "$EPS" --labels "$LBL" --nav "$LBL" \
  --device cuda --episodes-n 0 --window-stride 16 --no-navshuf \
  --no-lead-block --cost-metric ccosh \
  --cost-weights 0.0,0.0,64.29715042415070 \
  --plan-seed 0 \
  --dump-dir "$OUT/dump_${tag}" --out "$OUT/rec_${tag}.json" \
  --arm "refav1-21109-p4-${tag}" >> "$OUT/${tag}.log" 2>&1
echo "ZZARM-${tag}-EXIT-$?-$(date -u +%FT%TZ)ZZ"
echo "ZZQUEUEB2-DONE-$(date -u +%FT%TZ)ZZ"

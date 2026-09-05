#!/usr/bin/env bash
# RERUN of the two arms that died at startup on my own UnboundLocalError
# (`_inspect2` was bound inside the --goal-kappa-turn branch). Fixed by importing
# `inspect` inside the seed-ladder block; a bare `%` in the --kamm-mu help string
# also broke argparse's help formatting and is now escaped. NO GPU was wasted:
# the tool fails before the rollout, which is what its preflight design is for.
# Runs l3ladder immediately (kamm07 is the only other live arm), then combined.
set -u
M="C:/Users/Admin/tanitad-wt"
OUT="C:/Users/Admin/refav1_margin/p4out"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
export PYTHONPATH="$M/stack;$M/taniteval"
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=6
LBL=C:/Users/Admin/navcomp/data/s2_labels_v7.2_eval.jsonl.gz
CACHE=C:/Users/Admin/refav1_margin/p4/fp8
EPS=C:/Users/Admin/refav1_margin/p4/eps
common=(--ckpt C:/Users/Admin/refav1_eval_slice/ckpt_ep3/ckpt.pt
        --config C:/Users/Admin/refav1_eval_slice/ckpt_ep3/config.json
        --cache "$CACHE" --episodes "$EPS" --labels "$LBL" --nav "$LBL"
        --device cuda --episodes-n 0 --window-stride 16 --no-navshuf
        --no-lead-block --cost-metric ccos
        --cost-weights 0.0,0.0,64.29715042415070 --plan-seed 0)
run () { local tag="$1"; shift
  rm -rf "$OUT/dump_$tag"
  echo "ZZARM-${tag}-START-$(date -u +%FT%TZ)ZZ"
  "$PY" "$M/taniteval/tools/refav1_arm.py" "${common[@]}" "$@" \
    --dump-dir "$OUT/dump_${tag}" --out "$OUT/rec_${tag}.json" \
    --arm "refav1-21109-p4-${tag}" >> "$OUT/${tag}.log" 2>&1
  echo "ZZARM-${tag}-EXIT-$?-$(date -u +%FT%TZ)ZZ"; }
run l3ladder --seed-kappa-ladder 0.002,0.005,0.01,0.02,0.04
run combined --seed-kappa-ladder 0.002,0.005,0.01,0.02,0.04 --kamm-mu 0.7
echo "ZZQUEUEG-DONE-$(date -u +%FT%TZ)ZZ"

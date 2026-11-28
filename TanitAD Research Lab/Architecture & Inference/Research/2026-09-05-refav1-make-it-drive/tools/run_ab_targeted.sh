#!/usr/bin/env bash
# The TARGETED paired A/B — 3 arms, ~0.9 GPU-h, on the windows where the lever
# actually acts.
#
# ⛔ WHY THE OBVIOUS DESIGN WAS ABANDONED. A first launch used the first 30
# episodes (60 windows). MEASURED on the banked logits BEFORE spending the GPU:
# `prior tau=0.5` changes the decode on only **3 of those 60 windows**. A paired
# bootstrap where 95 % of windows are bit-identical ties would have produced an
# interval dominated by 3 windows — underpowered BY CONSTRUCTION, which is
# CLAUDE.md probe-failure #4 (n << d reads as absence, not as absence of power).
# Cost of the honest full-panel version: 282 windows x 40 s x 2 arms = **6.3
# GPU-h**, and 9.4 with the replicate.
#
# ⭐ THE ECONOMY, AND WHY IT IS NOT CHERRY-PICKING. `icem_plan` is SEEDED, so on
# a window where the decode does NOT change the two arms are bit-identical and
# contribute EXACTLY 0 to a paired delta. The aggregate therefore factorises:
#
#     marginal_delta = (18/282) x conditional_delta
#
# so measuring the CONDITIONAL effect on the 14 episodes that carry the 18
# changed windows recovers the aggregate exactly, at 1/5 the cost. The claim
# being measured is stated as conditional ("when the rule changes the decision,
# does driving improve?") and the marginal rate 0.064 is carried beside it.
#
# ⛔ AND THE PREMISE IS NOT ASSUMED — IT IS THE BUILT-IN CONTROL. These 14
# episodes carry 28 windows, of which only 18 change; the other **10 must come
# out BIT-IDENTICAL between arms**. If they do not, the factorisation above is
# void and so is every number from this run. Check them first.
#
# Arms: argmax@seed0 (control) | prior050@seed0 (lever) | argmax@seed1
# (REPLICATE — H-ESTIM-SEED-1: the planner is stochastic, so a separated CI
# against the control is necessary-not-sufficient until read against this).
set -u
M="C:/Users/Admin/tanitad-wt"
OUT="C:/Users/Admin/refav1_drive/abt"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
export PYTHONPATH="$M/stack;$M/taniteval"
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=6
mkdir -p "$OUT"

LBL=C:/Users/Admin/navcomp/data/s2_labels_v7.2_eval.jsonl.gz
CACHE=C:/Users/Admin/refav1_drive/tgt/fp8
EPS=C:/Users/Admin/refav1_drive/tgt/eps

common=(--ckpt C:/Users/Admin/refav1_eval_slice/ckpt_ep3/ckpt.pt
        --config C:/Users/Admin/refav1_eval_slice/ckpt_ep3/config.json
        --cache "$CACHE" --episodes "$EPS" --labels "$LBL" --nav "$LBL"
        --device cuda --episodes-n 0 --window-stride 40 --no-navshuf
        --no-lead-block --cost-metric ccos
        --cost-weights 0.0,0.0,64.29715042415070)

run () { local name="$1"; shift
  echo "=== ARM $name  $(date -u +%FT%TZ) ==="
  "$PY" "$M/taniteval/tools/refav1_arm.py" "${common[@]}" "$@" \
      --dump-dir "$OUT/dump_$name" --out "$OUT/rec_$name.json" \
      --arm "refav1-21109-abt-$name" >> "$OUT/$name.log" 2>&1
  echo "    exit=$? $(date -u +%FT%TZ)"; }

run argmax        --plan-seed 0
run prior050      --plan-seed 0 --lat-logit-bias "0.197442,0,0,0,1.191905,0.976514,1.669661,1.434659"
run argmax_seed1  --plan-seed 1
echo "ABT_DONE $(date -u +%FT%TZ)"

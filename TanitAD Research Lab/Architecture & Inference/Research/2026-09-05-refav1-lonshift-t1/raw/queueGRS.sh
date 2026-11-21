#!/usr/bin/env bash
# P4's LIVE next lever: `goal_reach_s`.
#
# WHY. HANDOFF.md 4.2: every LON token realises only 0.6513x its named dv inside the
# 2 s plan window, because canonical_controls drives v -> v_t with a time constant
# GOAL_REACH_S = 2.0 s over a 2.0 s horizon (1 - e^-1 = 0.632). The token's own
# semantics are never delivered inside the window the planner optimises.
#
# WARNING carried into every report from this queue: GOAL_REACH_S = TACTICAL_S[0], the
# vocabulary's own 2 s tactical band. Overriding it DECOUPLES the goal profile from the
# band the tokens are defined on. These arms MEASURE a candidate; they do not authorise
# a default.
#
# PRIORITY ORDER -- the CONTROL runs FIRST on purpose: if the default path does not
# reproduce T_lonshift bit-for-bit, the patch is broken and every other arm here is
# VOID. Never read a lever before its control.
set -u
R=/home/nvidia/refav1_lon
OUT=$R/out
PY=/home/nvidia/venvs/tanitad-edge/bin/python
export PYTHONPATH=$R/code/stack:$R/code/taniteval
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=6
LBL=/home/nvidia/data/v72/labels/s2_labels_v7.2_eval.jsonl.gz
MAXJ=${MAXJ:-3}

common=(--ckpt $R/ckpt/ckpt.pt
        --config $R/ckpt/config.json
        --cache $R/p4/fp8 --episodes $R/p4/eps --labels "$LBL" --nav "$LBL"
        --device cuda --episodes-n 0 --window-stride 16 --no-navshuf
        --no-lead-block --cost-metric ccos
        --cost-weights 0.0,15.11245,64.29715042415070
        --a-sustain-mode a0_shift)

run () {
  local tag="$1"; shift
  rm -rf "$OUT/dump_$tag"
  echo "ZZARM-${tag}-START-$(date -u +%FT%TZ)ZZ" >> "$OUT/queue.log"
  ( "$PY" "$R/code/taniteval/tools/refav1_arm.py" "${common[@]}" "$@" \
      --dump-dir "$OUT/dump_${tag}" --out "$OUT/rec_${tag}.json" \
      --arm "refav1-21109-thor-${tag}" >> "$OUT/${tag}.log" 2>&1
    echo "ZZARM-${tag}-EXIT-$?-$(date -u +%FT%TZ)ZZ" >> "$OUT/queue.log" ) &
  while [ "$(jobs -p | wc -l)" -ge "$MAXJ" ]; do sleep 20; done
}

# 1 THE CONTROL. --goal-reach-s 2.0 is the module default, so this MUST reproduce
#   T_lonshift bit-for-bit. If it does not, STOP: the patch changed the default path.
run T_grsCTL  --plan-seed 0 --goal-reach-s 2.0
# 2 THE LEVER. A shorter reach makes a token deliver its own dv inside the window.
run T_grs1    --plan-seed 0 --goal-reach-s 1.0
# 3 THE DELIBERATE REGRESSION. A LONGER reach delivers even less inside the window,
#   so it must be WORSE on LON speed. If it is not, the knob is not doing what the
#   mechanism says and no result here is admissible.
run T_grs8    --plan-seed 0 --goal-reach-s 8.0
# 4 THE INFERENCE-SEED REPLICATE of the lever -- mandatory before any claim
#   (H-ESTIM-SEED-1 / D-REFAV1-SEED-LON: --plan-seed alone reads `separated` on all
#   three longitudinal metrics on this rig).
run T_grs1_s1 --plan-seed 1 --goal-reach-s 1.0
wait
echo "ZZQUEUEGRS-DONE-$(date -u +%FT%TZ)ZZ" >> "$OUT/queue.log"

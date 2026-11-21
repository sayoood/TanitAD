#!/bin/bash
# ============================================================================
# D-TACGOAL-2 / PI queue item 10 -- the --w-tac-goal WEIGHT SWEEP.
#
# The question is NOT "what number feels right"; it is "at what weight does
# tac_goal_tok_head actually learn without displacing what already works".
# 11,286 parameters took grad_abs_sum EXACTLY 0.00000 for all 40,284 steps of
# refcv5-v2 because --w-tac-goal defaults to 0.0 and was never passed.
#
# Six arms, 400 steps each, seed 0, one lever moved between them and nothing
# else. Arms are ordered so that a killed sweep still yields the load-bearing
# triple first: control, replicate (the rig's own noise floor), candidate.
#
#   A_w0       --w-tac-goal ABSENT  -- the control. MUST read grad_abs_sum
#                                      EXACTLY 0.0 and n_grad_none 2 of 2.
#   A_w0_rep   --w-tac-goal ABSENT  -- A_w0's flags, A_w0's seed, run again,
#                                      ZERO levers moved. A separated result
#                                      from a one-seed arm is NECESSARY, NOT
#                                      SUFFICIENT; this arm is the floor any
#                                      "the lever moved it" claim is read
#                                      against.
#   C_w0p05    0.05                 -- the value pre-registered for D-TACGOAL-1
#                                      arm A1, never measured.
#   D_w0p5     0.5
#   B_w0p005   0.005
#   E_w5p0     5.0
#
# Corpus: physicalai-b1-w120-256x640cyl + s2_labels_v7.2_train.jsonl.gz -- the
# SAME corpus refcv5-v2 trained on (its argv reads --v2-cache /root/data/train
# with these exact v7.2 labels) and the same one the WP-D arms used on Thor. No
# episode is re-selected; parity is untouched.
#
# Schedule: warmup 20 + cosine to 0 over 400 steps = 5 % warmup, matching
# refcv5-v2's 2000/40284 FRACTION, so each arm runs a complete LR schedule
# rather than sitting in a ramp.
# ============================================================================
set -u
R=/home/nvidia/tacgoal_sweep
STACK=$R/stack
PY=/home/nvidia/venvs/tanitad-train/bin/python
OUTROOT=/home/nvidia/experiments/tacgoal-wsweep
PROG=$OUTROOT/sweep_progress.log

BASE=(
  --arm hier --size base --image-hw 256 640
  --v2-cache /home/nvidia/data/physicalai-b1-w120-256x640cyl
  --v7-labels /home/nvidia/data/v72/labels/s2_labels_v7.2_train.jsonl.gz
  --eval-cache /home/nvidia/data/physicalai-b1-EVAL6-w120-256x640cyl
  --eval-labels /home/nvidia/data/v72/labels/s2_labels_v7.2_eval.jsonl.gz
  --eval-every 400 --eval-batches 8
  --batch 20 --workers 6 --prefetch-factor 1 --v2-lru 24
  --lr 1e-4 --warmup 20 --log-every 20 --save-every 100000 --u8-batches
  --nav-from-v7 --ego-state-inject --ego-dropout 0.5
  --anchors /home/nvidia/data/anchors/refc_anchors_6s_v0cond_alat_117.pt
  --n-anchors 117 --anchor-v0-conditioned --anchor-control-units alat
  --sel-accel-max 2.0
  --sampler ddim --w-u0 0.5 --sel-refined --sel-score-emitted
  --goal-str --tac-goal-tok-head --agents off
  --grad-probe-modules tac_goal_tok_head,core.decoder.offset_head,scorer.goal_point
  --steps 400 --seed 0
)

run_arm() {                     # $1 = tag   $2 = weight ("none" = flag absent)
  local tag="$1"
  local w="$2"
  local out="$OUTROOT/$tag"
  local rc=0
  rm -rf "$out"; mkdir -p "$out"; cd "$out" || return 1
  local extra=()
  if [ "$w" != "none" ]; then extra=(--w-tac-goal "$w"); fi
  echo "ZZSTART $tag w=$w $(date -u +%FT%TZ)" >> "$PROG"
  PYTHONPATH="$STACK" PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=6 \
    "$PY" -u "$STACK/scripts/refc_v3_train.py" "${BASE[@]}" "${extra[@]}" \
    --out "$out" > "$out/train.log" 2> "$out/train.stderr.log"
  rc=$?
  echo "ZZARM $tag w=$w exit=$rc rows=$(wc -l < "$out/metrics.jsonl" 2>/dev/null || echo 0) $(date -u +%FT%TZ)" >> "$PROG"
}

mkdir -p "$OUTROOT"
: > "$PROG"
echo "ZZSWEEP_START $(date -u +%FT%TZ) trainer_md5=$(md5sum $STACK/scripts/refc_v3_train.py | cut -d' ' -f1)" >> "$PROG"
run_arm A_w0      none
run_arm A_w0_rep  none
run_arm C_w0p05   0.05
run_arm D_w0p5    0.5
run_arm B_w0p005  0.005
run_arm E_w5p0    5.0
echo "ZZSWEEP_DONE $(date -u +%FT%TZ)" >> "$PROG"

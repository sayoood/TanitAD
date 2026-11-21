#!/bin/bash
# ============================================================================
# D-TACGOAL-2 -- FOLLOW-UP, three arms, each answering something the six-arm
# sweep left open. All pre-declared in PREREG_TACGOAL_WEIGHT_SWEEP.md.
#
#  1) F_w0p15   w=0.15, 400 steps  -- PRE-REGISTERED LEVER L1 (bisection).
#       MEASURED on the sweep: w=0.05 spends 0.190x the tactical-aux budget
#       and w=0.5 spends 1.708x, so criterion (c)'s ceiling -- w x term equal
#       to the MANEUVER_WEIGHT contribution -- sits at w ~ 0.26-0.29. The rule
#       recommends the LARGEST ADMISSIBLE w, and with only {0.005, 0.05, 0.5,
#       5.0} tested that would be 0.05 purely because nothing between was
#       tried. 0.15 is comfortably inside the bound (~0.51x) and 3x above the
#       pre-registered 0.05.
#
#  2) A_w0_rep2 w absent, seed 0   -- the floor, sampled a SECOND time.
#       The whole panel is read against |A_w0_rep - A_w0|, which is ONE
#       difference from ONE pair. MEASURED: the rig is NOT deterministic even
#       at a fixed seed (the two controls differ at all 20 logged steps,
#       tail-5 |d traj| = 0.084444), so the floor is real and its size matters.
#       A third same-seed control turns 1 pairwise difference into 3.
#
#  3) EVALFIX_smoke w=0.05, 40 steps, --eval-every 40, on the FIXED trainer
#       -- D-TACGOAL-EVAL-1 end to end. Arm C_w0p05 finished all 400 training
#       steps and then EXITED 1 when the step-400 eval hit the
#       REFUSE-DO-NOT-SKIP guard, because e_ds never got tac_goal_targets.
#       40 steps is deliberate: the question is "does an eval row appear
#       instead of a crash", not "what does it say".
#
# TRAINER DISCIPLINE. Arms 1-2 run on the SAME binary as the six-arm sweep
# (md5 3575e1f4ab2851a51d282ff7e4f44d73) so the panel stays one binary. The
# swap to the fixed trainer happens ONCE, between arm 2 and arm 3, and both
# md5s are written into the progress log.
# ============================================================================
set -u
R=/home/nvidia/tacgoal_sweep
STACK=$R/stack
PY=/home/nvidia/venvs/tanitad-train/bin/python
OUTROOT=/home/nvidia/experiments/tacgoal-wsweep
PROG=$OUTROOT/followup_progress.log
SWEEP_TRAINER=/home/nvidia/tacgoal_sweep/trainer_SWEEP.py
FIXED_TRAINER=/home/nvidia/tacgoal_sweep/trainer_FIXED.py

BASE=(
  --arm hier --size base --image-hw 256 640
  --v2-cache /home/nvidia/data/physicalai-b1-w120-256x640cyl
  --v7-labels /home/nvidia/data/v72/labels/s2_labels_v7.2_train.jsonl.gz
  --eval-cache /home/nvidia/data/physicalai-b1-EVAL6-w120-256x640cyl
  --eval-labels /home/nvidia/data/v72/labels/s2_labels_v7.2_eval.jsonl.gz
  --eval-batches 8
  --batch 20 --workers 6 --prefetch-factor 1 --v2-lru 24
  --lr 1e-4 --log-every 20 --save-every 100000 --u8-batches
  --nav-from-v7 --ego-state-inject --ego-dropout 0.5
  --anchors /home/nvidia/data/anchors/refc_anchors_6s_v0cond_alat_117.pt
  --n-anchors 117 --anchor-v0-conditioned --anchor-control-units alat
  --sel-accel-max 2.0
  --sampler ddim --w-u0 0.5 --sel-refined --sel-score-emitted
  --goal-str --tac-goal-tok-head --agents off
  --grad-probe-modules tac_goal_tok_head,core.decoder.offset_head,scorer.goal_point
)

run_arm() {          # $1 tag  $2 weight|none  $3 seed  $4 steps  $5 warmup
                     # $6 eval_every  $7 trainer_path
  local tag="$1"; local w="$2"; local sd="$3"; local st="$4"
  local wu="$5"; local ee="$6"; local tp="$7"
  local out="$OUTROOT/$tag"
  local rc=0
  rm -rf "$out"; mkdir -p "$out"; cd "$out" || return 1
  local extra=()
  if [ "$w" != "none" ]; then extra=(--w-tac-goal "$w"); fi
  echo "ZZSTART $tag w=$w seed=$sd steps=$st trainer_md5=$(md5sum "$tp" | cut -d' ' -f1) $(date -u +%FT%TZ)" >> "$PROG"
  cp "$tp" "$STACK/scripts/refc_v3_train.py"
  PYTHONPATH="$STACK" PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=6 \
    "$PY" -u "$STACK/scripts/refc_v3_train.py" "${BASE[@]}" "${extra[@]}" \
    --steps "$st" --warmup "$wu" --eval-every "$ee" --seed "$sd" \
    --out "$out" > "$out/train.log" 2> "$out/train.stderr.log"
  rc=$?
  echo "ZZARM $tag w=$w exit=$rc rows=$(wc -l < "$out/metrics.jsonl" 2>/dev/null || echo 0) evalrows=$(grep -c 'eval_' "$out/metrics.jsonl" 2>/dev/null || echo 0) $(date -u +%FT%TZ)" >> "$PROG"
}

mkdir -p "$OUTROOT"

# ⛔⛔ REFUSE, DO NOT WAIT-AND-HOPE. This script OVERWRITES
# $STACK/scripts/refc_v3_train.py -- the exact file the six-arm sweep launches
# every arm from. Starting it while that sweep is live would swap the binary
# under a running panel. Two independent conditions, both required, and the
# check is on ARTIFACTS (a marker line, a process table) rather than on a
# belief about the clock.
if ! grep -q ZZSWEEP_DONE "$OUTROOT/sweep_progress.log" 2>/dev/null; then
  echo "REFUSING: sweep_progress.log has no ZZSWEEP_DONE -- the six-arm sweep is not finished" >&2
  exit 3
fi
# the pattern is assembled at runtime so this command's own argv cannot match it
_PAT="$(echo refc_v3)$(echo _train)"
if ps -eo args | grep -v " grep " | grep -q -- "$_PAT"; then
  echo "REFUSING: a $_PAT process is still alive" >&2
  ps -eo pid,args | grep -v " grep " | grep -- "$_PAT" >&2
  exit 4
fi
: > "$PROG"
echo "ZZFOLLOWUP_START $(date -u +%FT%TZ) sweep=$(md5sum $SWEEP_TRAINER | cut -d' ' -f1) fixed=$(md5sum $FIXED_TRAINER | cut -d' ' -f1)" >> "$PROG"
run_arm F_w0p15       0.15 0 400 20 400 "$SWEEP_TRAINER"
run_arm A_w0_rep2     none 0 400 20 400 "$SWEEP_TRAINER"
run_arm EVALFIX_smoke 0.05 0  40  5  40 "$FIXED_TRAINER"
echo "ZZFOLLOWUP_DONE $(date -u +%FT%TZ)" >> "$PROG"

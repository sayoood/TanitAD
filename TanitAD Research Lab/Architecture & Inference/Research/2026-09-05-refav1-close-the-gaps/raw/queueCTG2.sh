#!/usr/bin/env bash
# refav1 A4 -- THE ARM THAT FOLLOWS FROM THE ATTRIBUTION, and it is the one the
# PI's question actually needs: an arm that drives accurately AND turns.
#
# ⛔⛔ WHY A3 ALONE CANNOT BE THE ANSWER -- MEASURED, 0 GPU, paired window-by-window
# on the shared grid with the ha/ha0/ha0_ext/ol/g grid control PASSED
# (`raw/attribute.txt`):
#
#   goal token   n   ADE ccos_argmax -> wk15 -> wk151      ha0 (straight) floor
#   TURN_L       9       1.5043   ->  1.2429  ->  1.1959          1.4630
#
#   MORE PENALTY = LESS TURNING = BETTER ADE ON TURN WINDOWS. And the STRAIGHT-LINE
#   floor (1.4630) BEATS the faithfully-turning planner (1.5043).
#   ⇒ THE GOAL'S COMMANDED CURVATURE OF 0.08 (R = 12.5 m) IS WORSE THAN DRIVING
#     STRAIGHT ON THIS CORPUS. The corpus curves at R 100-1000 m, so the command is
#     ~6.9x too tight and executing it faithfully drives you off the road.
#     (Independent reproduction of D-REFAV1-GOAL-MARGIN by a completely different
#     route -- that one measured the vocabulary, this one measures the ADE it buys.)
#
# ⇒ A3 restores the COMMANDED curvature, but the COMMAND is wrong. A3 alone will
#   recover turn recall and WORSEN ADE on turn windows. That is not A3 failing; it
#   is A3 correctly obeying a wrong goal. A3 is NECESSARY AND NOT SUFFICIENT.
#
# ⭐ THE FIX IS THE PAIR: goal-conditioned cost + a CORRECTED turn magnitude.
#   `--goal-kappa-turn 0.02` is a CONSTANT, not an oracle chooser, so the arm stays
#   T1 and admissible (the LEVEL-SET form needs `goal_kappa_hint`, which the v7.0
#   head cannot supply -- that route is an ORACLE bound and is deliberately NOT
#   taken here). Banked measurement: 100 % of real turns expressible at 0.02
#   against 38.7 % at 0.08, median curvature error on a turn down 2.5x.
#
# ⛔ ATTRIBUTION: A4a and A4b are ONE VARIABLE APART, and each is one variable from
#   an arm that already exists, so the two levers can be separated:
#     T_wk15        -> A4b   : kappa_turn alone      (0.08 -> 0.02)
#     G_gkappa      -> A4a   : kappa_turn alone      (0.08 -> 0.02)
#     A4b           -> A4a   : goal-conditioned cost alone
set -u
R=/home/nvidia/refav1_ctg
LON=/home/nvidia/refav1_lon
OUT=$R/out
PY=/home/nvidia/venvs/tanitad-edge/bin/python
export PYTHONPATH=$R/code/stack:$R/code/taniteval
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=6
LBL=/home/nvidia/data/v72/labels/s2_labels_v7.2_eval.jsonl.gz
MAXARMS=${MAXARMS:-3}

common=(--ckpt $LON/ckpt/ckpt.pt
        --config $LON/ckpt/config.json
        --cache $LON/p4/fp8 --episodes $LON/p4/eps --labels "$LBL" --nav "$LBL"
        --device cuda --episodes-n 0 --window-stride 16 --no-navshuf
        --no-lead-block --cost-metric ccos
        --cost-weights 0.0,15.11245,64.29715042415070)

# ⛔ Count ARMS (a --dump-dir), never processes: one arm is several python entries
# and a gate on processes can never open. Counts EVERY tree's arms, so this queue
# cannot oversubscribe the box against the sibling stream.
live_arms () { ps -eo args | grep "[r]efav1_arm.py" | grep -o -- "--dump-dir [^ ]*" | wc -l; }

wait_slot () {
  while [ "$(live_arms)" -ge "$MAXARMS" ]; do sleep 30; done
}

run () {
  local tag="$1"; shift
  wait_slot
  rm -rf "$OUT/dump_$tag"
  echo "ZZARM-${tag}-START-$(date -u +%FT%TZ)ZZ" >> "$OUT/queue.log"
  ( "$PY" "$R/code/taniteval/tools/refav1_arm.py" "${common[@]}" "$@" \
      --dump-dir "$OUT/dump_${tag}" --out "$OUT/rec_${tag}.json" \
      --arm "refav1-21109-thorctg-${tag}" >> "$OUT/${tag}.log" 2>&1
    echo "ZZARM-${tag}-EXIT-$?-$(date -u +%FT%TZ)ZZ" >> "$OUT/queue.log" ) &
  sleep 45   # let the new arm appear in the process table before the next gate check
}

# PRIORITY ORDER -- a killed queue still yields the arm that answers the PI.
# 1 THE ARM: goal-conditioned cost AND a turn magnitude the corpus can use.
run A4a_gk_kt02  --plan-seed 0 --w-kappa-by-goal 15.11245,0.0 --goal-kappa-turn 0.02
# 2 THE ATTRIBUTION ARM: the corrected magnitude ALONE, on the shipped scalar cost.
#   Without it, A4a's result cannot be split between its two levers.
run A4b_kt02     --plan-seed 0 --goal-kappa-turn 0.02
wait
echo "ZZQUEUECTG2-DONE-$(date -u +%FT%TZ)ZZ" >> "$OUT/queue.log"

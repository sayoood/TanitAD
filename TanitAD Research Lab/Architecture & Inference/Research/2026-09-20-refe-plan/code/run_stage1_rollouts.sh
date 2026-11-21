#!/bin/bash
# Stage 1 ROLLOUT-LEVEL control + first augmented arm, nuPlan mini, non-reactive, no TTS.
#   arm A  rank 0 -> the patch is NOT installed -> MUST reproduce Stage 0 EXACTLY (97.192507)
#   arm B  rank 1 -> the route starts in the neighbouring lane -> a genuinely different rollout
# The route-level control already passed exactly (8/8, max diff 0.000e+00); this is the same control
# one level up, at the ROLLOUT, which is what actually produces REFe's training signal.
set -u
PKG="D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
DB="D:/Projects/TanitAD/data/nuplan/dblinks/driverl_val14"
for RANK in 0 1; do
  echo "$(date +%T) === Stage-1 arm: lane rank $RANK ==="
  if DRIVERL_EVAL_ROUTE_LANE_RANK=$RANK bash "$PKG/code/run_mini_teacher.sh" "$DB" nr 8 0 \
       > "$PKG/raw/stage1_rank${RANK}.log" 2>&1; then
    grep -E "^RESULT " "$PKG/raw/stage1_rank${RANK}.log" | cut -c1-170
    echo "STAGE1_RANK_DONE $RANK"
  else
    tail -4 "$PKG/raw/stage1_rank${RANK}.log" | cut -c1-170
    echo "STAGE1_RANK_FAIL $RANK"
  fi
done
echo "$(date +%T) STAGE1_ROLLOUTS_DONE"

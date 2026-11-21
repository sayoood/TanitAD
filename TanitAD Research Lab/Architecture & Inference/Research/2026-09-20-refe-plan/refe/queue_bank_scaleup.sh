#!/bin/bash
# Scale the REFe target bank from 8 scenarios toward the whole mini split (POD_HANDOFF arm B).
#
# Stage 0 ran `one_of_each_scenario_type` with limit 8. Raising the limit admits one scenario per
# TYPE for every type present in mini (~60-70 types exist in nuPlan), so this is a genuine widening
# of scenario COVERAGE rather than more samples of the same eight.
#
# Cost, from the MEASURED Stage-0 rate (15m12s for 8 scenarios = ~114 s/scenario):
#   64 scenarios x 2 route ranks ~ 4.0 h on this 4060. Runs unattended.
#
# Waits for the GPU: the image-isolation arms must finish first, or both sets of timings are junk.
set -u
PKG="D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
DB="D:/Projects/TanitAD/data/nuplan/dblinks/driverl_val14"
W="C:/Users/Admin/dz/out/image_isolation.log"
LIMIT="${1:-64}"
for _ in $(seq 1 120); do
  grep -q "IMAGE_ISOLATION_DONE" "$W" 2>/dev/null && break
  sleep 30
done
echo "$(date +%T) GPU free -- widening the bank to limit=$LIMIT, ranks 0 and 1"
for RANK in 0 1; do
  echo "$(date +%T) === teacher rollouts, lane rank $RANK, limit $LIMIT ==="
  if DRIVERL_EVAL_ROUTE_LANE_RANK=$RANK bash "$PKG/code/run_mini_teacher.sh" "$DB" nr "$LIMIT" 0 \
       > "$PKG/raw/bank_scaleup_rank${RANK}.log" 2>&1; then
    grep -E "^RESULT " "$PKG/raw/bank_scaleup_rank${RANK}.log" | cut -c1-160
    echo "BANK_ROLLOUT_DONE $RANK"
  else
    tail -4 "$PKG/raw/bank_scaleup_rank${RANK}.log" | cut -c1-160
    echo "BANK_ROLLOUT_FAIL $RANK"
  fi
done
echo "$(date +%T) BANK_SCALEUP_DONE"

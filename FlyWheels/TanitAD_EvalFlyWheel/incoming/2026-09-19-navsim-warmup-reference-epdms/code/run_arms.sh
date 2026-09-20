#!/usr/bin/env bash
# Run scoring arms SEQUENTIALLY (RAM headroom), bank each devkit CSV into raw/<arm>/devkit_<ts>.csv.
set -u
PKG="/d/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-warmup-reference-epdms"
EXP="/c/Users/Admin/navsim/exp/e1"
declare -A OUT=([A1]=A1_cv_two_stage [R1]=R1_cv_two_stage_replicate [A2]=A2_human_two_stage
                [A1b]=A1b_cv_one_stage [A2b]=A2b_human_one_stage [M1]=M1_human_one_stage_filter_off)
for arm in "$@"; do
  mkdir -p "$PKG/raw/$arm"
  echo "== $arm start $(date +%H:%M:%S)"
  bash "$PKG/code/run_warmup_reference.sh" "$arm" > "$PKG/raw/$arm/$arm.log" 2>&1
  for f in "$EXP/${OUT[$arm]}"/*.csv; do [ -f "$f" ] && cp "$f" "$PKG/raw/$arm/devkit_$(basename "$f")"; done
  cp "$EXP/${OUT[$arm]}/run_pdm_score"*.log "$PKG/raw/$arm/" 2>/dev/null
  echo "== $arm end $(date +%H:%M:%S) :: $(grep -a 'E1_WRAPPER_DONE\|E1_RAM_GUARD_ABORT' "$PKG/raw/$arm/$arm.log" | tail -1)"
done

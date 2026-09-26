#!/bin/bash
# DZ-10 — value-guided test-time search N-sweep on the released DriveRL teacher, nuPlan mini, SAME 8 scenarios.
# Runs after first light (tts=0) succeeded. Scenario selection is deterministic (filter override +
# limit + shuffle=false), so every N sees identical tokens; tts_sweep_analysis.py refuses otherwise.
# Usage: bash run_tts_sweep.sh [nr|r] [limit]      (default nr 8)   Markers: SWEEP_N_DONE <N> / SWEEP_N_FAIL <N> / SWEEP_DONE
set -u
PROTO="${1:-nr}"; LIMIT="${2:-8}"
PKG="D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
DB="D:/Projects/TanitAD/data/nuplan/dblinks/driverl_val14"
for N in 8 16 32 64; do
  echo "$(date +%T) === TTS N=$N ($PROTO, limit $LIMIT) ==="
  if bash "$PKG/code/run_mini_teacher.sh" "$DB" "$PROTO" "$LIMIT" "$N" > "$PKG/raw/stage0_tts${N}_${PROTO}.log" 2>&1; then
    grep -E "^RESULT " "$PKG/raw/stage0_tts${N}_${PROTO}.log" | cut -c1-200
    echo "SWEEP_N_DONE $N"
  else
    tail -5 "$PKG/raw/stage0_tts${N}_${PROTO}.log" | cut -c1-200
    echo "SWEEP_N_FAIL $N"
  fi
done
echo "$(date +%T) SWEEP_DONE"

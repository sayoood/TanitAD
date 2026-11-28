#!/bin/bash
# Supervisor for the WP-C oracle gate runner.
#
# It supervises the RUNNER, not a single trainer: the runner is resumable at arm
# granularity, so a restart re-enters at the first arm without a done-marker.
#
# LOCK-FD DISCIPLINE (2026-09-02, twice):
#   * every child gets 200>&- -- the runner AND the sleeps. A `sleep 180` that
#     inherited fd 200 once outlived its parent and made a live run permanently
#     unsupervisable, and patching only the trainer was NOT enough.
#   * the flock lives on a per-run path so a stale holder never blocks a fresh
#     supervisor forever.
set -u

LOCK=/workspace/wpc_gate/.sup_wpcgate.lock
ROOT=/workspace/experiments/wpc-oracle-gate
RUNNER=/workspace/wpc_gate/run_gate.sh
LOG=/workspace/wpc_gate/sup_wpcgate.out

mkdir -p /workspace/wpc_gate "$ROOT"
exec 200>"$LOCK"
if ! flock -n 200; then
  echo "$(date -u +%FT%TZ) another supervisor holds $LOCK -- exiting" >> "$LOG"
  exit 0
fi

n=0
while true; do
  if [ -f "$ROOT/summary.json" ] && grep -q '"done": true' "$ROOT/summary.json" 2>/dev/null; then
    echo "$(date -u +%FT%TZ) gate done-marker present -- supervisor exiting cleanly" >> "$LOG"
    exit 0
  fi
  n=$((n + 1))
  echo "$(date -u +%FT%TZ) launch #$n of $RUNNER" >> "$LOG"
  bash "$RUNNER" >> /workspace/wpc_gate/runner.log 2>&1 200>&-
  rc=$?
  echo "$(date -u +%FT%TZ) runner exited rc=$rc" >> "$LOG"
  if [ -f "$ROOT/summary.json" ] && grep -q '"done": true' "$ROOT/summary.json" 2>/dev/null; then
    echo "$(date -u +%FT%TZ) all arms done -- supervisor exiting cleanly" >> "$LOG"
    exit 0
  fi
  if [ "$n" -ge 12 ]; then
    echo "$(date -u +%FT%TZ) 12 relaunches without completion -- giving up, needs a human" >> "$LOG"
    exit 1
  fi
  sleep 60 200>&-
done

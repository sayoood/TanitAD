#!/usr/bin/env bash
# Waits for retry_scoring.sh to declare a successful run, then runs the (content-gated) finisher on
# it. ⛔ Asserts on the ARTIFACT — SUCCESSFUL_RUN_DIR.txt is written only after summary.json exists.
set -u
PKG="D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-20-navhard-refcv4b"
MARK="$PKG/raw/SUCCESSFUL_RUN_DIR.txt"
LOG="$PKG/raw/arm_finisher.log"
T0=$(date +%s)
while [ ! -s "$MARK" ]; do
  if [ $(( $(date +%s) - T0 )) -gt $((20 * 3600)) ]; then
    echo "$(date '+%F %T') giving up: no successful run after 20 h" >> "$LOG"; exit 3
  fi
  sleep 120
done
RD=$(head -1 "$MARK")
echo "$(date '+%F %T') arming the finisher on $RD" >> "$LOG"
bash "$PKG/code/finish_when_done.sh" "$RD" >> "$LOG" 2>&1
echo "$(date '+%F %T') finisher exit recorded (see raw/finisher.log)" >> "$LOG"

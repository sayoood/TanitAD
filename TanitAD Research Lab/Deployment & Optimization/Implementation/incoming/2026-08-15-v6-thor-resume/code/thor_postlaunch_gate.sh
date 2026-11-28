#!/bin/bash
# Post-launch integrity gate for the v6F resume.
#
# The chain launches as soon as the pull reports ALL DONE, and its own check is
# "load one mid-episode per split". That cannot see a SHORT shard — the exact gap
# that understated one defect 14x and another 356x this week. This runs the
# byte-level comparison against the far side AFTER the launch and, only on a real
# size mismatch or a load failure, stops the trainer by EXPLICIT PID before it can
# spend days learning from corrupt data.
#
# Deliberately conservative: it refuses to kill on anything it cannot attribute
# (an unreachable far side is UNKNOWN, never a failure). A false kill costs a
# restart; a missed truncation costs 4.8 days.
set -u
L=~/logs/postlaunch_gate.log
say() { echo "[$(date -u +%FT%TZ)] $*" >> "$L"; }

say "gate armed; waiting for the pull to report ALL DONE"
for _ in $(seq 1 480); do                      # up to ~4 h
  grep -qa "ALL DONE" ~/logs/v6_pull.log 2>/dev/null && break
  sleep 30
done
if ! grep -qa "ALL DONE" ~/logs/v6_pull.log 2>/dev/null; then
  say "TIMEOUT waiting for ALL DONE — not verifying, not killing"
  exit 1
fi

say "pull complete; running the byte-level verification"
~/venvs/tanitad-train/bin/python -u ~/thor_verify_caches.py >> "$L" 2>&1
RC=$?
say "verification rc=$RC"

if [ "$RC" -eq 0 ]; then
  say "PASS — caches agree with the far side by bytes; leaving the run alone"
  exit 0
fi

# Only a genuine SHORT/missing shard justifies stopping a launched run.
if grep -qE "SIZE-MISMATCH [1-9]|LOAD FAILED|DIRECTORY MISSING" "$L"; then
  PID=$(ps -eo pid,args --no-headers | awk '/train_v6_staged\.py/ && !/awk/ {print $1; exit}')
  if [ -n "${PID:-}" ]; then
    say "REAL CORRUPTION FOUND — stopping trainer by explicit PID $PID"
    kill "$PID"
  else
    say "corruption found but no trainer running — nothing to stop"
  fi
else
  say "rc!=0 but no attributable corruption (likely an unreachable far side) — NOT killing"
fi

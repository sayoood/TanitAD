#!/usr/bin/env bash
# pod_gapmargin_after.sh -- card ADD117 (GAPMARGIN10K): waits on the pod for the done marker of the queue of card ADD115 and then starts the queue of card ADD117.
# A stop marker of card ADD115 starts nothing (exit 5, GAPMARGIN_AFTER_NOT_STARTED.txt). A wait above GM_WAIT_S seconds starts nothing (exit 6). Nothing is removed or moved.
# The gate of the queue itself (lead_tests/lead_pytest_queue.out) stays in force: without it the queue exits 4.
set -u
PREV=${GM_PREV:-/workspace/v8proof/presattrc}
GE=${GM_ROOT:-/workspace/v8proof/gapmargin}
KQ=${GM_KIT:-$GE/kit_queue}
WAIT_S=${GM_WAIT_S:-12600}
STEP_S=${GM_STEP_S:-20}
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
[ -f "$KQ/pod_gapmargin10k_queue.sh" ] || { echo "REFUSED: no queue script in $KQ"; exit 2; }
( set -C; echo "$(date +%s) $(ts) pid $$ waiting for $PREV/PRESATTRC10K_DONE.txt" > "$GE/GAPMARGIN_AFTER_START.txt" ) 2>/dev/null || { echo "REFUSED: $GE/GAPMARGIN_AFTER_START.txt exists"; exit 2; }
t0=$(date +%s)
while :; do
  if [ -f "$PREV/PRESATTRC10K_STOP.txt" ]; then
    echo "$(ts) the queue of card ADD115 wrote a stop marker; nothing started" > "$GE/GAPMARGIN_AFTER_NOT_STARTED.txt"; exit 5
  fi
  if [ -f "$PREV/PRESATTRC10K_DONE.txt" ]; then break; fi
  if [ $(( $(date +%s) - t0 )) -gt "$WAIT_S" ]; then
    echo "$(ts) waited more than $WAIT_S s; nothing started" > "$GE/GAPMARGIN_AFTER_NOT_STARTED.txt"; exit 6
  fi
  sleep "$STEP_S"
done
echo "$(date +%s) $(ts) done marker of card ADD115 seen; the queue starts" >> "$GE/GAPMARGIN_AFTER_START.txt"
exec bash "$KQ/pod_gapmargin10k_queue.sh"

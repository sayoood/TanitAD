#!/usr/bin/env bash
# pod_preshead_after.sh -- card ADD118 (PRESHEAD-T0): waits on the pod for the done marker of the queue of card ADD117 and then starts the fit queue of card ADD118.
# A stop marker of card ADD117 starts nothing (exit 5, PRESHEAD_AFTER_NOT_STARTED.txt). A wait above PH_WAIT_S seconds starts nothing (exit 6). Nothing is removed or moved.
# The gate of the queue itself (lead_tests/lead_pytest_fit.out) stays in force: without it the queue exits 4.
set -u
PREV=${PH_PREV:-/workspace/v8proof/gapmargin}
GP=${PH_ROOT:-/workspace/v8proof/preshead}
K=${PH_KIT:-$GP/kit_fit}
WAIT_S=${PH_WAIT_S:-12600}
STEP_S=${PH_STEP_S:-20}
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
[ -f "$K/pod_preshead_fit_queue.sh" ] || { echo "REFUSED: no queue script in $K"; exit 2; }
( set -C; echo "$(date +%s) $(ts) pid $$ waiting for $PREV/GAPMARGIN10K_DONE.txt" > "$GP/PRESHEAD_AFTER_START.txt" ) 2>/dev/null || { echo "REFUSED: $GP/PRESHEAD_AFTER_START.txt exists"; exit 2; }
t0=$(date +%s)
while :; do
  if [ -f "$PREV/GAPMARGIN10K_STOP.txt" ]; then
    echo "$(ts) the queue of card ADD117 wrote a stop marker; nothing started" > "$GP/PRESHEAD_AFTER_NOT_STARTED.txt"; exit 5
  fi
  if [ -f "$PREV/GAPMARGIN10K_DONE.txt" ]; then break; fi
  if [ $(( $(date +%s) - t0 )) -gt "$WAIT_S" ]; then
    echo "$(ts) waited more than $WAIT_S s; nothing started" > "$GP/PRESHEAD_AFTER_NOT_STARTED.txt"; exit 6
  fi
  sleep "$STEP_S"
done
echo "$(date +%s) $(ts) done marker of card ADD117 seen; the queue starts" >> "$GP/PRESHEAD_AFTER_START.txt"
exec bash "$K/pod_preshead_fit_queue.sh"

#!/bin/sh
# wait_navfeed.sh -- the background WAITER of the nav-feed experiment.  Start ONCE, detached:
#     nohup sh <hold>/code/wait_navfeed.sh < /dev/null > C:/Users/Admin/refcv8_navfeed_out/waiter.out 2>&1 &
#   NEW file (2026-10-20, EvalFlyWheel).  It polls cheap conditions every NF_POLL_S (60) seconds and, only when ALL hold, runs run_navfeed.sh in the foreground:
#     (i) the GPU lock file is absent (the wrapper then takes it through gpu_lock.py; a foreign lock is never broken), (ii) >= 3 GB free on the output volume
#     + a real 16 MB write test, (iii) the final (step 48,852) chain is not active (navfeed_common.sh: conditions_ok), (iv) >= 6 GB free commit (the wrapper's gate).
#   The wrapper re-checks (ii)/(iii) after it holds the lock.  Wrapper exit 10/11/12 = not now (YIELD / no lock or commit / a condition flipped): keep polling.
#   Wrapper exit 20/21 = a roll or the smoke FAILED, or the seal is bad: the waiter HALTS (a failed GPU run is never retried unattended).  Exit 0 = done.
#   Single instance (pid file + kill -0).  Kill ONLY by the explicit pid in $OUT/state/waiter.pid.  Markers: opaque ZZNF...ZZ lines in $OUT/navfeed.log.
HOLD_CODE="$(cd "$(dirname "$0")" && pwd)"
. "$HOLD_CODE/navfeed_common.sh"
POLL="${NF_POLL_S:-60}"
LOCKF="${NF_LOCKF:-/c/Users/Admin/qland/work/refcv7/devbox_gpu.lock}"
PIDF="$OUTL/state/waiter.pid"
if [ -s "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then mark DUP "another waiter (pid $(cat "$PIDF")) is alive; exiting"; exit 5; fi
echo $$ > "$PIDF"
cat "/proc/$$/winpid" > "$OUTL/state/waiter.winpid" 2>/dev/null
trap '[ "$(cat "$PIDF" 2>/dev/null)" = "$$" ] && rm -f "$PIDF"' EXIT
trap 'exit 143' TERM INT HUP
mark WAITERSTART "waiter pid $$ (windows pid $(cat "$OUTL/state/waiter.winpid" 2>/dev/null)); polling every ${POLL}s"
n=0
while :; do
  n=$((n + 1))
  # all six rolls banked AND the analysis written -> nothing left to do
  if jq_ok "$OUTL/NAV_FEED.json" "j['committed_label']['label'] in ('FEED-FIX-MATTERS','FEED-FIX-NULL','MIXED','NOT-EVALUABLE')"; then mark WAITERDONE "NAV_FEED.json already present"; exit 0; fi
  why=""
  [ -e "$LOCKF" ] && why="${why}lock-held "
  _fm=$(free_mb); [ "$_fm" -ge "$MIN_FREE_MB" ] 2>/dev/null || why="${why}disk(${_fm}MB) "
  chain_marker_present && why="${why}chain-marker "
  bfinal_pending && why="${why}final-chain-pending "
  if [ -z "$why" ]; then
    mark WAITERGO "poll $n: cheap conditions hold; running the wrapper"
    sh "$HOLD_CODE/run_navfeed.sh" < /dev/null >> "$OUTL/logs/wrapper.out" 2>&1
    # the wrapper's verdict is read from ITS ARTIFACTS (the log markers / NAV_FEED.json), the exit code only routes the loop
    RC=$?
    mark WAITERBACK "wrapper returned code $RC"
    case "$RC" in
      0) mark WAITERDONE "wrapper finished"; exit 0 ;;
      10|11|12) : ;;
      *) mark WAITERHALT "wrapper code $RC: a GPU run failed or the seal is bad; NOT retrying unattended (read logs/ and the markers)"; exit "$RC" ;;
    esac
  elif [ $((n % 30)) -eq 1 ]; then
    mark WAITING "poll $n: not now: $why"
  fi
  sleep "$POLL"
done

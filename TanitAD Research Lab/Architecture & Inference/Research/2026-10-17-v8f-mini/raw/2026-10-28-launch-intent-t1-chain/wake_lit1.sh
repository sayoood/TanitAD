#!/usr/bin/env bash
# Waker for lit1_chain.sh: SCRIPT polling (60 s sleep), not model turns.  Returns when the chain's own log shows an ANCHORED end marker, when the chain died without one, or after LIT1_WAKE_MAX_POLLS polls.
#   bash wake_lit1.sh           wait.  exit 0 = ZZLIT1-DONE (the reader rc is in the line) | 4 = ZZLIT1-STOP | 6 = the chain's process is gone and no marker was written | 5 = timeout
#   bash wake_lit1.sh --status  once: the chain's last lines, the box-2 log's per-run state, the verdict (exit codes as above, 1 = still running)
# The markers are matched at LINE START (^ZZLIT1-...), so the chain's own prose lines (timestamp-prefixed) and this script never match themselves; chain.out is rotated at every chain start, so a stale marker of an
# earlier attempt (chain.out.prev_*) is never seen.  Environment: LIT1_LOG (default <this dir>/logs)  LIT1_B2_OUT  LIT1_ART  LIT1_WAKE_POLL_S (60)  LIT1_WAKE_MAX_POLLS (1500).
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
L=${LIT1_LOG:-$H/logs}; O=$L/chain.out
B2=${LIT1_B2_OUT:-C:/Users/Admin/box2_ship/logs/chain_v8f_launchintent.out}
ART=${LIT1_ART:-E:/Projects/TanitAD-artifacts/v8f-launch-intent}
PYX=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
verdict() {   # -> prints, returns the exit code (1 = still running)
  if grep -aq '^ZZLIT1-STOP' "$O" 2>/dev/null; then echo "CHAIN STOP: $(grep -a '^ZZLIT1-STOP' "$O" | tail -1)"; return 4; fi
  if grep -aq '^ZZLIT1-DONE' "$O" 2>/dev/null; then echo "CHAIN DONE: $(grep -a '^ZZLIT1-DONE' "$O" | tail -1)"; return 0; fi
  local pid; pid=$(cat "$L/chain.winpid" 2>/dev/null)
  if [ -n "$pid" ] && ! tasklist //FI "PID eq $pid" //NH 2>/dev/null | grep -q " $pid "; then echo "CHAIN DEAD without an end marker (winpid $pid is gone): rerun the same launcher to resume"; return 6; fi
  return 1
}
if [ "${1:-}" = "--status" ]; then
  tail -6 "$O" 2>/dev/null | cut -c1-260
  echo "-- box-2 log, per run:"; "$PYX" -I "$(cygpath -m "$H")/lit1_lib.py" boxlog --log "$B2" --art "$ART" table < /dev/null
  verdict; rc=$?; [ $rc -eq 1 ] && echo "CHAIN RUNNING (or not started)"; exit $rc
fi
for i in $(seq 1 "${LIT1_WAKE_MAX_POLLS:-1500}"); do
  verdict && exit 0
  rc=$?; [ $rc -ne 1 ] && { tail -8 "$O" 2>/dev/null | cut -c1-260; exit $rc; }
  sleep "${LIT1_WAKE_POLL_S:-60}"
done
echo "WAKER TIMEOUT"; tail -5 "$O" 2>/dev/null | cut -c1-260; exit 5

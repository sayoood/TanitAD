#!/usr/bin/env bash
# Waker for dagorct1_chain.sh: SCRIPT polling (60 s sleep), not model turns.  Returns when the chain's own log shows an ANCHORED end marker, when the chain died without one, or after DORT1_WAKE_MAX_POLLS polls.
#   bash wake_dagorct1.sh           wait.  exit 0 = ZZDAGORCT1-DONE (reader rc in the line) | 4 = ZZDAGORCT1-STOP | 6 = the chain's process is gone and no marker was written | 5 = timeout
#   bash wake_dagorct1.sh --status  once: the chain's last lines, the box-2 log's per-run state, the verdict (exit codes as above, 1 = still running)
# The markers are matched at LINE START (^ZZ...), so the chain's own prose lines (timestamp-prefixed) and this script never match themselves; chain.out is rotated at every chain start, so a stale marker of an earlier
# attempt (chain.out.prev_*) is never seen.
L=${DORT1_LOG:-/e/Projects/TanitAD-artifacts/dagorc_t1/logs}; O=$L/chain.out
B2=${DORT1_B2_OUT:-C:/Users/Admin/box2_ship/logs/chain_v8f_dagoaccorc_r2.out}
ART=${DORT1_ART:-/e/Projects/TanitAD-artifacts/v8f-dag-orc-oacc}
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
verdict() {   # -> prints, returns the exit code (1 = still running)
  if grep -aq '^ZZDAGORCT1-STOP' "$O" 2>/dev/null; then echo "CHAIN STOP: $(grep -a '^ZZDAGORCT1-STOP' "$O" | tail -1)"; return 4; fi
  if grep -aq '^ZZDAGORCT1-DONE' "$O" 2>/dev/null; then echo "CHAIN DONE: $(grep -a '^ZZDAGORCT1-DONE' "$O" | tail -1)"; return 0; fi
  local pid; pid=$(cat "$L/chain.winpid" 2>/dev/null)
  if [ -n "$pid" ] && ! tasklist //FI "PID eq $pid" //NH 2>/dev/null | grep -q " $pid "; then echo "CHAIN DEAD without an end marker (winpid $pid is gone): rerun the same launcher to resume"; return 6; fi
  return 1
}
if [ "${1:-}" = "--status" ]; then
  tail -6 "$O" 2>/dev/null | cut -c1-260
  echo "-- box-2 log, per run:"; C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I "$(cygpath -m "$H")/dagorct1_lib.py" boxlog --log "$B2" --art "$ART" table < /dev/null
  verdict; rc=$?; [ $rc -eq 1 ] && echo "CHAIN RUNNING (or not started)"; exit $rc
fi
for i in $(seq 1 "${DORT1_WAKE_MAX_POLLS:-1500}"); do
  verdict && exit 0
  rc=$?; [ $rc -ne 1 ] && { tail -8 "$O" 2>/dev/null | cut -c1-260; exit $rc; }
  sleep 60
done
echo "WAKER TIMEOUT"; tail -5 "$O" 2>/dev/null | cut -c1-260; exit 5

#!/usr/bin/env bash
# Waker for soup_chain.sh: SCRIPT polling (60 s sleep), not model turns.  Returns when the chain's own log shows an ANCHORED end marker, when the chain died without one, or after SOUP_WAKE_MAX_POLLS polls.
#   bash wake_soup.sh           wait.  exit 0 = ZZSOUP-DONE (the reader rc is in the line) | 4 = ZZSOUP-STOP | 6 = the chain's process is gone and no marker was written | 5 = timeout
#   bash wake_soup.sh --status  once: the chain's last lines and the verdict (exit codes as above, 1 = still running)
# The markers are matched at LINE START (^ZZSOUP-...), so the chain's own prose lines (timestamp-prefixed) and this script never match themselves; chain.out is rotated at every chain start, so a stale marker of an
# earlier attempt (chain.out.prev_*) is never seen.  The launcher's own log (soup_chain.launch.out) is read too: a refusal printed before the chain log existed still ends the wait.
# Environment: SOUP_LOG (default <this dir>/logs)  SOUP_WAKE_POLL_S (60)  SOUP_WAKE_MAX_POLLS (1500)
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
L=${SOUP_LOG:-$H/logs}; O=$L/chain.out; LO=$L/soup_chain.launch.out
verdict() {   # -> prints, returns the exit code (1 = still running)
  if grep -aq '^ZZSOUP-STOP' "$O" 2>/dev/null; then echo "CHAIN STOP: $(grep -a '^ZZSOUP-STOP' "$O" | tail -1)"; return 4; fi
  if grep -aq '^ZZSOUP-DONE' "$O" 2>/dev/null; then echo "CHAIN DONE: $(grep -a '^ZZSOUP-DONE' "$O" | tail -1)"; return 0; fi
  if [ ! -e "$L/chain.lock" ] && grep -aq '^ZZSOUP-STOP' "$LO" 2>/dev/null; then echo "CHAIN STOP (launcher log): $(grep -a '^ZZSOUP-STOP' "$LO" | tail -1)"; return 4; fi
  local pid; pid=$(cat "$L/chain.winpid" 2>/dev/null)
  if [ -n "$pid" ] && ! tasklist //FI "PID eq $pid" //NH 2>/dev/null | grep -q " $pid "; then echo "CHAIN DEAD without an end marker (winpid $pid is gone): rerun the same launcher to resume"; return 6; fi
  return 1
}
if [ "${1:-}" = "--status" ]; then
  tail -6 "$O" 2>/dev/null | cut -c1-260
  verdict; rc=$?; [ $rc -eq 1 ] && echo "CHAIN RUNNING (or not started)"; exit $rc
fi
for i in $(seq 1 "${SOUP_WAKE_MAX_POLLS:-1500}"); do
  verdict && exit 0
  rc=$?; [ $rc -ne 1 ] && { tail -8 "$O" 2>/dev/null | cut -c1-260; exit $rc; }
  sleep "${SOUP_WAKE_POLL_S:-60}"
done
echo "WAKER TIMEOUT"; tail -5 "$O" 2>/dev/null | cut -c1-260; exit 5

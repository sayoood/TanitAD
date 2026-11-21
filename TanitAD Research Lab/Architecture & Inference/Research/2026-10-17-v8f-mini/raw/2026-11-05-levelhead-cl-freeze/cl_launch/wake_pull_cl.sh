#!/usr/bin/env bash
# LEVELHEAD arm CL: wait for the job with wake_cl.sh (script polling only), then pull. Usage: wake_pull_cl.sh wprobe | chain. REFUSES without CL_PI_GO.
#   wprobe: wake_cl.sh wprobe -> pull_wprobe.sh (writes cl_prereg/W_PROBE.json). Exit 0 only when both succeed.
#   chain : re-arms wake_cl.sh after each new RUN_CL_*.done until the chain ends (the box-2 marker folder is new: it holds 0 dones before CL_0), then pull_cl.sh.
# Exit codes: 0 done and pull rc 0; 3 run failed; 4 safety stop; 5 waker timeout; 6 more dones than the chain can make; 7 pull failed; 9 cannot cd.
set -u
: "${CL_PI_GO:?set CL_PI_GO to the PI go time (HH:MM): this script refuses to contact box 2 without the go of the PI}"
MODE=${1:-}; case "$MODE" in wprobe|chain) ;; *) echo "usage: wake_pull_cl.sh wprobe|chain"; exit 64;; esac
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 9
if [ "$MODE" = wprobe ]; then
  out=$(bash wake_cl.sh wprobe); ec=$?
  echo "[$(date '+%m-%d %H:%M:%S')] wprobe ec=$ec $out"
  [ "$ec" -eq 0 ] || exit "$ec"
  bash pull_wprobe.sh > pull_wprobe.out 2>&1; pec=$?
  tail -4 pull_wprobe.out; echo "[$(date '+%m-%d %H:%M:%S')] pull rc=$pec"
  [ "$pec" -eq 0 ] || exit 7
  exit 0
fi
s=${CL_SEEN:-0}
while :; do
  out=$(CL_SEEN=$s bash wake_cl.sh chain); ec=$?
  echo "[$(date '+%m-%d %H:%M:%S')] seen=$s ec=$ec $out"
  case "$out" in *"CHAIN DONE"*) break ;; esac
  [ "$ec" -ne 0 ] && exit "$ec"
  s=$((s + 1)); [ "$s" -gt 2 ] && { echo "more RUN_CL_*.done than the chain makes"; exit 6; }
done
echo "[$(date '+%m-%d %H:%M:%S')] pulling"
bash pull_cl.sh > pull_cl.out 2>&1; pec=$?
tail -5 pull_cl.out
echo "[$(date '+%m-%d %H:%M:%S')] pull rc=$pec"
[ "$pec" -eq 0 ] || exit 7

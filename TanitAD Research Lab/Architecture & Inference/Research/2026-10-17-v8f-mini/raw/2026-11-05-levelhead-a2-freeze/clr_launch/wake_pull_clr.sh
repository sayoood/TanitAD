#!/usr/bin/env bash
# LEVELHEAD arm CLR: wait for the job with wake_clr.sh (script polling only), then pull. Usage: wake_pull_clr.sh chain. REFUSES without CLR_PI_GO.
#   chain : re-arms wake_clr.sh after each new RUN_CLR_*.done until the chain ends (the box-2 marker folder is new: it holds 0 dones before CLR_0), then pull_clr.sh.
# Exit codes: 0 done and pull rc 0; 3 run failed; 4 safety stop; 5 waker timeout; 6 more dones than the chain can make; 7 pull failed; 9 cannot cd.
set -u
: "${CLR_PI_GO:?set CLR_PI_GO to the PI go time (HH:MM): this script refuses to contact box 2 without the go of the PI}"
MODE=${1:-}; case "$MODE" in chain) ;; *) echo "usage: wake_pull_clr.sh chain"; exit 64;; esac
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 9
s=${CLR_SEEN:-0}
while :; do
  out=$(CLR_SEEN=$s bash wake_clr.sh chain); ec=$?
  echo "[$(date '+%m-%d %H:%M:%S')] seen=$s ec=$ec $out"
  case "$out" in *"CHAIN DONE"*) break ;; esac
  [ "$ec" -ne 0 ] && exit "$ec"
  s=$((s + 1)); [ "$s" -gt 2 ] && { echo "more RUN_CLR_*.done than the chain makes"; exit 6; }
done
echo "[$(date '+%m-%d %H:%M:%S')] pulling"
bash pull_clr.sh > pull_clr.out 2>&1; pec=$?
tail -5 pull_clr.out
echo "[$(date '+%m-%d %H:%M:%S')] pull rc=$pec"
[ "$pec" -eq 0 ] || exit 7

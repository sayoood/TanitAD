#!/usr/bin/env bash
# LEVELHEAD arm LLR: wait for the job with wake_llr.sh (script polling only), then pull. Usage: wake_pull_llr.sh wprobe | chain. REFUSES without LLR_PI_GO.
#   chain : re-arms wake_llr.sh after each new RUN_LLR_*.done until the chain ends (the box-2 marker folder is new: it holds 0 dones before LLR_0), then pull_llr.sh.
# Exit codes: 0 done and pull rc 0; 3 run failed; 4 safety stop; 5 waker timeout; 6 more dones than the chain can make; 7 pull failed; 9 cannot cd.
set -u
: "${LLR_PI_GO:?set LLR_PI_GO to the PI go time (HH:MM): this script refuses to contact box 2 without the go of the PI}"
MODE=${1:-}; case "$MODE" in wprobe|chain) ;; *) echo "usage: wake_pull_llr.sh wprobe|chain"; exit 64;; esac
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 9
if [ "$MODE" = wprobe ]; then
  out=$(bash wake_llr.sh wprobe); ec=$?
  echo "[$(date '+%m-%d %H:%M:%S')] wprobe ec=$ec $out"
  [ "$ec" -eq 0 ] || exit "$ec"
  bash pull_wleadprobe.sh > pull_wleadprobe.out 2>&1; pec=$?
  tail -4 pull_wleadprobe.out; echo "[$(date '+%m-%d %H:%M:%S')] pull rc=$pec"
  [ "$pec" -eq 0 ] || exit 7
  exit 0
fi
s=${LLR_SEEN:-0}
while :; do
  out=$(LLR_SEEN=$s bash wake_llr.sh chain); ec=$?
  echo "[$(date '+%m-%d %H:%M:%S')] seen=$s ec=$ec $out"
  case "$out" in *"CHAIN DONE"*) break ;; esac
  [ "$ec" -ne 0 ] && exit "$ec"
  s=$((s + 1)); [ "$s" -gt 2 ] && { echo "more RUN_LLR_*.done than the chain makes"; exit 6; }
done
echo "[$(date '+%m-%d %H:%M:%S')] pulling"
bash pull_llr.sh > pull_llr.out 2>&1; pec=$?
tail -5 pull_llr.out
echo "[$(date '+%m-%d %H:%M:%S')] pull rc=$pec"
[ "$pec" -eq 0 ] || exit 7

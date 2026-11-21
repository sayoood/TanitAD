#!/usr/bin/env bash
# Wait for the dev-box GPU to clear of python compute, then run the >=GT chain.
# ⛔ The emitted marker is DISJOINT from the searched token on purpose. A monitor
# whose filter contains the pattern it searches for matches the PTY echo of its
# own command line and reports a failure that never happened (MEASURED 3x).
set -u
LOG=/c/Users/Admin/tanitad-rlrun/waiter_gtbar.out
MAX_TRIES=120          # 120 x 30s = 1h ceiling on the wait itself
TRY=0
: > "$LOG"
while [ "$TRY" -lt "$MAX_TRIES" ]; do
  TRY=$((TRY+1))
  BUSY=$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null | grep -ci "pyth""on")
  echo "$(date -u +%FT%TZ) try=$TRY ZZBUSY-${BUSY:-x}ZZ" >> "$LOG"
  if [ "${BUSY:-1}" -eq 0 ]; then
    echo "$(date -u +%FT%TZ) ZZGPU-CLEAR-LAUNCHINGZZ" >> "$LOG"
    cd /c/Users/Admin/tanitad-rlrun || exit 1
    bash run_rc21_gtbar.sh >> /c/Users/Admin/tanitad-rlrun/chain_gtbar.out 2>&1
    echo "$(date -u +%FT%TZ) ZZCHAIN-RETURNED-$?ZZ" >> "$LOG"
    exit 0
  fi
  sleep 30
done
echo "$(date -u +%FT%TZ) ZZWAITER-GAVE-UP-AFTER-${TRY}ZZ" >> "$LOG"
exit 7

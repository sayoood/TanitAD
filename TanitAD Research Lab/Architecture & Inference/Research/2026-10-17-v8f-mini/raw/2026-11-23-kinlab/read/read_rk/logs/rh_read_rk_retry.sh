#!/bin/bash
# rh_read_rk_retry.sh -- MM 2026-11-23. Runs rh_read_rk_chain.sh and re-runs it ONLY when it stopped on the dev-box commit-headroom gate
# (C: holds the pagefile and has ~2 GB free, so commit headroom sits near 3.7 GB while other sessions run). The chain is resumable
# (out/steps_rk/<name>.ok) and the gate fails BEFORE a step runs, so a re-run repeats no work. Any other failure stops here.
# At most 8 runs, 10 min apart. Last line of out/rh_read_rk_retry.log: "ZZ RK-RETRY DONE ZZ" or "ZZ RK-RETRY STOP ZZ <why>".
export PATH="/usr/bin:/bin:$PATH"
E=F:/TanitAD-scratch/rollhist/read
O=$E/out; LOG=$O/rh_read_rk_chain.log; RL=$O/rh_read_rk_retry.log
mkdir -p "$O"
for i in 1 2 3 4 5 6 7 8; do
  echo "$(date -u +%FT%TZ) run $i" >> "$RL"
  bash $E/rh_read_rk_chain.sh < /dev/null
  last=$(tail -n 1 "$LOG" 2>/dev/null)
  case "$last" in
    *"ZZ ROLLHIST-READ-RK DONE ZZ"*) echo "$(date -u +%FT%TZ) ZZ RK-RETRY DONE ZZ" >> "$RL"; exit 0 ;;
    *"commit headroom stayed below"*) echo "$(date -u +%FT%TZ) headroom stop, waiting 10 min: $last" >> "$RL"; rm -f -- "F:/TanitAD-scratch/rollhist/read/out/ROLLHIST_READ_RK.done"; sleep 600 ;;
    *) echo "$(date -u +%FT%TZ) ZZ RK-RETRY STOP ZZ chain ended without DONE: $last" >> "$RL"; exit 1 ;;
  esac
done
echo "$(date -u +%FT%TZ) ZZ RK-RETRY STOP ZZ 8 runs, headroom still short" >> "$RL"; exit 1

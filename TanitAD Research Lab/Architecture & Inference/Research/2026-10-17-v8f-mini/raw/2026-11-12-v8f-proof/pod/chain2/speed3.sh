#!/usr/bin/env bash
# speed3.sh <hierrep arm, for example DAG_s1> <spatial arm> <dataeff arm>
# Waits until the HIERREP chain logs "train <arm>", then 5 minutes, then reads the last step of the three arms at both ends of a 5-minute window.
# Writes /workspace/v8proof/chains/speed3_<arm>.txt. Read-only on the runs; no GPU; a few sleeps.
set -u
A=${1:?hierrep arm}; S=${2:?spatial arm}; D=${3:?dataeff arm}
W=/workspace/v8proof; O=$W/chains/speed3_$A.txt
rd() { for a in hierrep/$A spatial/$S dataeff/$D; do n=$(tail -n 1 "$W/runs/$a/train_log.jsonl" 2>/dev/null | head -c 400 | grep -o '"step": *[0-9]*' | grep -o '[0-9]*$'); echo "$a ${n:-none} $(stat -c %Y "$W/runs/$a/train_log.jsonl" 2>/dev/null)"; done; }
n=0; until grep -q "train $A " "$W/hierrep/stageB.log" 2>/dev/null; do n=$((n+1)); [ $n -gt 1440 ] && { echo "TIMEOUT waiting for train $A" > "$O"; exit 1; }; sleep 10; done
{ echo "train $A logged; $(date -u +%Y-%m-%dT%H:%M:%SZ)"; } > "$O"
sleep 300
{ echo "window start $(date -u +%H:%M:%SZ)"; rd; } >> "$O"
sleep 300
{ echo "window end $(date -u +%H:%M:%SZ)"; rd; tail -n 1 "$W/chains/spatial/chain.log" | cut -c1-100; tail -n 1 "$W/chains/dataeff/chain.log" | cut -c1-100; echo SPEED3_DONE; } >> "$O"

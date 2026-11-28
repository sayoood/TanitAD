#!/usr/bin/env bash
# early_wait.sh <ARM> : dev box. Polls the pod every 180 s until both fine-tunes of <ARM> (bases 0 and 1, chains actlevel_0 and actlevel_1) have finished training
# (the chain log shows the T1-kin dump or the ARM DONE line of the arm), bound 6 h; then runs actlevel/early_read2_cap.sh <ARM>. It starts no GPU job, starts no
# training, deletes nothing. Last line: that of early_read.sh, or ZZ EARLY-WAIT <ARM> TIMEOUT | STOP.
ARM=${1:?usage: early_wait.sh ARM}
case "$ARM" in RLVc|RLVd) ;; *) echo "not an arm of note N7: $ARM"; exit 2 ;; esac
P=/c/Users/Admin/v8pod/pssh.sh
H=/f/TanitAD-scratch/v8proof
cd $H || exit 1
ts() { date -u +%H:%M:%SZ; }
n=0
while :; do
  S=$(timeout 60 bash "$P" "cd /workspace/v8proof/chains; for k in 0 1; do grep -aq 'T1-kin dump ${ARM}_s'\$k'\|ARM DONE ${ARM}_s'\$k actlevel2_\$k/chain.log 2>/dev/null && echo TRAINED\$k; [ -f actlevel2_\$k/actlevel2_\${k}_STOP.txt ] && echo STOP\$k; done; echo MARK" 2>/dev/null | sed '/hf_/d' | tr -d '\r')
  if echo "$S" | grep -q '^TRAINED0$' && echo "$S" | grep -q '^TRAINED1$'; then break; fi
  if echo "$S" | grep -q '^STOP'; then echo "ZZ EARLY-WAIT $ARM STOP: a chain wrote its STOP marker ($(echo "$S" | tr '\n' ' ')) $(ts)"; exit 0; fi
  n=$((n+1)); [ $((n % 10)) = 1 ] && echo "[$(ts)] poll $n: $(echo "$S" | tr '\n' ' ')"
  [ "$n" -gt 120 ] && { echo "ZZ EARLY-WAIT $ARM TIMEOUT $(ts)"; exit 8; }
  sleep 180
done
echo "[$(ts)] both fine-tunes of $ARM have finished training"
bash actlevel/early_read2_cap.sh "$ARM"

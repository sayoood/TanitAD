#!/usr/bin/env bash
# wait_i10k_n2.sh : dev-box waiter for the continuation queue of card PREREG_V8F_PROOF_ADD46 (note N2), for the case that the first local chain process ended
# before the pod queue did. It starts nothing on the pod. It polls the pod once a minute for INTEG10K_N2_DONE.txt or INTEG10K_N2_STOP.txt (at most 8 h),
# then runs chain_i10k_n2.sh --no-start (push check and pull only) and, when the queue ended with DONE, read_i10k_n2.sh.
# Last line: "ZZ WAIT-I10K-N2 DONE", "ZZ WAIT-I10K-N2 STOP <reason>".
I=/f/TanitAD-scratch/v8proof/integ10k
P=/c/Users/Admin/v8pod/pssh.sh
W=/workspace/v8proof/integ10k
ts() { date -u +%H:%M:%SZ; }
k=0; st=""
while :; do
  s=$(timeout 60 bash "$P" "ls $W/INTEG10K_N2_DONE.txt $W/INTEG10K_N2_STOP.txt 2>/dev/null | sed 's|.*/||' | tr '\n' ' '" 2>/dev/null | sed '/hf_/d' | tr -d '\r')
  case "$s" in *STOP*) st=STOP; break;; *DONE*) st=DONE; break;; esac
  k=$((k+1)); [ "$k" -gt 480 ] && { echo "ZZ WAIT-I10K-N2 STOP not ended after 8 h $(ts)"; exit 3; }
  sleep 60
done
echo "[$(ts)] pod queue ended: $st"
cd "$I" || exit 2
bash chain_i10k_n2.sh --no-start > chain_i10k_n2_pull.out 2>&1
tail -n 3 chain_i10k_n2_pull.out | cut -c1-200
if [ "$st" = DONE ]; then
  bash read_i10k_n2.sh > read_i10k_n2.out 2>&1
  tail -n 60 read_i10k_n2.out | cut -c1-420
  echo "ZZ WAIT-I10K-N2 DONE $(ts)"
else
  echo "ZZ WAIT-I10K-N2 STOP pod queue stopped $(ts)"
fi

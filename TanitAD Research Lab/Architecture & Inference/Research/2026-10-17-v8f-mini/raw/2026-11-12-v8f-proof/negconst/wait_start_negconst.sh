#!/usr/bin/env bash
# wait_start_negconst.sh -- dev-box waiter of card ADD82 (lead's ops file): starts the pod check nk1, waits for its marker, and when the check says PASS runs the
# dry run of the queue and then starts the queue (the queue itself refuses without the PASS record and the hashes). Writes negconst/wait_start_negconst.out.
# No process stopping, no file removal. The pod address is read by pssh.sh only.
set -u
PS=/c/Users/Admin/v8pod/pssh.sh
W=/workspace/v8proof/integ10k
B=$W/build
OUT=/f/TanitAD-scratch/v8proof/negconst/wait_start_negconst.out
p() { timeout 120 bash $PS "$1" 2>/dev/null | sed '/hf_/d' | tr -d '\r'; }
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
echo "$(ts) start check nk1" > "$OUT"
p "cd $B && (nohup setsid bash pod_check_negconst.sh > pod_check_negconst_nk1.nohup 2>&1 < /dev/null &) ; sleep 2; echo started" >> "$OUT"
n=0
while [ $n -lt 60 ]; do
  sleep 60; n=$((n+1))
  m=$(p "ls $W/negconst_test/CHECK_NEG_DONE_nk1.txt $W/negconst_test/CHECK_NEG_STOP_nk1.txt 2>/dev/null | wc -l")
  [ "${m:-0}" -ge 1 ] 2>/dev/null && break
done
echo "$(ts) check marker seen after $n polls" >> "$OUT"
p "tail -1 $W/negconst_test/CHECK_NEG_DONE_nk1.txt 2>/dev/null | cut -c1-200; cat $W/negconst_test/CHECK_NEG_STOP_nk1.txt 2>/dev/null | cut -c1-200" >> "$OUT"
if ! grep -q "ZZ CHECK-NEGCONST PASS" "$OUT"; then echo "$(ts) ZZ WAIT-START NO-PASS" >> "$OUT"; exit 3; fi
p "cd $B && DRY=1 I10_W=$W bash pod_negconst10k_queue.sh > dry_negconst.out 2>&1; grep -c CMD dry_negconst.out; tail -1 dry_negconst.out | cut -c1-120" >> "$OUT"
if ! grep -q "DRY DONE" "$OUT"; then echo "$(ts) ZZ WAIT-START DRY-NOT-DONE" >> "$OUT"; exit 4; fi
p "cd $B && (nohup setsid bash pod_negconst10k_queue.sh > queue_negconst.nohup 2>&1 < /dev/null &) ; sleep 5; cat $W/START_NEGCONST10K.txt 2>/dev/null | cut -c1-60" >> "$OUT"
echo "$(ts) ZZ WAIT-START QUEUE-STARTED" >> "$OUT"

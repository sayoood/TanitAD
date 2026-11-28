#!/usr/bin/env bash
# wait_start_shufclose.sh -- dev-box waiter of card ADD83 (lead's ops file): starts the pod check sk1 (control u0), waits for its marker, and when the check says PASS runs the
# dry run of the queue and then starts the queue (the queue itself refuses without the PASS record, the hashes and the surrogate record). Writes shufclose/wait_start_shufclose.out.
# No process stopping, no file removal. The pod address is read by pssh.sh only.
set -u
PS=/c/Users/Admin/v8pod/pssh.sh
W=/workspace/v8proof/integ10k
B=$W/build
OUT=/f/TanitAD-scratch/v8proof/shufclose/wait_start_shufclose.out
p() { timeout 120 bash $PS "$1" 2>/dev/null | sed '/hf_/d' | tr -d '\r'; }
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
echo "$(ts) start check sk1" > "$OUT"
p "cd $B && (nohup setsid bash pod_check_shufclose.sh > pod_check_shufclose_sk1.nohup 2>&1 < /dev/null &) ; sleep 2; echo started" >> "$OUT"
n=0
while [ $n -lt 90 ]; do
  sleep 60; n=$((n+1))
  m=$(p "ls $W/shufclose_test/ 2>/dev/null | grep -c '^CHECK_SHUF_\(DONE\|STOP\)_sk1.txt'")
  [ "${m:-0}" -ge 1 ] 2>/dev/null && break
done
echo "$(ts) check marker seen after $n polls" >> "$OUT"
p "tail -1 $W/shufclose_test/CHECK_SHUF_DONE_sk1.txt 2>/dev/null | cut -c1-200; cat $W/shufclose_test/CHECK_SHUF_STOP_sk1.txt 2>/dev/null | cut -c1-200" >> "$OUT"
if ! grep -q "ZZ CHECK-SHUFCLOSE PASS" "$OUT"; then echo "$(ts) ZZ WAIT-START NO-PASS" >> "$OUT"; exit 3; fi
p "cd $B && DRY=1 I10_W=$W bash pod_shufclose10k_queue.sh > dry_shufclose.out 2>&1; grep -c CMD dry_shufclose.out; tail -1 dry_shufclose.out | cut -c1-120" >> "$OUT"
if ! grep -q "DRY DONE" "$OUT"; then echo "$(ts) ZZ WAIT-START DRY-NOT-DONE" >> "$OUT"; exit 4; fi
p "cd $B && (nohup setsid bash pod_shufclose10k_queue.sh > queue_shufclose.nohup 2>&1 < /dev/null &) ; sleep 5; cat $W/START_SHUFCLOSE10K.txt 2>/dev/null | cut -c1-60" >> "$OUT"
echo "$(ts) ZZ WAIT-START QUEUE-STARTED" >> "$OUT"

#!/bin/bash
# run_pod_read.sh -- dev box. Card PREREG_V8F_PROOF_ADD41 with note N1. Waits for the last line of finegrid/wait_finegrid.out (the pilot ended on the pod).
# On DONE it starts podkit/pod_read.sh on the pod (nohup setsid; the start line holds no process test; a second start is refused there by a marker file),
# polls for its READ_DONE.txt, and pulls the reader's small outputs into finegrid/out_pod/. On STOP or TIMEOUT of the pilot it ends without a read.
# Nothing is started on the pod GPU. Deletes nothing. Last line: ZZ FINEGRID-READ <state>.
set -u
cd /f/TanitAD-scratch/v8proof/finegrid || exit 9
P=/c/Users/Admin/v8pod/pssh.sh
C=/c/Users/Admin/v8pod/pcat.sh
RK=/workspace/v8proof/finegrid/rk
ts() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
pod() { timeout 60 bash "$P" "$1" 2>/dev/null | sed '/hf_/d' | tr -d '\r'; }
n=0
while true; do
  l=$(grep -a "^ZZ FINEGRID-POD " wait_finegrid.out 2>/dev/null | tail -n 1)
  [ -n "$l" ] && break
  n=$((n+1)); [ "$n" -gt 420 ] && { echo "ZZ FINEGRID-READ TIMEOUT (no end line of the pilot waiter in 14 h) $(ts)"; exit 8; }
  sleep 120
done
echo "[$(ts)] pilot waiter: $l"
case "$l" in *" DONE "*) : ;; *) echo "ZZ FINEGRID-READ NOT-RUN (the pilot did not end with DONE) $(ts)"; exit 0 ;; esac
a=$(sha256sum podkit/pod_read.sh | cut -d' ' -f1)
timeout 60 bash "$P" "cat > $RK/pod_read.sh" < podkit/pod_read.sh 2>/dev/null | sed '/hf_/d'
b=$(pod "sha256sum $RK/pod_read.sh | cut -d' ' -f1")
[ "$a" = "$b" ] || { echo "ZZ FINEGRID-READ STOP (pod_read.sh sha mismatch) $(ts)"; exit 5; }
timeout 30 bash "$P" "cd $RK && nohup setsid bash pod_read.sh > pod_read.nohup 2>&1 < /dev/null &" > /dev/null 2>&1
n=0
while true; do
  sleep 60
  d=$(pod "cat $RK/READ_DONE.txt 2>/dev/null | head -n 1")
  [ -n "$d" ] && break
  n=$((n+1)); [ "$n" -gt 240 ] && { echo "ZZ FINEGRID-READ TIMEOUT (no READ_DONE.txt in 4 h): $(pod "tail -n 2 $RK/pod_read.nohup | cut -c1-200")"; exit 8; }
done
echo "[$(ts)] pod read ended: $d"
mkdir -p out_pod
for f in out_read/RESULT_FINEGRID.json out_read/VERDICT_FINEGRID.md out_read/TK_REPRO_G7.json read.out pod_read.nohup READ_DONE.txt READ_STARTED.txt; do
  s=$(pod "sha256sum $RK/$f 2>/dev/null | cut -d' ' -f1")
  [ -n "$s" ] || { echo "[$(ts)] not on the pod: $f"; continue; }
  o=out_pod/$(basename "$f")
  bash "$C" "$RK/$f" "$o" < /dev/null > /dev/null 2>&1
  t=$(sha256sum "$o" 2>/dev/null | cut -d' ' -f1)
  if [ "$s" = "$t" ]; then echo "[$(ts)] pulled $f sha256 ${s:0:16}"; else echo "[$(ts)] PULL-MISMATCH $f"; fi
done
tail -n 3 out_pod/read.out 2>/dev/null | cut -c1-240
echo "ZZ FINEGRID-READ ${d%% *} $(ts)"

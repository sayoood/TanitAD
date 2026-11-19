#!/usr/bin/env bash
# wait_podread_cycle.sh : dev-box waiter of card ADD52, bash only (no Python on the dev box). Waits for the pod marker CYCLE10K_DONE.txt (or _STOP), then runs the read on
# the pod (podread_cycle.sh, pushed before) and brings the small result files back as text. Last line: ZZ WAIT-PODREAD-CYCLE DONE or STOP.
H=/f/TanitAD-scratch/v8proof; I=$H/integ10k; P=/c/Users/Admin/v8pod/pssh.sh
W=/workspace/v8proof/integ10k; R=/workspace/v8proof/podread
ts() { date -u +%H:%M:%SZ; }
pod() { timeout "${PODT:-60}" bash "$P" "$1" 2>/dev/null | sed '/hf_/d' | tr -d '\r'; }
k=0
while :; do
  s=$(pod "ls $W/CYCLE10K_DONE.txt $W/CYCLE10K_STOP.txt 2>/dev/null | sed 's|.*/||' | tr '\n' ' '")
  case "$s" in *STOP*) echo "[$(ts)] queue STOP: $(pod "cat $W/CYCLE10K_STOP.txt | cut -c1-200")"; echo "ZZ WAIT-PODREAD-CYCLE STOP queue"; exit 3;; *DONE*) break;; esac
  k=$((k+1)); [ "$k" -gt 600 ] && { echo "ZZ WAIT-PODREAD-CYCLE STOP not ended after 5 h"; exit 3; }
  sleep 30
done
echo "[$(ts)] queue DONE on the pod; the pod read starts"
PODT=1500 pod "cd $R && bash podread_cycle.sh > podread_cycle.out 2>&1; tail -n 140 podread_cycle.out | cut -c1-430"
mkdir -p "$I/read_cycle/pod"
for f in out/read_cycle/READ_CYCLE10K.json out/e1y/ENV_E1.json out/read_cycle/hw/RESULT_HEADWAY.json out/read_cycle/q3/RESULT_Q3.json out/read_cycle/hw/VERDICT_HEADWAY.md out/read_cycle/q3/VERDICT_Q3.md podread_cycle.out; do
  pod "cat $R/$f" > "$I/read_cycle/pod/$(basename $f)"
  a=$(pod "sha256sum $R/$f | cut -c1-64"); b=$(sha256sum "$I/read_cycle/pod/$(basename $f)" | cut -c1-64)
  [ "$a" = "$b" ] && echo "[$(ts)] pulled $(basename $f) sha256 ${a:0:16}" || echo "[$(ts)] PULL-MISMATCH $(basename $f)"
done
echo "ZZ WAIT-PODREAD-CYCLE DONE $(ts)"

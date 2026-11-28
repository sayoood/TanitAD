#!/usr/bin/env bash
# wait_podread_rule1s.sh -- written from wait_podread_accwin.sh / wait_podread_brake.sh: dev-box waiter of card ADD59, bash only (no Python on the dev box).
# Waits for the pod marker RULE1S10K_DONE.txt (or _STOP), then runs the read on the pod (podread_rule1s.sh, pushed before into the podread folder, together with
# read_rule1s10k.py and RULE1S10K_REUSED.json in its root/integ10k folder) and brings the small result files back as text into read_rule1s/pod/.
# Last line: ZZ WAIT-PODREAD-RULE1S DONE or STOP.
H=/f/TanitAD-scratch/v8proof; I=$H/integ10k; P=/c/Users/Admin/v8pod/pssh.sh
W=/workspace/v8proof/integ10k; R=/workspace/v8proof/podread
ts() { date -u +%H:%M:%SZ; }
pod() { timeout "${PODT:-60}" bash "$P" "$1" 2>/dev/null | sed '/hf_/d' | tr -d '\r'; }
k=0
while :; do
  s=$(pod "ls $W/RULE1S10K_DONE.txt $W/RULE1S10K_STOP.txt 2>/dev/null | sed 's|.*/||' | tr '\n' ' '")
  case "$s" in *STOP*) echo "[$(ts)] queue STOP: $(pod "cat $W/RULE1S10K_STOP.txt | cut -c1-200")"; echo "ZZ WAIT-PODREAD-RULE1S STOP queue"; exit 3;; *DONE*) break;; esac
  k=$((k+1)); [ "$k" -gt 1800 ] && { echo "ZZ WAIT-PODREAD-RULE1S STOP not ended after 15 h"; exit 3; }
  sleep 30
done
echo "[$(ts)] queue DONE on the pod; the pod read starts"
PODT=2400 pod "cd $R && bash podread_rule1s.sh > podread_rule1s.out 2>&1; tail -n 240 podread_rule1s.out | cut -c1-430"
mkdir -p "$I/read_rule1s/pod"
for f in out/read_rule1s/READ_RULE1S10K.json out/e1r/ENV_E1.json out/read_rule1s/hw/RESULT_HEADWAY.json out/read_rule1s/q3/RESULT_Q3.json out/read_rule1s/hw/VERDICT_HEADWAY.md out/read_rule1s/q3/VERDICT_Q3.md out/read_rule1s/read_rule1s10k.log podread_rule1s.out; do
  pod "cat $R/$f" > "$I/read_rule1s/pod/$(basename $f)"
  a=$(pod "sha256sum $R/$f | cut -c1-64"); b=$(sha256sum "$I/read_rule1s/pod/$(basename $f)" | cut -c1-64)
  [ "$a" = "$b" ] && echo "[$(ts)] pulled $(basename $f) sha256 ${a:0:16}" || echo "[$(ts)] PULL-MISMATCH $(basename $f)"
done
echo "ZZ WAIT-PODREAD-RULE1S DONE $(ts)"

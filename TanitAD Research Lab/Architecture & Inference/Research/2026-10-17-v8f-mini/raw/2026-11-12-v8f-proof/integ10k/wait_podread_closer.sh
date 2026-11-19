#!/usr/bin/env bash
# wait_podread_closer.sh : dev-box waiter of card ADD49, bash only (no Python on the dev box). Waits for the pod marker CLOSER10K_DONE.txt (or _STOP), then runs the read on
# the pod (podread_closer.sh, pushed before) and brings the small result files back as text. Last line: ZZ WAIT-PODREAD-CLOSER DONE or STOP.
H=/f/TanitAD-scratch/v8proof; I=$H/integ10k; P=/c/Users/Admin/v8pod/pssh.sh
W=/workspace/v8proof/integ10k; R=/workspace/v8proof/podread
ts() { date -u +%H:%M:%SZ; }
pod() { timeout "${PODT:-60}" bash "$P" "$1" 2>/dev/null | sed '/hf_/d' | tr -d '\r'; }
k=0
while :; do
  s=$(pod "ls $W/CLOSER10K_DONE.txt $W/CLOSER10K_STOP.txt 2>/dev/null | sed 's|.*/||' | tr '\n' ' '")
  case "$s" in *STOP*) echo "[$(ts)] queue STOP: $(pod "cat $W/CLOSER10K_STOP.txt | cut -c1-200")"; echo "ZZ WAIT-PODREAD-CLOSER STOP queue"; exit 3;; *DONE*) break;; esac
  k=$((k+1)); [ "$k" -gt 240 ] && { echo "ZZ WAIT-PODREAD-CLOSER STOP not ended after 2 h"; exit 3; }
  sleep 30
done
echo "[$(ts)] queue DONE on the pod; the pod read starts"
PODT=900 pod "cd $R && bash podread_closer.sh > podread_closer.out 2>&1; tail -n 80 podread_closer.out | cut -c1-430"
mkdir -p "$I/read_closer/pod"
for f in out/read_closer/READ_CLOSER10K.json out/e1h/ENV_E1.json out/read_closer/hw/RESULT_HEADWAY.json out/read_closer/q3/RESULT_Q3.json out/read_closer/hw/VERDICT_HEADWAY.md out/read_closer/q3/VERDICT_Q3.md podread_closer.out; do
  pod "cat $R/$f" > "$I/read_closer/pod/$(basename $f)"
  a=$(pod "sha256sum $R/$f | cut -c1-64"); b=$(sha256sum "$I/read_closer/pod/$(basename $f)" | cut -c1-64)
  [ "$a" = "$b" ] && echo "[$(ts)] pulled $(basename $f) sha256 ${a:0:16}" || echo "[$(ts)] PULL-MISMATCH $(basename $f)"
done
echo "ZZ WAIT-PODREAD-CLOSER DONE $(ts)"

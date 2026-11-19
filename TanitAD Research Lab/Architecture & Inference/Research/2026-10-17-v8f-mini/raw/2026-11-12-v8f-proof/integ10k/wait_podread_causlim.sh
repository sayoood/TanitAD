#!/usr/bin/env bash
# wait_podread_causlim.sh -- dev-box waiter of card PREREG_V8F_PROOF_ADD61 (CAUSLIM10K), written from wait_podread_accwin2s.sh (card ADD58), bash only (no Python on the dev box).
# Waits for the pod marker CAUSLIM10K_DONE.txt (or _STOP), then runs the read on the pod (podread_causlim.sh, pushed before) and brings the small result files back as text
# into read_causlim/pod/. Last line: ZZ WAIT-PODREAD-CAUSLIM DONE or STOP.
H=/f/TanitAD-scratch/v8proof; I=$H/integ10k; P=/c/Users/Admin/v8pod/pssh.sh
W=/workspace/v8proof/integ10k; R=/workspace/v8proof/podread
ts() { date -u +%H:%M:%SZ; }
pod() { timeout "${PODT:-60}" bash "$P" "$1" 2>/dev/null | sed '/hf_/d' | tr -d '\r'; }
k=0
while :; do
  s=$(pod "ls $W/CAUSLIM10K_DONE.txt $W/CAUSLIM10K_STOP.txt 2>/dev/null | sed 's|.*/||' | tr '\n' ' '")
  case "$s" in *STOP*) echo "[$(ts)] queue STOP: $(pod "cat $W/CAUSLIM10K_STOP.txt | cut -c1-200")"; echo "ZZ WAIT-PODREAD-CAUSLIM STOP queue"; exit 3;; *DONE*) break;; esac
  k=$((k+1)); [ "$k" -gt 5760 ] && { echo "ZZ WAIT-PODREAD-CAUSLIM STOP not ended after 48 h"; exit 3; }
  sleep 30
done
echo "[$(ts)] queue DONE on the pod; the pod read starts"
PODT=2400 pod "cd $R && bash podread_causlim.sh > podread_causlim.out 2>&1; tail -n 240 podread_causlim.out | cut -c1-430"
mkdir -p "$I/read_causlim/pod"
for f in out/read_causlim/READ_CAUSLIM10K.json out/e1cl/ENV_E1.json out/read_causlim/hw/RESULT_HEADWAY.json out/read_causlim/q3/RESULT_Q3.json out/read_causlim/hw/VERDICT_HEADWAY.md out/read_causlim/q3/VERDICT_Q3.md podread_causlim.out; do
  pod "cat $R/$f" > "$I/read_causlim/pod/$(basename $f)"
  a=$(pod "sha256sum $R/$f | cut -c1-64"); b=$(sha256sum "$I/read_causlim/pod/$(basename $f)" | cut -c1-64)
  [ "$a" = "$b" ] && echo "[$(ts)] pulled $(basename $f) sha256 ${a:0:16}" || echo "[$(ts)] PULL-MISMATCH $(basename $f)"
done
echo "ZZ WAIT-PODREAD-CAUSLIM DONE $(ts)"

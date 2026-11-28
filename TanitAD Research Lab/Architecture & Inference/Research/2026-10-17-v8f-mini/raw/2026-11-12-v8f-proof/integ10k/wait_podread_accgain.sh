#!/usr/bin/env bash
# wait_podread_accgain.sh -- dev-box waiter of card PREREG_V8F_PROOF_ADD63 (ACCGAIN10K), written by make_accgain10k.py from wait_podread_accgain.sh (card ADD61), bash only (no Python on the dev box).
# Waits for the pod marker ACCGAIN10K_DONE.txt (or _STOP), then runs the read on the pod (podread_accgain.sh, pushed before) and brings the small result files back as text
# into read_accgain/pod/. Last line: ZZ WAIT-PODREAD-ACCGAIN DONE or STOP.
H=/f/TanitAD-scratch/v8proof; I=$H/integ10k; P=/c/Users/Admin/v8pod/pssh.sh
W=/workspace/v8proof/integ10k; R=/workspace/v8proof/podread
ts() { date -u +%H:%M:%SZ; }
pod() { timeout "${PODT:-60}" bash "$P" "$1" 2>/dev/null | sed '/hf_/d' | tr -d '\r'; }
k=0
while :; do
  s=$(pod "ls $W/ACCGAIN10K_DONE.txt $W/ACCGAIN10K_STOP.txt 2>/dev/null | sed 's|.*/||' | tr '\n' ' '")
  case "$s" in *STOP*) echo "[$(ts)] queue STOP: $(pod "cat $W/ACCGAIN10K_STOP.txt | cut -c1-200")"; echo "ZZ WAIT-PODREAD-ACCGAIN STOP queue"; exit 3;; *DONE*) break;; esac
  k=$((k+1)); [ "$k" -gt 5760 ] && { echo "ZZ WAIT-PODREAD-ACCGAIN STOP not ended after 48 h"; exit 3; }
  sleep 30
done
echo "[$(ts)] queue DONE on the pod; the pod read starts"
PODT=2400 pod "cd $R && bash podread_accgain.sh > podread_accgain.out 2>&1; tail -n 240 podread_accgain.out | cut -c1-430"
mkdir -p "$I/read_accgain/pod"
for f in out/read_accgain/READ_ACCGAIN10K.json out/e1cl/ENV_E1.json out/read_accgain/hw/RESULT_HEADWAY.json out/read_accgain/q3/RESULT_Q3.json out/read_accgain/hw/VERDICT_HEADWAY.md out/read_accgain/q3/VERDICT_Q3.md podread_accgain.out; do
  pod "cat $R/$f" > "$I/read_accgain/pod/$(basename $f)"
  a=$(pod "sha256sum $R/$f | cut -c1-64"); b=$(sha256sum "$I/read_accgain/pod/$(basename $f)" | cut -c1-64)
  [ "$a" = "$b" ] && echo "[$(ts)] pulled $(basename $f) sha256 ${a:0:16}" || echo "[$(ts)] PULL-MISMATCH $(basename $f)"
done
pod "cat $W/accgain_test/CHECK_ACC_GAIN.txt" > "$I/read_accgain/pod/CHECK_ACC_GAIN.txt"
a=$(pod "sha256sum $W/accgain_test/CHECK_ACC_GAIN.txt | cut -c1-64"); b=$(sha256sum "$I/read_accgain/pod/CHECK_ACC_GAIN.txt" | cut -c1-64)
[ "$a" = "$b" ] && echo "[$(ts)] pulled CHECK_ACC_GAIN.txt sha256 ${a:0:16}" || echo "[$(ts)] PULL-MISMATCH CHECK_ACC_GAIN.txt"
echo "ZZ WAIT-PODREAD-ACCGAIN DONE $(ts)"

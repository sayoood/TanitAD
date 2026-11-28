#!/usr/bin/env bash
# wait_podread_boundtier.sh -- dev-box waiter of card PREREG_V8F_PROOF_ADD68 (BOUNDTIER10K), adapted from wait_podread_headbound.sh (card ADD67), bash only (no Python on the dev box).
# Waits for the pod marker BOUNDTIER10K_DONE.txt (or _STOP), then runs the read on the pod (podread_boundtier.sh, pushed before) and brings the small result files back as text
# into read_boundtier/pod/. Last line: ZZ WAIT-PODREAD-BOUNDTIER DONE or STOP.
H=/f/TanitAD-scratch/v8proof; I=$H/integ10k; P=/c/Users/Admin/v8pod/pssh.sh
W=/workspace/v8proof/integ10k; R=/workspace/v8proof/podread
ts() { date -u +%H:%M:%SZ; }
pod() { timeout "${PODT:-60}" bash "$P" "$1" 2>/dev/null | sed '/hf_/d' | tr -d '\r'; }
k=0
while :; do
  s=$(pod "ls $W/BOUNDTIER10K_DONE.txt $W/BOUNDTIER10K_STOP.txt 2>/dev/null | sed 's|.*/||' | tr '\n' ' '")
  case "$s" in *STOP*) echo "[$(ts)] queue STOP: $(pod "cat $W/BOUNDTIER10K_STOP.txt | cut -c1-200")"; echo "ZZ WAIT-PODREAD-BOUNDTIER STOP queue"; exit 3;; *DONE*) break;; esac
  k=$((k+1)); [ "$k" -gt 8640 ] && { echo "ZZ WAIT-PODREAD-BOUNDTIER STOP not ended after 72 h"; exit 3; }
  sleep 30
done
echo "[$(ts)] queue DONE on the pod; the pod read starts"
PODT=3600 pod "cd $R && bash podread_boundtier.sh > podread_boundtier.out 2>&1; tail -n 240 podread_boundtier.out | cut -c1-430"
mkdir -p "$I/read_boundtier/pod"
for f in out/read_boundtier/READ_BOUNDTIER10K.json out/e1bt/ENV_E1.json out/read_boundtier/hw/RESULT_HEADWAY.json out/read_boundtier/q3/RESULT_Q3.json out/read_boundtier/hw/VERDICT_HEADWAY.md out/read_boundtier/q3/VERDICT_Q3.md podread_boundtier.out; do
  pod "cat $R/$f" > "$I/read_boundtier/pod/$(basename $f)"
  a=$(pod "sha256sum $R/$f | cut -c1-64"); b=$(sha256sum "$I/read_boundtier/pod/$(basename $f)" | cut -c1-64)
  [ "$a" = "$b" ] && echo "[$(ts)] pulled $(basename $f) sha256 ${a:0:16}" || echo "[$(ts)] PULL-MISMATCH $(basename $f)"
done
for f in CHECK_HEAD_BOUND.txt CHECK_RULE_EVERY_RZGFH.txt CHECK_ACC_GAIN_RZGFH.txt; do
  pod "cat $W/bt_test/$f" > "$I/read_boundtier/pod/$f"
  a=$(pod "sha256sum $W/bt_test/$f | cut -c1-64"); b=$(sha256sum "$I/read_boundtier/pod/$f" | cut -c1-64)
  [ "$a" = "$b" ] && echo "[$(ts)] pulled $f sha256 ${a:0:16}" || echo "[$(ts)] PULL-MISMATCH $f"
done
echo "ZZ WAIT-PODREAD-BOUNDTIER DONE $(ts)"

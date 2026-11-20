#!/usr/bin/env bash
# wait_podread_cand.sh -- dev-box waiter of card PREREG_V8F_PROOF_ADD65 (CAND10K), written by make_cand10k.py from wait_podread_rulefast.sh (card ADD62), bash only (no Python on the dev box).
# Waits for the pod marker CAND10K_DONE.txt (or _STOP), then runs the read on the pod (podread_cand.sh, pushed before) and brings the small result files back as text
# into read_cand/pod/. Last line: ZZ WAIT-PODREAD-CAND DONE or STOP. (These two comment lines were set by hand at the review; the generator carried the parent's names in them.)
H=/f/TanitAD-scratch/v8proof; I=$H/integ10k; P=/c/Users/Admin/v8pod/pssh.sh
W=/workspace/v8proof/integ10k; R=/workspace/v8proof/podread
ts() { date -u +%H:%M:%SZ; }
pod() { timeout "${PODT:-60}" bash "$P" "$1" 2>/dev/null | sed '/hf_/d' | tr -d '\r'; }
k=0
while :; do
  s=$(pod "ls $W/CAND10K_DONE.txt $W/CAND10K_STOP.txt 2>/dev/null | sed 's|.*/||' | tr '\n' ' '")
  case "$s" in *STOP*) echo "[$(ts)] queue STOP: $(pod "cat $W/CAND10K_STOP.txt | cut -c1-200")"; echo "ZZ WAIT-PODREAD-CAND STOP queue"; exit 3;; *DONE*) break;; esac
  k=$((k+1)); [ "$k" -gt 8640 ] && { echo "ZZ WAIT-PODREAD-CAND STOP not ended after 72 h"; exit 3; }
  sleep 30
done
echo "[$(ts)] queue DONE on the pod; the pod read starts"
PODT=2400 pod "cd $R && bash podread_cand.sh > podread_cand.out 2>&1; tail -n 240 podread_cand.out | cut -c1-430"
mkdir -p "$I/read_cand/pod"
for f in out/read_cand/READ_CAND10K.json out/e1cd/ENV_E1.json out/read_cand/hw/RESULT_HEADWAY.json out/read_cand/q3/RESULT_Q3.json out/read_cand/hw/VERDICT_HEADWAY.md out/read_cand/q3/VERDICT_Q3.md podread_cand.out; do
  pod "cat $R/$f" > "$I/read_cand/pod/$(basename $f)"
  a=$(pod "sha256sum $R/$f | cut -c1-64"); b=$(sha256sum "$I/read_cand/pod/$(basename $f)" | cut -c1-64)
  [ "$a" = "$b" ] && echo "[$(ts)] pulled $(basename $f) sha256 ${a:0:16}" || echo "[$(ts)] PULL-MISMATCH $(basename $f)"
done
for f in CHECK_RULE_EVERY_RZGF.txt CHECK_ACC_GAIN_RZGF.txt; do
  pod "cat $W/cand_test/$f" > "$I/read_cand/pod/$f"
  a=$(pod "sha256sum $W/cand_test/$f | cut -c1-64"); b=$(sha256sum "$I/read_cand/pod/$f" | cut -c1-64)
  [ "$a" = "$b" ] && echo "[$(ts)] pulled $f sha256 ${a:0:16}" || echo "[$(ts)] PULL-MISMATCH $f"
done
echo "ZZ WAIT-PODREAD-CAND DONE $(ts)"

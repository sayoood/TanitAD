#!/usr/bin/env bash
# wait_podread_longroll.sh -- dev-box waiter of card PREREG_V8F_PROOF_ADD70 (LONGROLL10K), adapted from wait_podread_cyclebound.sh (card ADD69), bash only (no Python on the dev box).
# Waits for the pod marker LONGROLL10K_DONE.txt (or _STOP), checks that the read kit (read_longroll10k.py, LONGROLL10K_REUSED.json, strat_roll_read.py under $R/root/integ10k and podread_longroll.sh under $R) is on the pod and equal to the local files (the lead pushes it, as for the earlier cards),
# runs the pod read (podread_longroll.sh, pushed before) and brings the result files back as text into read_longroll/pod/ with a sha256 compare. The lead starts it with output to wait_podread_longroll.out.
# Last line: ZZ WAIT-PODREAD-LONGROLL DONE|STOP <why>. No rm, no mv, no kill.
H=/f/TanitAD-scratch/v8proof; I=$H/integ10k; P=/c/Users/Admin/v8pod/pssh.sh
W=/workspace/v8proof/integ10k; R=/workspace/v8proof/podread
ts() { date -u +%H:%M:%SZ; }
pod() { timeout "${PODT:-60}" bash "$P" "$1" 2>/dev/null | sed '/hf_/d' | tr -d '\r'; }
k=0
while :; do
  s=$(pod "ls $W/LONGROLL10K_DONE.txt $W/LONGROLL10K_STOP.txt 2>/dev/null | sed 's|.*/||' | tr '\n' ' '")
  case "$s" in *STOP*) echo "[$(ts)] queue STOP: $(pod "cat $W/LONGROLL10K_STOP.txt | cut -c1-200")"; echo "ZZ WAIT-PODREAD-LONGROLL STOP queue"; exit 3;; *DONE*) break;; esac
  k=$((k+1)); [ "$k" -gt 8640 ] && { echo "ZZ WAIT-PODREAD-LONGROLL STOP not ended after 72 h"; exit 3; }
  sleep 30
done
echo "[$(ts)] queue DONE on the pod; the kit is checked, then the pod read starts"
# ---- the read kit must be on the pod already (pushed by the lead through the channel of the earlier cards; this waiter only CHECKS it, sha256 against the local files, and stops if it differs)
for f in read_longroll10k.py LONGROLL10K_REUSED.json strat_roll_read.py; do
  want=$(sha256sum "$I/$f" | cut -c1-64)
  got=$(pod "sha256sum $R/root/integ10k/$f | cut -c1-64")
  [ "$got" = "$want" ] || { echo "[$(ts)] KIT-NOT-ON-POD-OR-DIFFERENT $f (local ${want:0:16}, pod ${got:0:16})"; echo "ZZ WAIT-PODREAD-LONGROLL STOP kit $f"; exit 3; }
  echo "[$(ts)] kit file $f on the pod, sha256 ${want:0:16}"
done
want=$(sha256sum "$I/podread_longroll.sh" | cut -c1-64); got=$(pod "sha256sum $R/podread_longroll.sh | cut -c1-64")
[ "$got" = "$want" ] || { echo "[$(ts)] KIT-NOT-ON-POD-OR-DIFFERENT podread_longroll.sh"; echo "ZZ WAIT-PODREAD-LONGROLL STOP kit podread_longroll.sh"; exit 3; }
PODT=3600 pod "cd $R && bash podread_longroll.sh > podread_longroll.out 2>&1; tail -n 240 podread_longroll.out | cut -c1-430"
mkdir -p "$I/read_longroll/pod"
for f in out/read_longroll/READ_LONGROLL10K.json out/e1lr/ENV_E1.json out/read_longroll/POD_HASHES.txt out/read_longroll/read_longroll10k.log podread_longroll.out; do
  pod "cat $R/$f" > "$I/read_longroll/pod/$(basename $f)"
  a=$(pod "sha256sum $R/$f | cut -c1-64"); b=$(sha256sum "$I/read_longroll/pod/$(basename $f)" | cut -c1-64)
  [ "$a" = "$b" ] && echo "[$(ts)] pulled $(basename $f) sha256 ${a:0:16}" || echo "[$(ts)] PULL-MISMATCH $(basename $f)"
done
for f in test/CHECK_PREROLL_B.txt; do
  pod "cat $W/longroll/$f" > "$I/read_longroll/pod/$(basename $f)"
  a=$(pod "sha256sum $W/longroll/$f | cut -c1-64"); b=$(sha256sum "$I/read_longroll/pod/$(basename $f)" | cut -c1-64)
  [ "$a" = "$b" ] && echo "[$(ts)] pulled $(basename $f) sha256 ${a:0:16}" || echo "[$(ts)] PULL-MISMATCH $(basename $f)"
done
echo "ZZ WAIT-PODREAD-LONGROLL DONE $(ts)"

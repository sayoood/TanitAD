#!/usr/bin/env bash
# chain_read.sh closrc|midlook -- dev-box chain of cards PREREG_V8F_PROOF_ADD75 (closrc) and ADD76 (midlook): waits for the end marker of the pod queue, runs the pod-side read once
# (under nohup on the pod), pulls the result files with a hash compare and prints which controls passed (handsrc_verdict.py prints no value of a card). Polls every 60 s.
# No rm, no mv, no kill. Output: integ10k/chain_<name>.out, last line ZZ CHAIN-<NAME> END.
set -u
NM=${1:-}
cd /f/TanitAD-scratch/v8proof || exit 2
PS=/c/Users/Admin/v8pod/pssh.sh
PYL=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
W=/workspace/v8proof/integ10k; P=/workspace/v8proof/podread
if [ "$NM" = closrc ]; then MK=CLOSRC10K; RD=read_closrc10k; JS=READ_CLOSRC10K.json; LG=read_closrc10k.log; SCR=podread_closrc10k.sh; QL=queue_closrc.log; SMD=small_cl; MAXW=60
  RUNS="I10RQ50KP30Hb20 I10RQ50cKP30Hb20 I10RQ50hKP30Hb20 I10RLbKP30Hb20 I10RLcKP30Hb20"
elif [ "$NM" = midlook ]; then MK=MIDLOOK10K; RD=read_midlook10k; JS=READ_MIDLOOK10K.json; LG=read_midlook10k.log; SCR=podread_midlook10k.sh; QL=queue_midlook.log; SMD=small_ml; MAXW=150
  RUNS="I10RQ50cKF5P30Hb20 I10RQ50hKF5P30Hb20 I10RQ50cKF5Hb20 I10RQ50hKF5Hb20"
else echo "usage: chain_read.sh closrc|midlook"; exit 2; fi
OUT=integ10k/chain_$NM.out
UP=$(echo "$NM" | tr 'a-z' 'A-Z')
ts() { date -u +%H:%M:%SZ; }
say() { echo "[$(ts)] $*" >> "$OUT"; }
pod() { timeout "$1" bash "$PS" "$2" 2>/dev/null | sed '/hf_/d' | tr -d '\r'; }
wait_marker() {
  local i r
  for i in $(seq 1 "$MAXW"); do
    r=$(pod 40 "cd $W; ls ${MK}_DONE.txt ${MK}_STOP.txt 2>/dev/null | tr '\n' ' '")
    case "$r" in *"${MK}_STOP"*) echo STOP; return 0;; *"${MK}_DONE"*) echo DONE; return 0;; esac
    sleep 60
  done
  echo TIMEOUT
}
pull() {
  local a b
  mkdir -p "$(dirname "$2")"
  timeout 240 bash "$PS" "base64 -w0 '$1'" 2>/dev/null | tr -d '\r\n' | base64 -d > "$2" 2>/dev/null
  a=$(sha256sum "$2" | cut -c1-64); b=$(pod 40 "sha256sum '$1' | cut -c1-64")
  if [ -n "$a" ] && [ "$a" = "$b" ]; then say "pulled $2 ${a:0:16}"; return 0; fi
  say "PULL-MISMATCH $2 local ${a:0:16} pod ${b:0:16}"; return 1
}
say "chain $NM start"
m=$(wait_marker); say "queue marker $MK: $m"
if [ "$m" != DONE ]; then
  [ "$m" = STOP ] && say "stop text: $(pod 40 "cut -c1-200 $W/${MK}_STOP.txt")"
  say "ZZ CHAIN-$UP END (no read)"; exit 0
fi
say "pod read start"
CL=$P/out/chain_$RD.log
pod 60 "[ -e $CL ] && echo EXISTS || { (nohup setsid bash $P/$SCR > $CL 2>&1 < /dev/null &); echo STARTED; }" >> "$OUT"
for i in $(seq 1 160); do
  n=$(pod 40 "grep -c '^ZZ PODREAD' $CL 2>/dev/null")
  [ "${n:-0}" -ge 1 ] 2>/dev/null && break
  sleep 30
done
pod 40 "tail -n 16 $CL | cut -c1-300" >> "$OUT"
L=integ10k/$RD/pod
okp=1
for f in "$JS" "$LG" READER_HASHES.txt selftest_pod.log; do pull "$P/out/$RD/$f" "$L/$f" || okp=0; done
pull "$W/$QL" "$L/$QL" || okp=0
for r in $RUNS; do for k in 0 1; do pull "$W/$SMD/run_${r}_s$k.txt" "$L/small/run_${r}_s$k.txt" || okp=0; done; done
say "pulls ok: $okp"
v=$(PYTHONDONTWRITEBYTECODE=1 "$PYL" integ10k/handsrc_verdict.py "$L/$JS"); rc=$?
say "$v"
say "ZZ CHAIN-$UP END"

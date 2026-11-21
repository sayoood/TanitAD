#!/usr/bin/env bash
# chain_handsrc.sh 30|60 -- dev-box chain of cards PREREG_V8F_PROOF_ADD73 (30) and ADD74 (60): waits for the end marker of the pod queue, runs the pod-side read once, pulls the result
# files with a hash compare, prints which controls passed (handsrc_verdict.py prints no value of a card). With 30 and every control of part 1 passed it then starts the queue of card
# ADD74 on the pod (the queue checks the same condition itself). Polls every 60 s. No rm, no mv, no kill. Output: integ10k/chain_handsrc<P>.out, last line ZZ CHAIN-HANDSRC<P> END.
set -u
PP=${1:-30}
cd /f/TanitAD-scratch/v8proof || exit 2
PS=/c/Users/Admin/v8pod/pssh.sh
PYL=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
OUT=integ10k/chain_handsrc$PP.out
W=/workspace/v8proof/integ10k; P=/workspace/v8proof/podread
if [ "$PP" = 30 ]; then MK=HANDSRC10K; RD=read_handsrc_decomp; JS=READ_HANDSRC_DECOMP.json; LG=read_handsrc_decomp.log; SCR=podread_handsrc_decomp.sh; QL=queue_handsrc.log; SMD=small_hs; MAXW=45
elif [ "$PP" = 60 ]; then MK=HANDSRC60; RD=read_handsrc60_decomp; JS=READ_HANDSRC60_DECOMP.json; LG=read_handsrc60_decomp.log; SCR=podread_handsrc60_decomp.sh; QL=queue_handsrc60.log; SMD=small_hs60; MAXW=90
else echo "usage: chain_handsrc.sh 30|60"; exit 2; fi
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
say "chain $PP start"
m=$(wait_marker); say "queue marker $MK: $m"
if [ "$m" != DONE ]; then
  [ "$m" = STOP ] && say "stop text: $(pod 40 "cut -c1-200 $W/${MK}_STOP.txt")"
  say "ZZ CHAIN-HANDSRC$PP END (no read)"; exit 0
fi
say "pod read start"
CL=$P/out/chain_$RD.log
pod 60 "[ -e $CL ] && echo EXISTS || { (nohup setsid bash $P/$SCR > $CL 2>&1 < /dev/null &); echo STARTED; }" >> "$OUT"
for i in $(seq 1 120); do
  n=$(pod 40 "grep -c '^ZZ PODREAD' $CL 2>/dev/null")
  [ "${n:-0}" -ge 1 ] 2>/dev/null && break
  sleep 30
done
pod 40 "tail -n 16 $CL | cut -c1-300" >> "$OUT"
L=integ10k/$RD/pod
okp=1
for f in "$JS" "$LG" READER_HASHES.txt; do pull "$P/out/$RD/$f" "$L/$f" || okp=0; done
[ "$PP" = 30 ] && { pull "$P/out/$RD/selftest_pod.log" "$L/selftest_pod.log" || okp=0; pull "$W/handsrc_test/CHECK_HANDOVER_SRC_chk1n1.txt" "integ10k/CHECK_HANDOVER_SRC_chk1n1.txt" || okp=0; }
pull "$W/$QL" "$L/$QL" || okp=0
for a in Lg Sl Pl; do for k in 0 1; do pull "$W/$SMD/preroll_log_I10R0KP${PP}${a}Hb20_s$k.json" "$L/small/preroll_log_I10R0KP${PP}${a}Hb20_s$k.json" || okp=0; done; done
say "pulls ok: $okp"
v=$(PYTHONDONTWRITEBYTECODE=1 "$PYL" integ10k/handsrc_verdict.py "$L/$JS"); rc=$?
say "$v"
if [ "$PP" = 30 ] && [ "$rc" = 0 ]; then
  say "start of the queue of card ADD74: $(pod 60 "cd $W && (CHECKTAG=chk1n1 nohup setsid bash build/pod_handsrc60_queue.sh > handsrc60.nohup 2>&1 < /dev/null &); sleep 4; ls START_HANDSRC60.txt 2>/dev/null; tail -n 1 queue_handsrc60.log | cut -c1-140" | tr '\n' ' ')"
elif [ "$PP" = 30 ]; then
  say "queue of card ADD74 NOT started: part 1 of the read of card ADD73 did not pass"
fi
say "ZZ CHAIN-HANDSRC$PP END"

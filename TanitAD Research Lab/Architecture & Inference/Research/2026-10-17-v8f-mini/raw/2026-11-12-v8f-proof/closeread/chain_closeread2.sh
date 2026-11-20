#!/usr/bin/env bash
# chain_closeread2.sh -- dev box: wait for the end of the CLOSEREAD2 run of card ADD78 on the pod, pull the small files with a hash compare, print which controls passed
# (controls_verdict.py prints no value of a read). Polls every 30 s. No delete, no kill.
set -u
cd /f/TanitAD-scratch/v8proof || exit 2
PS=/c/Users/Admin/v8pod/pssh.sh
PYL=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
W=/workspace/v8proof/closeread
L=closeread/pod2
OUT=closeread/chain_closeread2.out
ts() { date -u +%H:%M:%SZ; }
say() { echo "[$(ts)] $*" >> "$OUT"; }
pod() { timeout "$1" bash "$PS" "$2" 2>/dev/null | sed '/hf_/d' | tr -d '\r'; }
pull() {
  local a b
  mkdir -p "$(dirname "$2")"
  timeout 240 bash "$PS" "base64 -w0 '$1'" 2>/dev/null | tr -d '\r\n' | base64 -d > "$2" 2>/dev/null
  a=$(sha256sum "$2" | cut -c1-64); b=$(pod 40 "sha256sum '$1' | cut -c1-64")
  if [ -n "$a" ] && [ "$a" = "$b" ]; then say "pulled $2 ${a:0:16}"; return 0; fi
  say "PULL-MISMATCH $2 local ${a:0:16} pod ${b:0:16}"; return 1
}
say "chain closeread2 start"
m=TIMEOUT
for i in $(seq 1 240); do
  r=$(pod 40 "cd $W; ls CLOSEREAD2_DONE.txt CLOSEREAD2_STOP.txt 2>/dev/null | tr '\n' ' '")
  case "$r" in *CLOSEREAD2_STOP*) m=STOP; break;; *CLOSEREAD2_DONE*) m=DONE; break;; esac
  sleep 30
done
say "marker: $m"
[ "$m" = STOP ] && say "stop text: $(pod 40 "cut -c1-240 $W/CLOSEREAD2_STOP.txt")"
okp=1
for f in selftest2_pod.log fit_read2.log read2/CLOSEREAD_CONTROLS.json read2/CLOSEREAD_RESULT.json read2/CLOSEREAD_VERDICT.md read2/CLOSEREAD2_CARD.json; do
  pod 40 "[ -f $W/$f ] && echo yes" | grep -q yes && { pull "$W/$f" "$L/$(basename "$f")" || okp=0; }
done
say "pulls ok: $okp"
if [ -f "$L/CLOSEREAD_CONTROLS.json" ]; then
  say "$(PYTHONDONTWRITEBYTECODE=1 "$PYL" closeread/controls_verdict.py "$L/CLOSEREAD_CONTROLS.json" 2>&1 | cut -c1-900)"
else
  say "no controls file"
fi
say "ZZ CHAIN-CLOSEREAD2 END"

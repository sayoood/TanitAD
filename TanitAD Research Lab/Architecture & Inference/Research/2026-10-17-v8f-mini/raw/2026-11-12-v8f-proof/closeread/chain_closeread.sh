#!/usr/bin/env bash
# chain_closeread.sh -- dev box: wait for the end of the CLOSEREAD run of card ADD77 on the pod, pull the small files with a hash compare, print which controls passed
# (controls_verdict.py prints no value of a read). Polls every 60 s. No delete, no kill.
set -u
cd /f/TanitAD-scratch/v8proof || exit 2
PS=/c/Users/Admin/v8pod/pssh.sh
PYL=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
W=/workspace/v8proof/closeread
L=closeread/pod
OUT=closeread/chain_closeread.out
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
say "chain closeread start"
m=TIMEOUT
for i in $(seq 1 240); do
  r=$(pod 40 "cd $W; ls CLOSEREAD_DONE.txt CLOSEREAD_STOP.txt SELFTEST_FAIL.txt 2>/dev/null | tr '\n' ' '")
  case "$r" in *SELFTEST_FAIL*) m=SELFTEST_FAIL; break;; *CLOSEREAD_STOP*) m=STOP; break;; *CLOSEREAD_DONE*) m=DONE; break;; esac
  sleep 60
done
say "marker: $m"
[ "$m" = STOP ] && say "stop text: $(pod 40 "cut -c1-240 $W/CLOSEREAD_STOP.txt")"
okp=1
for f in selftest_pod.log queue_closeread.log dump_s0.log dump_s1.log episode_map.json; do
  pod 40 "[ -f $W/$f ] && echo yes" | grep -q yes && { pull "$W/$f" "$L/$f" || okp=0; }
done
for f in dump/s0/reader_s0_meta.json dump/s1/reader_s1_meta.json read/CLOSEREAD_CONTROLS.json read/CLOSEREAD_RESULT.json read/CLOSEREAD_VERDICT.md fit_read.log; do
  pod 40 "[ -f $W/$f ] && echo yes" | grep -q yes && { pull "$W/$f" "$L/$(basename "$f")" || okp=0; }
done
say "pulls ok: $okp"
if [ -f "$L/CLOSEREAD_CONTROLS.json" ]; then
  say "$(PYTHONDONTWRITEBYTECODE=1 "$PYL" closeread/controls_verdict.py "$L/CLOSEREAD_CONTROLS.json" 2>&1 | cut -c1-900)"
else
  say "no controls file"
fi
say "ZZ CHAIN-CLOSEREAD END"

#!/usr/bin/env bash
# wait_leadscr.sh : dev-box waiter (script polling, no model turns; bounded 24 h). Card PREREG_V8F_PROOF_ADD39.
# Waits for the pod chain leadscr_a (two arms LCS trained from the start) and the zeroed-layer dumps (runs/leadscr_z/LCSZ_DONE.txt),
# pulls the arms and rollouts that exist and runs the reads of section 3 of the card into leadscr/read. It starts nothing on the pod.
P=/c/Users/Admin/v8pod/pssh.sh
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
W=F:/TanitAD-scratch/v8proof
PULL=/c/Users/Admin/v8proof_c/pull
ROOTS="$W/hiergap/pull,C:/Users/Admin/v8proof_c/pull"
export PULL_ROOT=$PULL RUNS_ROOT=/c/Users/Admin/v8proof_c/runs
cd /f/TanitAD-scratch/v8proof || exit 1
ts() { date -u +%H:%M:%SZ; }
pod() { timeout "$1" bash "$P" "$2" 2>/dev/null | sed '/hf_/d' | tr -d '\r'; }
k=0
while :; do
  k=$((k+1))
  S=$(pod 60 "cd /workspace/v8proof; ls chains/leadscr_a/leadscr_a_DONE.txt chains/leadscr_a/leadscr_a_STOP.txt runs/leadscr_z/LCSZ_DONE.txt 2>/dev/null; grep -a 'ARM DONE' chains/leadscr_a/chain.log 2>/dev/null | tail -n 1 | cut -c1-80")
  echo "[$(ts)] $(echo "$S" | tr '\n' ';' | cut -c1-300)"
  a=0; z=0
  echo "$S" | grep -q "leadscr_a_DONE\|leadscr_a_STOP" && a=1
  echo "$S" | grep -q "LCSZ_DONE" && z=1
  [ $a = 1 ] && [ $z = 1 ] && break
  [ "$k" -gt 288 ] && { echo "ZZ LEADSCR TIME LIMIT (24 h): a=$a z=$z; reading what exists"; break; }
  sleep 300
done
pod 90 "cd /workspace/v8proof; grep -a 'ARM DONE\|STOP\|ZZ CHAIN' chains/leadscr_a/chain.log 2>/dev/null | cut -c1-160 | tail -n 6; cat chains/leadscr_a/leadscr_a_STOP.txt 2>/dev/null | cut -c1-200; grep -a 'layer before zeroing\|dump rc\|s4:\|lcsz end' runs/leadscr_z/lcsz.log 2>/dev/null | cut -c1-330; cat runs/leadscr_z/S4.txt 2>/dev/null | cut -c1-240"
pull() { # queue arm...
  local q=$1 a; shift
  for a in "$@"; do [ -f "$PULL/$a/PULLED.txt" ] || Q=$q bash spatial/q3/pull_spatial_t1.sh "$a" 2>&1 | tail -n 1 | cut -c1-200; done
}
pull leadscr_a LCS_s0 LCS_s1
pull leadscr_z LCSZ_s0 LCSZ_s1
have() { local a; for a in "$@"; do [ -f "$PULL/${a}_s0/PULLED.txt" ] && [ -f "$PULL/${a}_s1/PULLED.txt" ] || return 1; done; return 0; }
echo "[$(ts)] pulled: $(for a in LS LCS LCSZ LFT LFTRL; do have $a && printf '%s ' $a; done)"
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8
OUT=$W/leadscr/read; mkdir -p "$OUT"
hw() { # name pairs
  ( cd /f/TanitAD-scratch/v8proof/headway && "$PY" headway_read.py read --t1-root "$ROOTS" --pairs "$2" --seeds 0,1 --spread-arm G --spread-seeds 0,1,2,3,4,5 --out-dir "$OUT/hw_$1" > "$OUT/hw_$1.log" 2>&1 ); echo "[$(ts)] headway $1 rc $? : $(tail -n 1 "$OUT/hw_$1.log" | cut -c1-200)"; }
q3() { # name pairs
  ( cd /f/TanitAD-scratch/v8proof/spatial/q3 && "$PY" q3_read.py read --t1-root "$ROOTS" --pairs "$2" --seeds 0,1 --spread-arm G --spread-seeds 0,1,2,3,4,5 --same-sign --out-dir "$OUT/q3_$1" > "$OUT/q3_$1.log" 2>&1 ); echo "[$(ts)] families $1 rc $? : $(tail -n 1 "$OUT/q3_$1.log" | cut -c1-200)"; }
if have LS LCS LFT LFTRL; then
  hw scr "LCS:LS,LFTRL:LFT"; q3 scr "LCS:LS,LFTRL:LFT"
else
  echo "[$(ts)] LCS, LS, LFT or LFTRL not all pulled: no effect read"
fi
if have LCS LCSZ; then hw zero "LCS:LCSZ"; q3 zero "LCS:LCSZ"; else echo "[$(ts)] LCSZ not pulled: no use read"; fi
echo "ZZ LEADSCR READS DONE $(ts)"

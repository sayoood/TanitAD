#!/usr/bin/env bash
# wait_leadcond.sh : dev-box waiter (script polling, no model turns; bounded 24 h). Card PREREG_V8F_PROOF_ADD35 with note N1.
# Waits for the pod queues RTRULE_LCA (after chain leadcond_a), RTRULE_LCB (after chain leadcond_b) and the zeroed-layer dumps (runs/leadcond_z/LCZ_DONE.txt),
# pulls the arms and rollouts that exist and runs the reads of section 5 of the card into leadcond/read. It starts nothing on the pod.
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
  S=$(pod 60 "cd /workspace/v8proof; ls distrule/RTRULE_LCA_DONE.txt distrule/RTRULE_LCA_STOP.txt distrule/RTRULE_LCB_DONE.txt distrule/RTRULE_LCB_STOP.txt runs/leadcond_z/LCZ_DONE.txt runs/leadcond_z/LCB_NOT_STARTED.txt runs/leadcond_z/Q3B_DONE.txt runs/leadcond_z/Q3B_FAIL.txt chains/leadcond_a/leadcond_a_STOP.txt chains/leadcond_b/leadcond_b_STOP.txt 2>/dev/null | sed 's|.*/||' | tr '\n' ' '; echo; tail -n 1 chains/leadcond_a/chain.log 2>/dev/null | cut -c1-100; tail -n 1 chains/leadcond_b/chain.log 2>/dev/null | cut -c1-100")
  echo "[$(ts)] $(echo "$S" | tr '\n' ';' | cut -c1-420)"
  a=0; b=0; z=0
  echo "$S" | grep -q "RTRULE_LCA_DONE\|RTRULE_LCA_STOP\|leadcond_a_STOP" && a=1
  echo "$S" | grep -q "RTRULE_LCB_DONE\|RTRULE_LCB_STOP\|leadcond_b_STOP\|LCB_NOT_STARTED" && b=1
  echo "$S" | grep -q "LCZ_DONE" && z=1
  [ $a = 1 ] && [ $b = 1 ] && [ $z = 1 ] && break
  [ "$k" -gt 288 ] && { echo "ZZ LEADCOND TIME LIMIT (24 h): a=$a b=$b z=$z; reading what exists"; break; }
  sleep 300
done
pod 90 "cd /workspace/v8proof; for c in leadcond_a leadcond_b; do grep -a 'ARM DONE\|STOP\|ZZ CHAIN' chains/\$c/chain.log 2>/dev/null | cut -c1-140 | tail -n 8; done; tail -n 4 distrule/rtrule_LCA_queue.log 2>/dev/null | cut -c1-160; tail -n 6 distrule/rtrule_LCB_queue.log 2>/dev/null | cut -c1-160; cat runs/leadcond_z/Q3B_DONE.txt runs/leadcond_z/Q3B_FAIL.txt 2>/dev/null | head -n 6 | cut -c1-200; tail -n 12 runs/leadcond_z/lcz.log 2>/dev/null | cut -c1-330; tail -n 12 runs/leadcond_z/lcb_auto.log 2>/dev/null | cut -c1-330"
pull() { # queue arm...
  local q=$1 a; shift
  for a in "$@"; do [ -f "$PULL/$a/PULLED.txt" ] || Q=$q bash spatial/q3/pull_spatial_t1.sh "$a" 2>&1 | tail -n 1 | cut -c1-200; done
}
pull leadcond_a LFT_s0 LFT_s1 LFTb_s0 LFTb_s1 LC_s0 LC_s1
pull leadcond_b LO_s0 LO_s1 LCO_s0 LCO_s1
pull distrule LFTRL_s0 LFTRL_s1 LORL_s0 LORL_s1 LCORL_s0 LCORL_s1
pull leadcond_z LCZ_s0 LCZ_s1 LCOZ_s0 LCOZ_s1
have() { local a; for a in "$@"; do [ -f "$PULL/${a}_s0/PULLED.txt" ] && [ -f "$PULL/${a}_s1/PULLED.txt" ] || return 1; done; return 0; }
echo "[$(ts)] pulled: $(for a in LFT LFTb LC LO LCO LFTRL LORL LCORL LCZ LCOZ; do have $a && printf '%s ' $a; done)"
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8
OUT=$W/leadcond/read; mkdir -p "$OUT"
hw() { # name pairs
  ( cd /f/TanitAD-scratch/v8proof/headway && "$PY" headway_read.py read --t1-root "$ROOTS" --pairs "$2" --seeds 0,1 --spread-arm G --spread-seeds 0,1,2,3,4,5 --out-dir "$OUT/hw_$1" > "$OUT/hw_$1.log" 2>&1 ); echo "[$(ts)] headway $1 rc $? : $(tail -n 1 "$OUT/hw_$1.log" | cut -c1-200)"; }
q3() { # name pairs
  ( cd /f/TanitAD-scratch/v8proof/spatial/q3 && "$PY" q3_read.py read --t1-root "$ROOTS" --pairs "$2" --seeds 0,1 --spread-arm G --spread-seeds 0,1,2,3,4,5 --same-sign --out-dir "$OUT/q3_$1" > "$OUT/q3_$1.log" 2>&1 ); echo "[$(ts)] families $1 rc $? : $(tail -n 1 "$OUT/q3_$1.log" | cut -c1-200)"; }
if have LFT LFTb LC && ! have LO LCO; then
  echo "[$(ts)] chain B arms absent: reading the pairs of chain A only"
  hw lever "LC:LFT,LFTb:LFT"; q3 lever "LC:LFT,LFTb:LFT"
  have LFTRL && hw rule "LFTRL:LFT,LC:LFTRL"
elif have LFT LFTb LC LO LCO; then
  hw lever "LC:LFT,LO:LFT,LCO:LFT,LFTb:LFT,LCO:LO,LCO:LC"
  q3 lever "LC:LFT,LO:LFT,LCO:LFT,LFTb:LFT,LCO:LO"
  if have LFTRL LORL LCORL; then hw rule "LFTRL:LFT,LCORL:LCO,LORL:LO,LCO:LFTRL,LO:LFTRL,LC:LFTRL"; elif have LFTRL; then hw rule "LFTRL:LFT,LCO:LFTRL,LO:LFTRL,LC:LFTRL"; fi
else
  echo "[$(ts)] the arms of chain A are not all pulled: no read"
fi
if have LC LCZ; then
  if have LCO LCOZ; then hw zero "LC:LCZ,LCO:LCOZ"; q3 zero "LC:LCZ,LCO:LCOZ"; else hw zero "LC:LCZ"; q3 zero "LC:LCZ"; fi
fi
echo "ZZ LEADCOND READS DONE $(ts)"

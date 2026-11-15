#!/usr/bin/env bash
# wait_ruleownl1.sh : dev-box waiter (script polling, no model turns; bounded 16 h). Note PREREG_V8F_PROOF_ADD34_N1 (linear form of the own-plan term).
# Waits for the pod queue RTRULE_OWNL1 (after chain ruleown2_a), pulls GOL, GOL4 and their label-fed rule rollouts, reads into ruletrain3/read.
# It starts nothing on the pod.
P=/c/Users/Admin/v8pod/pssh.sh
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
W=F:/TanitAD-scratch/v8proof
PULL=/c/Users/Admin/v8proof_c/pull
ROOTS="$W/hiergap/pull,C:/Users/Admin/v8proof_c/pull"
export PULL_ROOT=$PULL RUNS_ROOT=/c/Users/Admin/v8proof_c/runs
cd /f/TanitAD-scratch/v8proof || exit 1
ts() { date -u +%H:%M:%SZ; }
pod() { timeout "$1" bash "$P" "$2" 2>/dev/null | sed '/hf_/d' | tr -d '\r'; }
TAG=OWNL1; CHAIN=ruleown2_a; ARMS="GOL_s0 GOL_s1 GOL4_s0 GOL4_s1"; RLS="GOLRL_s0 GOLRL_s1 GOL4RL_s0 GOL4RL_s1"
k=0
while :; do
  k=$((k+1))
  S=$(pod 60 "cd /workspace/v8proof; ls distrule/RTRULE_${TAG}_DONE.txt distrule/RTRULE_${TAG}_STOP.txt chains/$CHAIN/${CHAIN}_STOP.txt 2>/dev/null | tr '\n' ' '; echo; tail -n 1 chains/$CHAIN/chain.log 2>/dev/null | cut -c1-120; tail -n 1 distrule/rtrule_${TAG}_queue.log 2>/dev/null | cut -c1-120")
  echo "[$(ts)] $(echo "$S" | tr '\n' ';' | cut -c1-380)"
  echo "$S" | grep -q "RTRULE_${TAG}_DONE\|RTRULE_${TAG}_STOP" && break
  [ "$k" -gt 192 ] && { echo "ZZ RULEOWNL1 TIME LIMIT (16 h)"; exit 3; }
  sleep 300
done
pod 60 "cd /workspace/v8proof; grep -a 'ARM DONE\|STOP\|gate\|train ' chains/$CHAIN/chain.log | cut -c1-160 | tail -n 14; ls chains/$CHAIN | tr '\n' ' '; echo; cat distrule/rtrule_${TAG}_queue.log | cut -c1-200 | tail -n 7"
if echo "$S" | grep -q "${CHAIN}_STOP"; then echo "ZZ RULEOWNL1 CHAIN STOP MARKER: $(pod 60 "cat /workspace/v8proof/chains/$CHAIN/${CHAIN}_STOP.txt")"; fi
for a in $ARMS; do [ -f "$PULL/$a/PULLED.txt" ] || Q=$CHAIN bash spatial/q3/pull_spatial_t1.sh "$a" 2>&1 | tail -n 1; done
for a in $RLS; do [ -f "$PULL/$a/PULLED.txt" ] || Q=distrule bash spatial/q3/pull_spatial_t1.sh "$a" 2>&1 | tail -n 1; done
j=0
while :; do   # one reader at a time
  r=$(wmic process where "name='python.exe'" get CommandLine 2>/dev/null | grep -c 'headway_read.py\|q3_read.py')
  [ "${r:-0}" = 0 ] && break
  j=$((j+1)); [ "$j" -gt 60 ] && { echo "[$(ts)] other readers still running after 2 h; going on"; break; }
  sleep 120
done
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8
hw() { # out-dir name pairs
  ( cd /f/TanitAD-scratch/v8proof/headway && "$PY" headway_read.py read --t1-root "$ROOTS" --pairs "$3" --seeds 0,1 --spread-arm G --spread-seeds 0,1,2,3,4,5 --out-dir "$1/hw_$2" > "$1/hw_$2.log" 2>&1 ); echo "[$(ts)] headway $2 rc $? : $(tail -n 1 "$1/hw_$2.log" | cut -c1-200)"; }
q3() { # out-dir name pairs
  ( cd /f/TanitAD-scratch/v8proof/spatial/q3 && "$PY" q3_read.py read --t1-root "$ROOTS" --pairs "$3" --seeds 0,1 --spread-arm G --spread-seeds 0,1,2,3,4,5 --same-sign --out-dir "$1/q3_$2" > "$1/q3_$2.log" 2>&1 ); echo "[$(ts)] families $2 rc $? : $(tail -n 1 "$1/q3_$2.log" | cut -c1-200)"; }
OUT=$W/ruletrain3/read; mkdir -p "$OUT"
hw "$OUT" lever "GOL:GFT,GOL4:GFT,GFTb:GFT,GOL:GOW,GOL4:GOW4"
hw "$OUT" rule "GFTRL:GFT,GOLRL:GOL,GOL4RL:GOL4,GOL:GFTRL,GOL4:GFTRL,GOLRL:GFTRL,GOL4RL:GFTRL"
q3 "$OUT" lever "GOL:GFT,GOL4:GFT,GFTb:GFT,GOL:GOW"
echo "ZZ RULEOWNL1 READS DONE $(ts)"

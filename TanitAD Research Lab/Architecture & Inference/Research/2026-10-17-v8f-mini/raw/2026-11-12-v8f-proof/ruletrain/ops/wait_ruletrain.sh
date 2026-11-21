#!/usr/bin/env bash
# wait_ruletrain.sh : dev-box waiter for card PREREG_V8F_PROOF_ADD32 (RULETRAIN). Script polling, no model turns. Bounded: 16 h.
# Waits for the pod rollout queue of the card (RTRULE_DONE.txt or RTRULE_STOP.txt; it runs after the chain ruletrain_a), pulls the six chain dumps (Q=ruletrain_a)
# and the four label-fed rule rollouts (Q=distrule), waits for the other readers of this box to end (one reader at a time), then reads. It starts nothing on the pod.
P=/c/Users/Admin/v8pod/pssh.sh
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
W=F:/TanitAD-scratch/v8proof
PULL=/c/Users/Admin/v8proof_c/pull
ROOTS="$W/hiergap/pull,C:/Users/Admin/v8proof_c/pull"
TK=/c/Users/Admin/AppData/Local/Temp/claude/4ac3a056b343bb35/d1b9de00-7d76-4030-8bc5-507afd36195b/tasks
export PULL_ROOT=$PULL RUNS_ROOT=/c/Users/Admin/v8proof_c/runs
cd /f/TanitAD-scratch/v8proof || exit 1
ts() { date -u +%H:%M:%SZ; }
pod() { timeout "$1" bash "$P" "$2" 2>/dev/null | sed '/hf_/d' | tr -d '\r'; }
k=0
while :; do
  k=$((k+1))
  S=$(pod 60 "cd /workspace/v8proof; ls distrule/RTRULE_DONE.txt distrule/RTRULE_STOP.txt chains/ruletrain_a/ruletrain_a_STOP.txt 2>/dev/null | tr '\n' ' '; echo; tail -n 1 chains/ruletrain_a/chain.log 2>/dev/null | cut -c1-160; tail -n 1 distrule/rtrule_queue.log 2>/dev/null | cut -c1-140")
  echo "[$(ts)] $(echo "$S" | tr '\n' ';' | cut -c1-380)"
  echo "$S" | grep -q 'RTRULE_DONE\|RTRULE_STOP' && break
  [ "$k" -gt 192 ] && { echo "ZZ RULETRAIN TIME LIMIT (16 h)"; exit 3; }
  sleep 300
done
pod 60 "cd /workspace/v8proof; grep -a 'ARM DONE\|STOP\|gate\|train ' chains/ruletrain_a/chain.log | cut -c1-200 | tail -n 24; cat distrule/rtrule_queue.log | cut -c1-200 | tail -n 8; ls chains/ruletrain_a | tr '\n' ' '"
if echo "$S" | grep -q 'ruletrain_a_STOP'; then echo "ZZ RULETRAIN CHAIN STOP MARKER: $(pod 60 'cat /workspace/v8proof/chains/ruletrain_a/ruletrain_a_STOP.txt')"; fi
for a in GFT_s0 GRT_s0 GFT_s1 GRT_s1 GFTb_s0 GFTb_s1; do
  [ -f "$PULL/$a/PULLED.txt" ] || Q=ruletrain_a bash spatial/q3/pull_spatial_t1.sh "$a" 2>&1 | tail -n 1
done
for a in GFTRL_s0 GRTRL_s0 GFTRL_s1 GRTRL_s1; do
  [ -f "$PULL/$a/PULLED.txt" ] || Q=distrule bash spatial/q3/pull_spatial_t1.sh "$a" 2>&1 | tail -n 1
done
j=0
while :; do   # one reader at a time: wait until no headway / q3 reader of this fork is running
  n=$(ps -W 2>/dev/null | grep -c 'python.exe'); r=$(wmic process where "name='python.exe'" get CommandLine 2>/dev/null | grep -c 'headway_read.py\|q3_read.py')
  [ "${r:-0}" = 0 ] && break
  j=$((j+1)); [ "$j" -gt 60 ] && { echo "[$(ts)] other readers still running after 2 h; going on"; break; }
  sleep 120
done
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8
OUT=$W/ruletrain/read; mkdir -p "$OUT"
hw() { # name pairs seeds spread-arm
  ( cd /f/TanitAD-scratch/v8proof/headway && "$PY" headway_read.py read --t1-root "$ROOTS" --pairs "$2" --seeds "$3" --spread-arm "$4" --spread-seeds 0,1,2,3,4,5 --out-dir "$OUT/hw_$1" > "$OUT/hw_$1.log" 2>&1 ); echo "[$(ts)] headway $1 rc $? : $(tail -n 1 "$OUT/hw_$1.log" | cut -c1-200)"; }
q3() { # name pairs seeds spread-arm
  ( cd /f/TanitAD-scratch/v8proof/spatial/q3 && "$PY" q3_read.py read --t1-root "$ROOTS" --pairs "$2" --seeds "$3" --spread-arm "$4" --spread-seeds 0,1,2,3,4,5 --same-sign --out-dir "$OUT/q3_$1" > "$OUT/q3_$1.log" 2>&1 ); echo "[$(ts)] families $1 rc $? : $(tail -n 1 "$OUT/q3_$1.log" | cut -c1-200)"; }
hw lever "GRT:GFT,GFTb:GFT,GFT:G,GRT:G" 0,1 G
hw rule "GFTRL:GFT,GRTRL:GRT,GRT:GFTRL,GRTRL:GFTRL" 0,1 G
q3 lever "GRT:GFT,GFTb:GFT,GFT:G" 0,1 G
echo "ZZ RULETRAIN READS DONE $(ts)"

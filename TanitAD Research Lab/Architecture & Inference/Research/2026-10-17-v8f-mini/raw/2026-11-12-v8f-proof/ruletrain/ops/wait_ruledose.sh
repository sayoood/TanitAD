#!/usr/bin/env bash
# wait_ruledose.sh : dev-box waiter for note PREREG_V8F_PROOF_ADD32_N1, stage 1 (6,000-step fine-tunes). Script polling, no model turns. Bounded: 16 h.
# Waits for the pod rollout queue of the note (RTRULE6_DONE.txt or RTRULE6_STOP.txt; it runs after the chains ruledose_a and ruledose_b), pulls the six chain dumps
# and the four label-fed rule rollouts, waits for the other readers of this box to end (one reader at a time), then reads. It starts nothing on the pod.
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
  S=$(pod 60 "cd /workspace/v8proof; ls distrule/RTRULE6_DONE.txt distrule/RTRULE6_STOP.txt chains/ruledose_a/ruledose_a_STOP.txt chains/ruledose_b/ruledose_b_STOP.txt 2>/dev/null | tr '\n' ' '; echo; tail -n 1 chains/ruledose_a/chain.log 2>/dev/null | cut -c1-110; tail -n 1 chains/ruledose_b/chain.log 2>/dev/null | cut -c1-110; tail -n 1 distrule/rtrule6_queue.log 2>/dev/null | cut -c1-120")
  echo "[$(ts)] $(echo "$S" | tr '\n' ';' | cut -c1-420)"
  echo "$S" | grep -q 'RTRULE6_DONE\|RTRULE6_STOP' && break
  [ "$k" -gt 192 ] && { echo "ZZ RULEDOSE TIME LIMIT (16 h)"; exit 3; }
  sleep 300
done
pod 60 "cd /workspace/v8proof; for c in a b; do grep -a 'ARM DONE\|STOP\|gate\|train ' chains/ruledose_\$c/chain.log | cut -c1-160 | tail -n 10; ls chains/ruledose_\$c | tr '\n' ' '; echo; done; cat distrule/rtrule6_queue.log | cut -c1-200 | tail -n 7"
if echo "$S" | grep -q 'ruledose_[ab]_STOP'; then echo "ZZ RULEDOSE CHAIN STOP MARKER: $(pod 60 'cat /workspace/v8proof/chains/ruledose_a/ruledose_a_STOP.txt /workspace/v8proof/chains/ruledose_b/ruledose_b_STOP.txt 2>/dev/null')"; fi
for a in GFT6_s0 GRT6_s0 GFT6b_s0; do
  [ -f "$PULL/$a/PULLED.txt" ] || Q=ruledose_a bash spatial/q3/pull_spatial_t1.sh "$a" 2>&1 | tail -n 1
done
for a in GFT6_s1 GRT6_s1 GFT6b_s1; do
  [ -f "$PULL/$a/PULLED.txt" ] || Q=ruledose_b bash spatial/q3/pull_spatial_t1.sh "$a" 2>&1 | tail -n 1
done
for a in GFT6RL_s0 GRT6RL_s0 GFT6RL_s1 GRT6RL_s1; do
  [ -f "$PULL/$a/PULLED.txt" ] || Q=distrule bash spatial/q3/pull_spatial_t1.sh "$a" 2>&1 | tail -n 1
done
j=0
while :; do   # one reader at a time: wait until no headway / q3 reader of this fork is running
  r=$(wmic process where "name='python.exe'" get CommandLine 2>/dev/null | grep -c 'headway_read.py\|q3_read.py')
  [ "${r:-0}" = 0 ] && break
  j=$((j+1)); [ "$j" -gt 60 ] && { echo "[$(ts)] other readers still running after 2 h; going on"; break; }
  sleep 120
done
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8
OUT=$W/ruletrain/read6; mkdir -p "$OUT"
hw() { # name pairs seeds spread-arm
  ( cd /f/TanitAD-scratch/v8proof/headway && "$PY" headway_read.py read --t1-root "$ROOTS" --pairs "$2" --seeds "$3" --spread-arm "$4" --spread-seeds 0,1,2,3,4,5 --out-dir "$OUT/hw_$1" > "$OUT/hw_$1.log" 2>&1 ); echo "[$(ts)] headway $1 rc $? : $(tail -n 1 "$OUT/hw_$1.log" | cut -c1-200)"; }
q3() { # name pairs seeds spread-arm
  ( cd /f/TanitAD-scratch/v8proof/spatial/q3 && "$PY" q3_read.py read --t1-root "$ROOTS" --pairs "$2" --seeds "$3" --spread-arm "$4" --spread-seeds 0,1,2,3,4,5 --same-sign --out-dir "$OUT/q3_$1" > "$OUT/q3_$1.log" 2>&1 ); echo "[$(ts)] families $1 rc $? : $(tail -n 1 "$OUT/q3_$1.log" | cut -c1-200)"; }
hw lever "GRT6:GFT6,GFT6b:GFT6,GFT6:GFT,GRT6:GRT" 0,1 G
hw rule "GFT6RL:GFT6,GRT6RL:GRT6,GRT6:GFT6RL,GRT6RL:GFT6RL" 0,1 G
q3 lever "GRT6:GFT6,GFT6b:GFT6,GFT6:GFT" 0,1 G
echo "ZZ RULEDOSE READS DONE $(ts)"

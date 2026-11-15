#!/usr/bin/env bash
# wait_ruleown.sh <own|dose2> : dev-box waiter (script polling, no model turns; bounded 16 h).
#   own   : card PREREG_V8F_PROOF_ADD34. Waits for the pod queue RTRULE_OWN (after chain ruleown_a), pulls GOW, GOW4 and their label-fed rule rollouts, reads.
#   dose2 : stage 2 of note PREREG_V8F_PROOF_ADD32_N1. Waits for the pod queue RTRULE_DOSE2 (after chain ruledose2_a), pulls GRW6 and GRW6RL, reads against
#           the stage 1 arms (GFT6, GRT6, GFT6RL; pulled by wait_ruledose.sh).
# It starts nothing on the pod.
MODE=${1:?own or dose2}
P=/c/Users/Admin/v8pod/pssh.sh
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
W=F:/TanitAD-scratch/v8proof
PULL=/c/Users/Admin/v8proof_c/pull
ROOTS="$W/hiergap/pull,C:/Users/Admin/v8proof_c/pull"
export PULL_ROOT=$PULL RUNS_ROOT=/c/Users/Admin/v8proof_c/runs
cd /f/TanitAD-scratch/v8proof || exit 1
ts() { date -u +%H:%M:%SZ; }
pod() { timeout "$1" bash "$P" "$2" 2>/dev/null | sed '/hf_/d' | tr -d '\r'; }
if [ "$MODE" = own ]; then TAG=OWN; CHAIN=ruleown_a; ARMS="GOW_s0 GOW_s1 GOW4_s0 GOW4_s1"; RLS="GOWRL_s0 GOWRL_s1 GOW4RL_s0 GOW4RL_s1"
elif [ "$MODE" = dose2 ]; then TAG=DOSE2; CHAIN=ruledose2_a; ARMS="GRW6_s0 GRW6_s1"; RLS="GRW6RL_s0 GRW6RL_s1"
else echo "unknown mode $MODE"; exit 2; fi
k=0
while :; do
  k=$((k+1))
  S=$(pod 60 "cd /workspace/v8proof; ls distrule/RTRULE_${TAG}_DONE.txt distrule/RTRULE_${TAG}_STOP.txt chains/$CHAIN/${CHAIN}_STOP.txt 2>/dev/null | tr '\n' ' '; echo; tail -n 1 chains/$CHAIN/chain.log 2>/dev/null | cut -c1-120; tail -n 1 distrule/rtrule_${TAG}_queue.log 2>/dev/null | cut -c1-120")
  echo "[$(ts)] $(echo "$S" | tr '\n' ';' | cut -c1-380)"
  echo "$S" | grep -q "RTRULE_${TAG}_DONE\|RTRULE_${TAG}_STOP" && break
  [ "$k" -gt 192 ] && { echo "ZZ RULEOWN $MODE TIME LIMIT (16 h)"; exit 3; }
  sleep 300
done
pod 60 "cd /workspace/v8proof; grep -a 'ARM DONE\|STOP\|gate\|train ' chains/$CHAIN/chain.log | cut -c1-160 | tail -n 14; ls chains/$CHAIN | tr '\n' ' '; echo; cat distrule/rtrule_${TAG}_queue.log | cut -c1-200 | tail -n 7"
if echo "$S" | grep -q "${CHAIN}_STOP"; then echo "ZZ RULEOWN $MODE CHAIN STOP MARKER: $(pod 60 "cat /workspace/v8proof/chains/$CHAIN/${CHAIN}_STOP.txt")"; fi
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
if [ "$MODE" = own ]; then
  OUT=$W/ruletrain2/read; mkdir -p "$OUT"
  hw "$OUT" lever "GOW:GFT,GOW4:GFT,GFTb:GFT,GOW4:GOW"
  hw "$OUT" rule "GFTRL:GFT,GOWRL:GOW,GOW4RL:GOW4,GOW:GFTRL,GOW4:GFTRL,GOWRL:GFTRL,GOW4RL:GFTRL"
  q3 "$OUT" lever "GOW:GFT,GOW4:GFT,GFTb:GFT"
else
  OUT=$W/ruletrain2/read_dose2; mkdir -p "$OUT"
  hw "$OUT" lever "GRW6:GFT6,GRW6:GRT6,GFT6b:GFT6"
  hw "$OUT" rule "GFT6RL:GFT6,GRW6RL:GRW6,GRW6:GFT6RL,GRW6RL:GFT6RL"
  q3 "$OUT" lever "GRW6:GFT6,GRW6:GRT6"
fi
echo "ZZ RULEOWN $MODE READS DONE $(ts)"

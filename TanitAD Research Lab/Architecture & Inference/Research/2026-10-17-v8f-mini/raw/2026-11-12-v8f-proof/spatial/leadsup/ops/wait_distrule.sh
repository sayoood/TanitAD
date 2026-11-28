#!/usr/bin/env bash
# wait_distrule.sh : dev-box waiter for card PREREG_V8F_PROOF_ADD26 (note N1). Script polling, no model turns. Bounded: 12 h.
# Waits for the pod queue (DISTRULE_DONE.txt or DISTRULE_STOP.txt), pulls every rollout that has T1_DONE (Q=distrule), then runs the reads:
#   distance keeping (headway_read.py) and the four families (q3_read.py), rule against no rule of the same checkpoint, arms A (six seeds) and arms B (two seeds),
#   and the distance-keeping read of the six arms of card ADD24 against G (Q3 of card ADD25).
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
k=0
while :; do
  k=$((k+1))
  S=$(pod 60 "cd /workspace/v8proof/distrule; ls DISTRULE_DONE.txt DISTRULE_STOP.txt 2>/dev/null | tr '\n' ' '; echo; tail -n 1 queue.log | cut -c1-200; ls -d /workspace/v8proof/runs/distrule/t1/*/T1_DONE 2>/dev/null | wc -l")
  echo "[$(ts)] $(echo "$S" | tr '\n' ';' | cut -c1-300)"
  echo "$S" | grep -q 'DISTRULE_DONE\|DISTRULE_STOP' && break
  [ "$k" -gt 144 ] && { echo "ZZ DISTRULE TIME LIMIT (12 h)"; exit 3; }
  sleep 300
done
pod 60 "cat /workspace/v8proof/distrule/queue.log | cut -c1-220" | tail -n 40
if echo "$S" | grep -q 'DISTRULE_STOP'; then echo "ZZ DISTRULE STOP MARKER: $(pod 60 'cat /workspace/v8proof/distrule/DISTRULE_STOP.txt')"; fi
ARMS=$(pod 60 "ls -d /workspace/v8proof/runs/distrule/t1/*/T1_DONE 2>/dev/null | sed 's#/T1_DONE##; s#.*/##' | grep -v SMOKE | tr '\n' ' '")
echo "[$(ts)] rollouts with T1_DONE: $ARMS"
for a in $ARMS; do
  [ -f "$PULL/$a/PULLED.txt" ] || Q=distrule bash spatial/q3/pull_spatial_t1.sh "$a" 2>&1 | tail -n 1
done
have() { [ -f "$PULL/$1/PULLED.txt" ] || [ -f "/f/TanitAD-scratch/v8proof/hiergap/pull/$1/PULLED.txt" ]; }
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8
OUT=$W/distrule/read; mkdir -p "$OUT"
hw() { # name pairs seeds spread-arm
  ( cd /f/TanitAD-scratch/v8proof/headway && "$PY" headway_read.py read --t1-root "$ROOTS" --pairs "$2" --seeds "$3" --spread-arm "$4" --spread-seeds 0,1,2,3,4,5 --out-dir "$OUT/hw_$1" > "$OUT/hw_$1.log" 2>&1 ); echo "[$(ts)] headway $1 rc $? : $(tail -n 1 "$OUT/hw_$1.log" | cut -c1-200)"; }
q3() { # name pairs seeds spread-arm
  ( cd /f/TanitAD-scratch/v8proof/spatial/q3 && "$PY" q3_read.py read --t1-root "$ROOTS" --pairs "$2" --seeds "$3" --spread-arm "$4" --spread-seeds 0,1,2,3,4,5 --same-sign --out-dir "$OUT/q3_$1" > "$OUT/q3_$1.log" 2>&1 ); echo "[$(ts)] families $1 rc $? : $(tail -n 1 "$OUT/q3_$1.log" | cut -c1-200)"; }
# control r3 (push must be named worse), arms A, arms B, and Q3 of card ADD25
have HIERRP_s0 && hw r3push HIERRP:HIER 0 HIER
hw A HIERRL:HIER,GRL:G 0,1,2,3,4,5 HIER
q3 A HIERRL:HIER,GRL:G 0,1,2,3,4,5 HIER
hw B LSRH:LS,LSRL:LS,LSHRH:LSH,LSHRL:LSH,LSDRH:LSD,LSDRL:LSD,LSRH:LSRL,LSHRH:LSHRL,LSDRH:LSDRL 0,1 G
q3 B LSRH:LS,LSRL:LS,LSHRH:LSH,LSHRL:LSH,LSDRH:LSD,LSDRL:LSD 0,1 G
hw ADD25Q3 LS:G,LSH:G,LSD:G 0,1 G
echo "ZZ DISTRULE READS DONE $(ts)"

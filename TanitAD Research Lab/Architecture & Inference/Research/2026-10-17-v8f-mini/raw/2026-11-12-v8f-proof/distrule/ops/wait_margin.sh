#!/usr/bin/env bash
# wait_margin.sh : dev-box waiter for card PREREG_V8F_PROOF_ADD30 (MARGIN). Script polling, no model turns. Bounded: 12 h.
# Waits for the pod queue (MARGIN_DONE.txt or MARGIN_STOP.txt), pulls the rollouts (Q=distrule), then runs the reads:
#   distance keeping (headway_read.py): each arm minus no rule, minus the head-fed rule of card ADD26, minus the label-fed rule; the four families (q3_read.py) against no rule.
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
  S=$(pod 60 "cd /workspace/v8proof/distrule; ls MARGIN_DONE.txt MARGIN_STOP.txt 2>/dev/null | tr '\n' ' '; echo; tail -n 1 margin_queue.log | cut -c1-200")
  echo "[$(ts)] $(echo "$S" | tr '\n' ';' | cut -c1-300)"
  echo "$S" | grep -q 'MARGIN_DONE\|MARGIN_STOP' && break
  [ "$k" -gt 144 ] && { echo "ZZ MARGIN TIME LIMIT (12 h)"; exit 3; }
  sleep 300
done
pod 60 "cat /workspace/v8proof/distrule/margin_queue.log | cut -c1-220" | tail -n 34
if echo "$S" | grep -q 'MARGIN_STOP'; then echo "ZZ MARGIN STOP MARKER: $(pod 60 'cat /workspace/v8proof/distrule/MARGIN_STOP.txt')"; fi
ARMS=$(pod 60 "ls -d /workspace/v8proof/runs/distrule/t1/*M[CTGLHR]_s[01]/T1_DONE 2>/dev/null | sed 's#/T1_DONE##; s#.*/##' | tr '\n' ' '")
echo "[$(ts)] rollouts with T1_DONE: $ARMS"
for a in $ARMS; do
  [ -f "$PULL/$a/PULLED.txt" ] || Q=distrule bash spatial/q3/pull_spatial_t1.sh "$a" 2>&1 | tail -n 1
done
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8
OUT=$W/distrule/read_margin; mkdir -p "$OUT"
hw() { # name pairs seeds spread-arm
  ( cd /f/TanitAD-scratch/v8proof/headway && "$PY" headway_read.py read --t1-root "$ROOTS" --pairs "$2" --seeds "$3" --spread-arm "$4" --spread-seeds 0,1,2,3,4,5 --out-dir "$OUT/hw_$1" > "$OUT/hw_$1.log" 2>&1 ); echo "[$(ts)] headway $1 rc $? : $(tail -n 1 "$OUT/hw_$1.log" | cut -c1-200)"; }
q3() { # name pairs seeds spread-arm
  ( cd /f/TanitAD-scratch/v8proof/spatial/q3 && "$PY" q3_read.py read --t1-root "$ROOTS" --pairs "$2" --seeds "$3" --spread-arm "$4" --spread-seeds 0,1,2,3,4,5 --same-sign --out-dir "$OUT/q3_$1" > "$OUT/q3_$1.log" 2>&1 ); echo "[$(ts)] families $1 rc $? : $(tail -n 1 "$OUT/q3_$1.log" | cut -c1-200)"; }
pairs() { # reference suffix ("" = no rule, RH, RL) -> the pair list of the five codes on both arm types
  local r=$1 o="" b c
  for b in LS LSD; do for c in MC MT MG ML MH; do o="$o,${b}${c}:${b}${r}"; done; done
  echo "${o#,}"
}
hw vs_none "$(pairs "")" 0,1 G
hw vs_head "$(pairs RH),LSMR:LSRH" 0,1 G
hw vs_label "$(pairs RL)" 0,1 G
q3 vs_none "$(pairs "")" 0,1 G
echo "ZZ MARGIN READS DONE $(ts)"

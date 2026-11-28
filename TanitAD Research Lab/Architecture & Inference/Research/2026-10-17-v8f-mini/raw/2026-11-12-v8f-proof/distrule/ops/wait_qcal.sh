#!/usr/bin/env bash
# wait_qcal.sh : dev-box waiter for card PREREG_V8F_PROOF_ADD31 (QCAL). Script polling, no model turns. Bounded: 14 h.
# Waits for the pod queue (QCAL_DONE.txt or QCAL_STOP.txt) AND for the reads of card ADD30 on this box to end (one reader at a time), pulls the rollouts,
# then reads QC against no rule, the head-fed rule, the label-fed rule and the fixed-margin arm MC. It starts nothing on the pod.
P=/c/Users/Admin/v8pod/pssh.sh
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
W=F:/TanitAD-scratch/v8proof
PULL=/c/Users/Admin/v8proof_c/pull
ROOTS="$W/hiergap/pull,C:/Users/Admin/v8proof_c/pull"
MT=/c/Users/Admin/AppData/Local/Temp/claude/4ac3a056b343bb35/d1b9de00-7d76-4030-8bc5-507afd36195b/tasks/b6a80lwup.output
export PULL_ROOT=$PULL RUNS_ROOT=/c/Users/Admin/v8proof_c/runs
cd /f/TanitAD-scratch/v8proof || exit 1
ts() { date -u +%H:%M:%SZ; }
pod() { timeout "$1" bash "$P" "$2" 2>/dev/null | sed '/hf_/d' | tr -d '\r'; }
k=0
while :; do
  k=$((k+1))
  S=$(pod 60 "cd /workspace/v8proof/distrule; ls QCAL_DONE.txt QCAL_STOP.txt 2>/dev/null | tr '\n' ' '; echo; tail -n 1 qcal_queue.log | cut -c1-200")
  echo "[$(ts)] $(echo "$S" | tr '\n' ';' | cut -c1-300)"
  echo "$S" | grep -q 'QCAL_DONE\|QCAL_STOP' && break
  [ "$k" -gt 168 ] && { echo "ZZ QCAL TIME LIMIT (14 h)"; exit 3; }
  sleep 300
done
pod 60 "cat /workspace/v8proof/distrule/qcal_queue.log | cut -c1-260" | tail -n 12
if echo "$S" | grep -q 'QCAL_STOP'; then echo "ZZ QCAL STOP MARKER: $(pod 60 'cat /workspace/v8proof/distrule/QCAL_STOP.txt')"; fi
j=0
while ! grep -q 'ZZ MARGIN READS DONE\|ZZ MARGIN TIME LIMIT' "$MT" 2>/dev/null; do
  j=$((j+1)); [ "$j" -gt 72 ] && { echo "[$(ts)] the reads of card ADD30 did not end in 3 h; going on"; break; }
  sleep 150
done
ARMS=$(pod 60 "ls -d /workspace/v8proof/runs/distrule/t1/*QC_s[01]/T1_DONE 2>/dev/null | sed 's#/T1_DONE##; s#.*/##' | tr '\n' ' '")
echo "[$(ts)] rollouts with T1_DONE: $ARMS"
for a in $ARMS LSMC_s0 LSMC_s1 LSDMC_s0 LSDMC_s1; do
  [ -f "$PULL/$a/PULLED.txt" ] || Q=distrule bash spatial/q3/pull_spatial_t1.sh "$a" 2>&1 | tail -n 1
done
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8
OUT=$W/distrule/read_qcal; mkdir -p "$OUT"
hw() { # name pairs seeds spread-arm
  ( cd /f/TanitAD-scratch/v8proof/headway && "$PY" headway_read.py read --t1-root "$ROOTS" --pairs "$2" --seeds "$3" --spread-arm "$4" --spread-seeds 0,1,2,3,4,5 --out-dir "$OUT/hw_$1" > "$OUT/hw_$1.log" 2>&1 ); echo "[$(ts)] headway $1 rc $? : $(tail -n 1 "$OUT/hw_$1.log" | cut -c1-200)"; }
q3() { # name pairs seeds spread-arm
  ( cd /f/TanitAD-scratch/v8proof/spatial/q3 && "$PY" q3_read.py read --t1-root "$ROOTS" --pairs "$2" --seeds "$3" --spread-arm "$4" --spread-seeds 0,1,2,3,4,5 --same-sign --out-dir "$OUT/q3_$1" > "$OUT/q3_$1.log" 2>&1 ); echo "[$(ts)] families $1 rc $? : $(tail -n 1 "$OUT/q3_$1.log" | cut -c1-200)"; }
hw vs_all "LSQC:LS,LSQC:LSRH,LSQC:LSRL,LSQC:LSMC,LSDQC:LSD,LSDQC:LSDRH,LSDQC:LSDRL,LSDQC:LSDMC" 0,1 G
q3 vs_none "LSQC:LS,LSDQC:LSD" 0,1 G
q3 vs_margin "LSQC:LSMC,LSDQC:LSDMC" 0,1 G
echo "ZZ QCAL READS DONE $(ts)"

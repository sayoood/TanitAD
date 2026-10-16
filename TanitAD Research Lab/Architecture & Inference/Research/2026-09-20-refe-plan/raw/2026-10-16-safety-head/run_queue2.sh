#!/bin/bash
# SAFEHEAD addendum A1 queue (PREREG_SAFEHEAD_A1_TTCPEN.md): waits for run_queue.sh's ZZSH_QUEUE_DONE, then FULL + TTC-penalised training reward, P K=50 then T K=250.  One run at a time.
export PYTHONUTF8=1 TMP=C:/Users/Admin/AppData/Local/Temp TEMP=C:/Users/Admin/AppData/Local/Temp SH_TTC_PEN=0.25
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
cd "$(dirname "$0")"
Q=logs/queue.txt
echo "$(date -u +%FT%TZ) queue2 (A1 TTCPEN) armed pid $$" >> $Q
while ! grep -q "ZZSH_QUEUE_DONE" $Q 2>/dev/null; do sleep 60; done
for spec in "P 50 25 50" "T 250 50 100 250"; do
  set -- $spec; rg=$1; K=$2; shift 2
  t0=$(date +%s)
  $PY sh_ttcpen.py --design FULL --lam 1.0 --regime $rg --K $K --snaps "$@" --out-dir out --threads 6 --tag _ttcpen0.25 > "logs/sh_${rg}_FULL_ttcpen0.25_K${K}.log" 2>&1
  echo "$(date -u +%FT%TZ) $rg FULL_TTCPEN0.25 K$K rc $? $(( $(date +%s) - t0 ))s" >> $Q
done
echo "$(date -u +%FT%TZ) ZZSH_QUEUE2_DONE" >> $Q

#!/bin/bash
# nt_e1_supervise.sh -- keep nt_e1_run.py alive until DONE_A1.json exists (resumable by design). At most MAX_RESTARTS restarts;
# after that it stops and PROGRESS.json keeps the last CRASHED status + traceback.  No flock (no lock fd to leak to children).
# Launch:  setsid nohup bash /home/nvidia/r8ns/nt/code/nt_e1_supervise.sh > /home/nvidia/r8ns/nt/logs/e1_sup.out 2>&1 < /dev/null &
set -u
NT=/home/nvidia/r8ns/nt; MAX_RESTARTS=${MAX_RESTARTS:-6}; n=0
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONIOENCODING=utf-8
echo "ZZSUPSTART $(date -Is) pid $$" >> $NT/logs/e1_sup.log
while [ ! -f $NT/DONE_A1.json ]; do
  nice -n 19 ionice -c3 /home/nvidia/venvs/tanitad-train/bin/python $NT/code/nt_e1_run.py --threads 3 >> $NT/logs/e1_run.log 2>&1
  rc=$?
  echo "ZZSUPEXIT rc=$rc n=$n $(date -Is)" >> $NT/logs/e1_sup.log
  [ -f $NT/DONE_A1.json ] && break
  n=$((n+1)); [ $n -gt $MAX_RESTARTS ] && { echo "ZZSUPGIVEUP $(date -Is)" >> $NT/logs/e1_sup.log; exit 1; }
  sleep 60
done
echo "ZZSUPEND $(date -Is)" >> $NT/logs/e1_sup.log

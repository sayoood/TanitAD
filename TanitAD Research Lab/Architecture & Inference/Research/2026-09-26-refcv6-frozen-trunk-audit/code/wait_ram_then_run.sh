#!/usr/bin/env bash
# wait (<= MAXW s) until available RAM >= NEED GB on two consecutive polls, then run the job
NEED=${NEED:-10.8}; MAXW=${MAXW:-7200}; JOB="$1"; OUT="$2"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
t=0; ok=0
while [ $t -lt $MAXW ]; do
  a=$($PY -c "import psutil;print('%.2f'%(psutil.virtual_memory().available/2**30))")
  if $PY -c "import sys;sys.exit(0 if float('$a')>=float('$NEED') else 1)"; then ok=$((ok+1)); else ok=0; fi
  if [ $ok -ge 2 ]; then echo "RAM_OK avail=$a after ${t}s -> running $JOB"; OMP_NUM_THREADS=4 $PY "$JOB" > "$OUT" 2>&1; rc=$?; echo "JOB_EXIT=$rc"; tail -3 "$OUT"; exit 0; fi
  sleep 30; t=$((t+30))
done
echo "RAM_TIMEOUT last_avail=$a"

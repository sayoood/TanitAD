#!/bin/sh
# GT-validation chain under the coordinator's rule FOR THIS RENDER (A6 suspended): START at >= 5.5 GB free on 3 consecutive
# samples 30 s apart; the watchdog aborts every job of the chain if box-free RAM drops below 4.0 GB.
# Steps: (1) extract the eval-only join subsets (2-D and 3-D in parallel), (2) the GT tool on them.
# Retries a step after an abort, up to 6 attempts. Progress goes to chain.log.
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
W=/c/Users/Admin/qland/work/gtval
PKG=D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-26-refcv6-gt-validation
LOG=$W/chain.log
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
say() { echo "$(date +%H:%M:%S) $*" >> $LOG; }
wait_floor() {
  n=0
  while [ $n -lt 3 ]; do
    if $PY -c "import psutil,sys; sys.exit(0 if psutil.virtual_memory().available/2**30 >= 5.5 else 1)"; then
      n=$((n + 1))
    else
      n=0
    fi
    [ $n -lt 3 ] && sleep 30
  done
  say "floor held: 3 consecutive samples >= 5.5 GB"
}
rm -f $W/WATCHDOG_STOP $W/ABORTED
$PY $W/ram_watchdog.py &
WD=$!
say "chain start; watchdog pid $WD"
attempt=0
while [ $attempt -lt 6 ]; do
  attempt=$((attempt + 1))
  rm -f $W/ABORTED
  wait_floor
  if [ ! -f $W/EXTRACT_OK ]; then
    say "attempt $attempt: extract 2d + 3d"
    $PY $W/extract_eval_joins.py 2d > $W/extract_2d.log 2>&1 &
    P1=$!
    $PY $W/extract_eval_joins.py 3d > $W/extract_3d.log 2>&1 &
    P2=$!
    wait $P1; E1=$?
    wait $P2; E2=$?
    say "extract exit 2d=$E1 3d=$E2"
    if [ $E1 -eq 0 ] && [ $E2 -eq 0 ] && [ ! -f $W/ABORTED ]; then touch $W/EXTRACT_OK; else continue; fi
  fi
  say "attempt $attempt: GT tool"
  cd /d/Projects/TanitAD || exit 1
  $PY taniteval/tools/render_refcv6_gt_validation.py --out-dir $PKG/raw \
    --agent-join-subset C:/Users/Admin/qland/work/gtval/eval_agents.jsonl \
    --join3d-subset C:/Users/Admin/qland/work/gtval/eval_agents_3d.jsonl \
    --extract-record "C:/Users/Admin/qland/work/gtval/extract_record_2d.json;C:/Users/Admin/qland/work/gtval/extract_record_3d.json" \
    --start-ram-gb 5.5 \
    >> $W/gtval.log 2>&1
  E=$?
  echo "EXIT=$E" >> $W/gtval.log
  say "GT tool exit $E"
  if [ $E -eq 0 ] && [ ! -f $W/ABORTED ]; then touch $W/GT_OK; break; fi
done
touch $W/WATCHDOG_STOP
wait $WD
say "chain end (GT_OK=$([ -f $W/GT_OK ] && echo yes || echo no))"
echo CHAIN_END >> $W/gtval.log

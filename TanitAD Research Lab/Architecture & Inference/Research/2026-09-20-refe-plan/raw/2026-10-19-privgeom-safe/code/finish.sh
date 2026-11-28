#!/bin/bash
# PRIVGEOM-SAFE finish watcher: waits for the extraction (41 logs, no locks), then join -> gate -> fit (2 shards) -> analyze.  Markers: run/FINISH_DONE.txt | run/FINISH_FAILED.txt.  Poll the markers, not this script.
R=/c/Users/Admin/par18/privgeom
PY=/c/Users/Admin/venvs/driverl-eval/Scripts/python.exe
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
LOG=$R/run/finish.log
log(){ echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }
fail(){ log "FAILED: $*"; echo "$*" > "$R/run/FINISH_FAILED.txt"; exit 1; }
N=41
last=""; stale=0
log "finish watcher started (pid $$)"
while true; do
  n=$(ls "$R"/run/scalars/*.npz 2>/dev/null | grep -vc '\.tmp\.')
  l=$(ls "$R"/run/scalars/*.lock 2>/dev/null | wc -l)
  f=$(ls "$R"/run/scalars/*.failed 2>/dev/null | wc -l)
  w=$(ls "$R"/run/scalars/WORKER_DONE_*.txt 2>/dev/null | wc -l)
  [ "$f" -gt 0 ] && fail "extraction failed on $f log(s): see run/scalars/*.failed"
  if [ "$n" -ge "$N" ] && [ "$l" -eq 0 ]; then break; fi
  if [ "$w" -ge 2 ] && [ "$l" -eq 0 ] && [ "$n" -lt "$N" ]; then fail "both workers finished but only $n of $N logs exist"; fi
  cur="$n/$l"
  if [ "$cur" = "$last" ]; then stale=$((stale+1)); else stale=0; last="$cur"; fi
  [ "$stale" -ge 60 ] && fail "no extraction progress for 60 min ($cur)"
  sleep 60
done
log "extraction complete: $n logs"
"$PY" -B -W ignore "$R/code/pg_run.py" join >> "$LOG" 2>&1 || fail "join"
"$PY" -B -W ignore "$R/code/pg_run.py" gate >> "$LOG" 2>&1 || fail "gate (see run/gate_report.json)"
"$PY" -B -W ignore "$R/code/pg_run.py" fit --shard 0 --nshards 2 > "$R/run/fit0.log" 2>&1 &
p0=$!
"$PY" -B -W ignore "$R/code/pg_run.py" fit --shard 1 --nshards 2 > "$R/run/fit1.log" 2>&1 &
p1=$!
wait $p0 || fail "fit shard 0 (see run/fit0.log)"
wait $p1 || fail "fit shard 1 (see run/fit1.log)"
"$PY" -B -W ignore "$R/code/pg_run.py" analyze >> "$LOG" 2>&1 || fail "analyze"
date +%FT%T > "$R/run/FINISH_DONE.txt"
log "FINISH_DONE"

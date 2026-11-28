#!/bin/bash
# PRIVGEOM-CLEAN SEEN REHEARSAL chain (NOT THE READ; banked SEEN arrays + SEEN labels; produces nothing about the 76): gate -> fit (2 shards) -> analyze, all through the SAME pgc_run.py the read uses.
# Markers: run_rehearsal/REHEARSAL_DONE.txt | run_rehearsal/REHEARSAL_FAILED.txt
R=/c/Users/Admin/par18/pgclean
PY=/c/Users/Admin/venvs/driverl-eval/Scripts/python.exe
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
export PGC_REHEARSAL=1 PGC_RUN="C:/Users/Admin/par18/pgclean/run_rehearsal" PGC_JOIN_DIR="C:/Users/Admin/par18/privgeom/run/join"
RR=$R/run_rehearsal
LOG=$RR/chain.log
log(){ echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }
fail(){ log "FAILED: $*"; echo "$*" > "$RR/REHEARSAL_FAILED.txt"; exit 1; }
rm -f "$RR/REHEARSAL_DONE.txt" "$RR/REHEARSAL_FAILED.txt"
log "rehearsal chain started (pid $$)"
"$PY" -B -W ignore "$R/code/pgc_run.py" gate >> "$RR/gate.log" 2>&1 || fail "gate (see run_rehearsal/gate_report.json)"
"$PY" -B -W ignore "$R/code/pgc_run.py" fit --shard 0 --nshards 2 >> "$RR/fit.log" 2>&1 &
p0=$!
"$PY" -B -W ignore "$R/code/pgc_run.py" fit --shard 1 --nshards 2 >> "$RR/fit2.log" 2>&1 &
p1=$!
wait $p0 || fail "fit shard 0"
wait $p1 || fail "fit shard 1"
"$PY" -B -W ignore "$R/code/pgc_run.py" analyze >> "$RR/chain.log" 2>&1 || fail "analyze"
date +%FT%T > "$RR/REHEARSAL_DONE.txt"
log "REHEARSAL_DONE"

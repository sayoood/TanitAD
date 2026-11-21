#!/bin/bash
# Supervisor for the corpus SAM3 map production on Thor (2026-09-15).
# - runs sam3map_prod.py in launches of BATCH clips: a fresh process (and CUDA context) every ~10 h, resumed from the ledger
# - keeps corpus_feeder.py alive (restarted when its pid is gone)
# - a launch printing ZZPROD-NOTHING-TO-DO ends the run: DONE is written and the feeder exits on it
# - crash-loop breaker: 3 launches in a row without ledger growth put the head todo clip into out/skip.txt
# - STOP file = clean off-switch, read between launches; the supervisor never kills the driver
# - lock on fd 200; every long-lived child gets 200>&- so no child can hold the lock after the supervisor is gone
C=/home/nvidia/sam3map/corpus; SM=/home/nvidia/sam3map; PY=/home/nvidia/venvs/tanitad-edge/bin/python
BATCH=${BATCH:-250}
exec 200>"$C/supervisor.lock"
flock -n 200 || { echo "$(date -Is) another supervisor holds $C/supervisor.lock - exiting"; exit 1; }
cd "$SM/eval" || exit 1
log() { echo "$(date -Is) $*" >> "$C/supervisor.log"; }
rows() { if [ -f "$C/out/manifest.jsonl" ]; then wc -l < "$C/out/manifest.jsonl"; else echo 0; fi; }
feeder_alive() {
  [ -f "$C/feeder.pid" ] || return 1
  p=$(cat "$C/feeder.pid"); [ -r "/proc/$p/cmdline" ] || return 1
  tr '\0' ' ' < "/proc/$p/cmdline" | grep -q corpus_feeder
}
start_feeder() {
  setsid nohup env PROD_MAX_FAILS=2 FEED_WINDOW=120 "$PY" "$SM/eval/corpus_feeder.py" >> "$C/feeder.out" 2>&1 < /dev/null 200>&- &
  echo $! > "$C/feeder.pid"; log "feeder started pid $!"
}
mkdir -p "$C/launch_logs"
n=$(ls "$C/launch_logs" | wc -l); noprog=0; startfail=0; skips=0
log "supervisor start pid $$ batch $BATCH"
while true; do
  if [ -f "$C/STOP" ]; then log "STOP present - exiting"; break; fi
  feeder_alive || start_feeder
  n=$((n + 1)); r0=$(rows); L="$C/launch_logs/launch_$(printf %04d $n).out"
  log "launch $n (ledger rows $r0)"
  env PYTHONPATH=/home/nvidia/sam3vendor:/home/nvidia/sam3paint:/home/nvidia/sam3map HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 \
      PROD_SCRATCH="$C/scratch" SRC_CAM="$C/cam" SRC_CAL="$C/calib" PROD_WAIT_S=7200 PROD_MAX_FAILS=2 PROD_KEEP_FAILED_WORK=20 \
      "$PY" sam3map_prod.py "$C/production_order.txt" "$C/out" "$BATCH" > "$L" 2>&1 < /dev/null 200>&-
  rc=$?; r1=$(rows)
  log "launch $n exit $rc (ledger rows $r1)"
  if grep -q '^ZZPROD-NOTHING-TO-DO' "$L"; then date -Is > "$C/DONE"; log "nothing left to do - DONE written"; break; fi
  if ! grep -q '^model ready' "$L"; then                       # died before any clip: not a clip's fault, never skip on it
    startfail=$((startfail + 1)); log "launch $n died before the model was ready (#$startfail)"
    if [ "$startfail" -ge 5 ]; then log "5 start failures - supervisor exits, needs attention"; break; fi
    sleep 300 200>&-; continue
  fi
  startfail=0
  if [ "$r1" -gt "$r0" ]; then noprog=0; else noprog=$((noprog + 1)); fi
  if [ "$noprog" -ge 3 ] && [ -s "$C/out/todo_head.txt" ]; then
    h=$(head -1 "$C/out/todo_head.txt"); echo "$h" >> "$C/out/skip.txt"; skips=$((skips + 1))
    log "3 launches without ledger growth - head clip $h added to skip.txt (#$skips)"; noprog=0
    if [ "$skips" -ge 10 ]; then log "10 crash-loop skips - supervisor exits, needs attention"; break; fi
  fi
  sleep 60 200>&-
done
log "supervisor exit"

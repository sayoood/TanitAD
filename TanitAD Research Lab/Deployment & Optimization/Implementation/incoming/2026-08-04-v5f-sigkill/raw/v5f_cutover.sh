#!/bin/bash
# v5f cutover: wait for the step-4000 checkpoint, then hand the run to the
# supervisor, which relaunches it from the UPDATED manifest
# (--batch 8 --accum 8 --v2-lru 64 --workers 8).
#
# eff_batch is 8*8 = 64, identical to the outgoing 4*16, so this is a DATA-PATH
# change and not a new experiment: the optimisation regime is untouched.
#
# WHY WAIT: measured 2026-08-03, killing at step 3950 would discard 200 steps
# (59.6 min at 17.89 s/step); the next save is 50 steps away (14.9 min).
PID=19412
CK=/workspace/experiments/flagship-v5f-w120-30k/ckpt.pt

log() { echo "[$(date -u +%FT%TZ)] $*"; }

log "watcher up; waiting for ckpt step >= 4000 (trainer pid $PID)"
for i in $(seq 1 200); do
  if ! kill -0 "$PID" 2>/dev/null; then
    log "trainer $PID already gone - supervisor owns it now"
    exit 0
  fi
  S=$(python3 -c "import torch;print(torch.load('$CK',map_location='cpu',weights_only=False).get('step',-1))" 2>/dev/null)
  log "poll $i: ckpt step=$S"
  if [ -n "$S" ] && [ "$S" -ge 4000 ] 2>/dev/null; then
    # Kill by EXPLICIT PID. Never `pkill -f <trainer>`: the pattern matches this
    # very script's own command line and would kill the watcher instead, with
    # empty output, looking exactly like nothing happened.
    log "checkpoint $S banked -> killing explicit PID $PID"
    kill "$PID"
    for _ in $(seq 1 60); do
      kill -0 "$PID" 2>/dev/null || break
      sleep 2
    done
    if kill -0 "$PID" 2>/dev/null; then
      log "still alive after 120 s - escalating to SIGKILL"
      kill -9 "$PID"
    fi
    log "trainer down at ckpt step $S; supervisor relaunches from the updated manifest"
    exit 0
  fi
  sleep 30
done
log "TIMEOUT: never saw step 4000 in ~100 min - deliberately NOT killing anything"
exit 1

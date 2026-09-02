#!/bin/bash
# Supervisor for the REF-C v3 H-arm 40,284-step run.
#
# WHY THIS EXISTS — MEASURED 2026-09-01: the run stopped at step 2000 with a
# CLEAN log (no traceback), no supervisor, and an empty container dmesg, so the
# cause could be neither confirmed nor excluded. It sat dead ~26 min before a
# routine probe noticed. An unsupervised 51 h run can lose a whole night to a
# cause nobody can even diagnose afterwards.
#
# THREE SUPERVISOR TRAPS THIS AVOIDS (all measured in this programme):
#  1. "SOURCES ITS MANIFEST ONCE" — a supervisor that captures TRAIN_CMD at boot
#     replays a stale command forever. Here the command is EMBEDDED, so what you
#     read is what runs; to change it you edit this file and restart.
#  2. "RESURRECTS A FINISHED RUN" — a supervisor keyed on a done-MARKER relaunches
#     forever when the trainer never writes one. Here the done condition is
#     DERIVED FROM THE DATA: the last step in metrics.jsonl reaching TARGET.
#  3. "RACES ITS OWN flock" — a restart before the old holder exits dies silently
#     with nothing running. flock -n makes that loud instead.
#
# It also caps relaunches, so a crash-on-start loop stops instead of spinning.
set -uo pipefail

OUT=/workspace/experiments/refcv3-b1-v72-30k
STACK=/workspace/TanitAD/stack
TARGET=40284
MAX_RELAUNCH=80
POLL=120
LOCK=/workspace/.sup_refcv3b.lock
SLOG=$OUT/supervisor.log

exec 200>"$LOCK"
if ! flock -n 200; then
  echo "REFUSING: another supervisor holds $LOCK — not starting a second" >&2
  exit 3
fi

log() { echo "[sup $(date -u +%FT%TZ)] $*" >> "$SLOG"; }

last_step() {
  # the run's OWN data is the progress truth, not a marker and not liveness
  tail -n 200 "$OUT/metrics.jsonl" 2>/dev/null | python3 -c '
import sys, json
s = 0
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        s = max(s, int(json.loads(line).get("step", 0)))
    except Exception:
        pass
print(s)
' 2>/dev/null || echo 0
}

trainer_pid() {
  # character-class regex so grep cannot match its OWN argv (measured trap:
  # a filter containing the pattern it searches for reports a phantom result)
  ps -eo pid=,args= | grep 'refc[_]v3[_]train' | grep -v ' grep ' | awk '{print $1}' | head -1
}

launch() {
  cd "$OUT" || return 1
  PYTHONPATH="$STACK" nohup python3 -u "$STACK/scripts/refc_v3_train.py" \
    --arm hier --size base \
    --v2-cache /root/data/train \
    --v7-labels /workspace/TanitAD/data/s2_labels_v7.2_train.jsonl.gz \
    --eval-cache /root/data/eval \
    --eval-labels /workspace/TanitAD/data/s2_labels_v7.2_eval.jsonl.gz \
    --eval-every 500 --eval-batches 8 \
    --image-hw 256 640 \
    --steps "$TARGET" --batch 20 --workers 6 --prefetch-factor 1 --v2-lru 24 \
    --lr 1e-4 --warmup 2000 --seed 0 \
    --log-every 50 --save-every 500 \
    --out "$OUT" >> "$OUT/train.log" 2>> "$OUT/train.stderr.log" 200>&- &
  echo $!
}

log "supervisor up; target $TARGET; resume point step $(last_step)"
n=0
while true; do
  st=$(last_step)
  if [ "${st:-0}" -ge "$TARGET" ]; then
    log "DONE: metrics.jsonl reports step $st >= $TARGET — exiting cleanly"
    # the done-marker is written HERE, in the same breath as the decision, so a
    # future supervisor cannot resurrect a finished run
    echo "{\"done\": true, \"final_step\": $st, \"target\": $TARGET}" > "$OUT/summary.json"
    exit 0
  fi
  pid=$(trainer_pid)
  if [ -z "$pid" ]; then
    if [ "$n" -ge "$MAX_RELAUNCH" ]; then
      log "REFUSING further relaunches: $n reached MAX_RELAUNCH at step $st"
      exit 4
    fi
    n=$((n + 1))
    newpid=$(launch)
    log "relaunch #$n from step $st -> pid $newpid"
    sleep 90     # let it get past import/dataset build before the next poll
  else
    sleep "$POLL"
  fi
done

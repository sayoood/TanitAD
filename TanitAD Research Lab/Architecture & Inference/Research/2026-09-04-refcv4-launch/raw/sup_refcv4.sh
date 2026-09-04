#!/bin/bash
# Supervisor for the REF-C v4 H-arm 40,284-step run (refcv4-b1-v72-40k).
#
# Adapted from sup_refcv3_v3.sh, which already carries the three supervisor
# traps this programme has measured, all of which are KEPT here verbatim:
#  1. "SOURCES ITS MANIFEST ONCE" — the command is EMBEDDED, so what you read
#     is what runs; to change it, stop this supervisor and edit this file.
#  2. "RESURRECTS A FINISHED RUN" — the done condition is DERIVED FROM THE DATA
#     (last step in metrics.jsonl >= TARGET), and the done-marker is written in
#     the same breath as the decision.
#  3. "RACES / INHERITS ITS OWN flock" — `200>&-` on the trainer AND on every
#     `sleep`, because EVERY child inherits the lock fd, not just the
#     interesting one (measured 2026-09-02: a `sleep 180` held the lock after
#     its parent died and made the run permanently unsupervisable).
#
# WHAT IS NEW IN v4 vs the refcv3 command this is otherwise a copy of:
#   --anchors            ⛔ BLOCKER 1. refcv3 passed NONE and silently carried
#                        refc.default_anchors (SYNTHETIC). Oracle-in-vocabulary
#                        ADE 0-2 s on 19,602 held-out eval windows: synthetic
#                        1.0882 m, a single straight line 0.6843 m, this file
#                        0.3796 m.
#   --sel-accel-max 2.0  ⛔ BLOCKER 2. `horizon_s` is DERIVED from max(horizons)
#                        = 6.0 s, so the inherited 2.5 opened the reach band to
#                        +-15.0 m/s (kills 18.02 %). 2.0 gives +-12.0 m/s,
#                        kills 26.23 %, deletes 0.000 % of eval / 0.007 % of
#                        train GT, dADE +0.00000 m.
#   --goal-str           refcv3's strategic head was NEVER supervised
#                        (`goal_str` in 0 of 614 logged rows). ⛔ NOT
#                        --graft-lan: that is the supplied corridor as a MODEL
#                        INPUT and is refused by E12 / the vision-only rule.
#                        --goal-str alone mints the LABEL (want_lan) without
#                        building the input pathway.
#   --ego-state-inject --ego-dropout 0.5   the registered v4 lever (E11' + X15).
#                        ⛔ NO --echo-base: the tiny rig MEASURED that it moved
#                        the verdict from READS_BOTH to ECHOING, so E14 is the
#                        SECOND arm, not this one.
# Everything else is the incumbent's own command, byte for byte, so
# refcv4-vs-refcv3 stays a lever set and not a bundle.
set -uo pipefail

OUT=/workspace/experiments/refcv4-b1-v72-40k
STACK=/workspace/TanitAD/stack
TARGET=40284
MAX_RELAUNCH=80
POLL=120
LOCK=/workspace/.sup_refcv4.lock
SLOG=$OUT/supervisor.log

mkdir -p "$OUT"
exec 200>"$LOCK"
if ! flock -n 200; then
  echo "REFUSING: another supervisor holds $LOCK — not starting a second" >&2
  exit 3
fi

log() { echo "[sup $(date -u +%FT%TZ)] $*" >> "$SLOG"; }

last_step() {
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
  # character-class regex so grep cannot match its OWN argv (measured trap: a
  # filter containing the pattern it searches for reports a phantom result),
  # and keyed on THIS run's out-dir so a sibling refc_v3_train is not adopted.
  ps -eo pid=,args= | grep 'refcv4[-]b1-v72-40k' | grep 'refc[_]v3[_]train' \
    | grep -v ' grep ' | awk '{print $1}' | head -1
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
    --nav-from-v7 \
    --u8-batches \
    --anchors "$OUT/anchors.pt" \
    --sel-accel-max 2.0 \
    --goal-str \
    --ego-state-inject --ego-dropout 0.5 \
    --out "$OUT" >> "$OUT/train.log" 2>> "$OUT/train.stderr.log" 200>&- &
  echo $!
}

# ⛔ REFUSE TO LAUNCH WITHOUT THE VOCABULARY. A missing --anchors file is
# exactly how refcv3 spent 53 h on the synthetic set: the trainer would not
# fail, it would fall back. This makes the fallback impossible here.
if [ ! -s "$OUT/anchors.pt" ]; then
  echo "REFUSING: $OUT/anchors.pt missing — that is the refcv3 defect" >&2
  exit 5
fi

log "supervisor refcv4 up (anchors=$OUT/anchors.pt, sel-accel-max 2.0, --goal-str, ego-inject dropout 0.5, NO --echo-base); target $TARGET; resume point $(last_step)"
n=0
while true; do
  st=$(last_step)
  if [ "${st:-0}" -ge "$TARGET" ]; then
    log "DONE: metrics.jsonl reports step $st >= $TARGET — exiting cleanly"
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
    sleep 90 200>&-
  else
    sleep "$POLL" 200>&-
  fi
done

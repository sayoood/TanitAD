#!/bin/bash
# WP-D REPLICATE chain supervisor -- the arms that make A3 readable.
#
#   D0b : D0's flags, D0's SEED (0), run again, ZERO levers moved.
#         = the `A0b_replicate` design (CLAUDE.md H-ESTIM-SEED-1). Measures the
#           rig's INFERENCE/NONDETERMINISM run-to-run floor.
#   D0c : D0's flags with --seed 1. = PREREG_WPD_BEV_AUX.md section 4's own D0b row.
#         Measures the STRICTLY LARGER floor that also contains init/shuffle variance.
#
# A3's denominator is the LARGER of the two once both land: a same-seed-only floor
# is anti-conservative (it excludes seed variance) and would make the 3x bar easier.
#
# LOCK-FD DISCIPLINE: every child gets `200>&-`. A child that inherits fd 200 holds
# the flock for its whole life and NO replacement supervisor can ever start. That
# includes the `sleep`s, not just the trainer (MEASURED 2026-09-02: the holder was a
# `sleep 180`, an already-dead supervisor's poll child).
# refc_v3_train.py has NO --resume, so a relaunch restarts the arm at step 0 --
# hence MAX_RELAUNCH is small, deliberately.
set -u
STACK=/home/nvidia/TanitAD/stack
PY=/home/nvidia/venvs/tanitad-train/bin/python
STEPS=4000
MAX_RELAUNCH=2
POLL=180
LOCK=/home/nvidia/.sup_wpd_rep.lock
SLOG=/home/nvidia/experiments/wpd_rep_supervisor.log
mkdir -p /home/nvidia/experiments
exec 200>"$LOCK"
if ! flock -n 200; then echo "REFUSING: another supervisor holds $LOCK" >&2; exit 3; fi
log() { echo "[suprep $(date -u +%FT%TZ)] $*" >> "$SLOG"; }

# EXACT copy of sup_wpd.sh's BASE_ARGS, with --seed made per-arm.
# Verified against wpd-D0-4k/config.json['argv'] by argv_audit.py (post-launch, on
# the ARTIFACT, not on this text).
base_args() {
  local seed="$1"
  echo "--arm hier --size base --image-hw 256 640" \
       "--v2-cache /home/nvidia/data/physicalai-b1-w120-256x640cyl" \
       "--v7-labels /home/nvidia/data/v72/labels/s2_labels_v7.2_train.jsonl.gz" \
       "--batch 20 --workers 6 --prefetch-factor 1 --v2-lru 24" \
       "--lr 1e-4 --warmup 2000 --log-every 50 --save-every 1000 --u8-batches" \
       "--nav-from-v7 --ego-state-inject --ego-dropout 0.5" \
       "--anchors /home/nvidia/data/anchors/refc_anchors_6s_v0cond_alat_117.pt" \
       "--n-anchors 117 --anchor-v0-conditioned --anchor-control-units alat" \
       "--sel-accel-max 2.0" \
       "--sampler ddim --w-u0 0.5 --sel-refined --sel-score-emitted" \
       "--goal-str --tac-goal-tok-head --agents off" \
       "--agent-join /home/nvidia/percprobe/raw/b1train_agents.jsonl.xz" \
       "--steps $STEPS --seed $seed"
}
arm_seed() { case "$1" in D0b) echo 0 ;; D0c) echo 1 ;; esac; }

is_done() {  # the TRAINER writes {"done": true} itself -- verified in the WP-D smoke
  local f="$1/summary.json"
  [ -s "$f" ] && grep -q '"done": true' "$f"
}
trainer_pid() {  # [-] so the pattern cannot match this command's own text
  ps -eo pid=,args= | grep "wpd[-]$1[-]4k" | grep -v ' grep ' | awk '{print $1}' | head -1
}
launch() {
  local arm="$1" out="$2"
  mkdir -p "$out"
  cd "$out" || return 1
  PYTHONPATH="$STACK" PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=6 \
  nohup "$PY" -u "$STACK/scripts/refc_v3_train.py" \
    $(base_args "$(arm_seed "$arm")") --out "$out" \
    >> "$out/train.log" 2>> "$out/train.stderr.log" 200>&- &
  echo $!
}

log "WP-D REPLICATE supervisor up: D0b(seed 0) -> D0c(seed 1), $STEPS steps each"
for ARM in D0b D0c; do
  OUT=/home/nvidia/experiments/wpd-$ARM-4k
  if is_done "$OUT"; then log "$ARM already done, skipping"; continue; fi
  n=0
  while true; do
    if is_done "$OUT"; then log "$ARM DONE"; break; fi
    pid=$(trainer_pid "$ARM")
    if [ -z "$pid" ]; then
      if [ "$n" -ge "$MAX_RELAUNCH" ]; then
        log "$ARM REFUSING further relaunches after $n attempts -- moving on"; break
      fi
      n=$((n + 1))
      newpid=$(launch "$ARM" "$OUT")
      log "$ARM launch #$n -> pid $newpid seed=$(arm_seed "$ARM")"
      sleep 240 200>&-
    else
      sleep "$POLL" 200>&-
    fi
  done
done
log "WP-D REPLICATE chain COMPLETE"

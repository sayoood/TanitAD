#!/bin/bash
# WP-C ORACLE GATE runner -- PREREG_WPB_WAYPOINT_INDEX.md section 7.2.
#
# Runs the six 6,000-step arms SEQUENTIALLY on one A40. Resumable at ARM
# granularity: an arm whose out-dir carries the TRAINER'S OWN summary.json with
# {"done": true, "step": >= 6000} is skipped, so a supervisor restart never
# re-burns a finished arm. `refc_v3_train.py` has no --resume, so a crash costs
# that ARM only.
#
# ⛔ COMPLETION IS ASSERTED ON THE ARTIFACT, NEVER ON THE EXIT CODE. The exit
# status of a trainer reaches this script through a shell that can overwrite it;
# `summary.json`'s recorded `step` is the thing we actually wanted.
#
# Lock-fd hygiene: spawned with 200>&- and every child it spawns inherits a
# CLOSED fd 200 -- the supervisor's flock must never be held by a trainer or by
# a sleep. (2026-09-02: a `sleep 180` held fd 200 and made a run permanently
# unsupervisable; patching only the trainer was not enough.)
set -u

ROOT=/workspace/experiments/wpc-oracle-gate
STACK=/workspace/TanitAD/stack
JOIN=/workspace/joins/b1_train_plus_eval_agents.jsonl.xz
ANCHORS=$ROOT/anchors.pt
TRAIN=$STACK/scripts/refc_v3_train.py
TARGET_STEPS=6000

export PYTHONPATH=$STACK
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=6
# MEASURED in the 20-step smoke: both arms peak at 43.4-44.7 GiB of the A40's
# 46.07 GiB (94-97 %). Identical for index-ON and index-OFF, so the WP-B bias is
# NOT the cause -- the base refcv5-v2 recipe plus the oracle seam is. This is an
# ALLOCATOR setting, applied identically to every arm, so it cannot enter the
# B1-B0 contrast; it only reduces fragmentation in a tight envelope.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$ROOT"

# Reads the TRAINER's own summary.json and echoes the step it recorded, or -1.
arm_done_step() {
  python3 - "$1" 200>&- <<'PY'
import json, os, sys
p = os.path.join(sys.argv[1], "summary.json")
try:
    with open(p, "r", encoding="utf-8") as f:
        r = json.load(f)
    print(int(r.get("step", -1)) if r.get("done") is True else -1)
except Exception:
    print(-1)
PY
}

# --- the base flag set, verbatim from refcv5-v2's own config.json `argv` ------
# (read back from that run, not retyped from prose). Only --steps, --seed,
# --out, --anchors and the agent / wp-index block differ from that record.
base_args() {
  echo --arm hier --size base \
    --v2-cache /root/data/train \
    --v7-labels /workspace/TanitAD/data/s2_labels_v7.2_train.jsonl.gz \
    --eval-cache /root/data/eval \
    --eval-labels /workspace/TanitAD/data/s2_labels_v7.2_eval.jsonl.gz \
    --eval-every 500 --eval-batches 8 \
    --image-hw 256 640 \
    --steps $TARGET_STEPS --batch 20 --workers 6 --prefetch-factor 1 --v2-lru 24 \
    --lr 1e-4 --warmup 2000 \
    --log-every 50 --save-every 500 \
    --u8-batches \
    --nav-from-v7 \
    --ego-state-inject --ego-dropout 0.5 \
    --anchors "$ANCHORS" \
    --n-anchors 117 --anchor-v0-conditioned --anchor-control-units alat \
    --sel-accel-max 2.0 \
    --sampler ddim --w-u0 0.5 \
    --sel-refined --sel-score-emitted \
    --goal-str \
    --tac-goal-tok-head \
    --agents oracle --agent-join "$JOIN" --w-agent 0
}

# arm | seed | wp-index block
# Ordered so a killed chain still yields value: the effect (B0,B1), then the
# replicate FLOOR without which neither of those is readable, then the controls.
ARMS=(
  "B0|0|--wp-index off"
  "B1|0|--wp-index on"
  "B0r|1|--wp-index off"
  "B1r|1|--wp-index on"
  "B1const|0|--wp-index on --wp-index-mode const"
  "B1shuf|0|--wp-index on --wp-index-mode shuffle"
)

for spec in "${ARMS[@]}"; do
  name="${spec%%|*}"; rest="${spec#*|}"
  seed="${rest%%|*}"; wp="${rest#*|}"
  out="$ROOT/$name"
  mkdir -p "$out"

  step=$(arm_done_step "$out")
  if [ "${step:--1}" -ge "$TARGET_STEPS" ]; then
    echo "ZZARM $name ALREADY_DONE step=$step ZZ"
    continue
  fi

  echo "ZZARM $name START $(date -u +%FT%TZ) seed=$seed wp=[$wp] ZZ"
  # shellcheck disable=SC2046
  python3 "$TRAIN" $(base_args) --seed "$seed" --out "$out" $wp \
      >> "$out/train.log" 2>> "$out/train.stderr.log" 200>&-
  rc=$?

  step=$(arm_done_step "$out")
  echo "ZZARM $name EXIT rc=$rc recorded_step=$step $(date -u +%FT%TZ) ZZ"
  if [ "${step:--1}" -ge "$TARGET_STEPS" ]; then
    echo "ZZARM $name DONE ZZ"
  else
    echo "ZZARM $name INCOMPLETE recorded_step=$step -- supervisor will relaunch ZZ"
    exit 17
  fi
done

echo "ZZGATE ALL_ARMS_DONE $(date -u +%FT%TZ) ZZ"
printf '{"done": true, "gate": "wpc-oracle-gate", "arms": 6, "steps": %s}\n' \
  "$TARGET_STEPS" > "$ROOT/summary.json"

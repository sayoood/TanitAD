#!/usr/bin/env bash
# STREAM A (re-issue): the close-following / cut-in test on REAL scene geometry.
#
# WHY THIS EXISTS — and what it corrects.
# The first pass concluded "no close-following / cut-in geometry exists in the material"
# from TWO probes (sequence_tracks.json + clipgt/obstacle.parquet) — but BOTH probes read
# the SAME scene, 00040136. There are TWO NuRec scenes on Thor. The second one,
# 7c72937c-c620-4776-9555-d57222c0081f, was never probed, and it HAS the geometry:
#     min in-lane headway   3.268 m   (scene 1: 46.246 m)
#     close-following rows       146   (scene 1: 0)
#     cut-in events               44   (scene 1: 0)
# both probes agreeing (obstacle.parquet: 3.234 m / 146 rows). So the constructed
# synth_actor lead was never necessary — this panel runs the REAL thing.
#
# A SECOND defect had to be fixed before the actors were visible at all: actor_map's
# relative-margin rule rejected 31 of 92 cuboids that matched their track at cost EXACTLY
# 0 us, including BOTH vehicles the ego follows and ALL THREE cut-in tracks. The pixel
# falsifier adjudicated the corrected rule: mean grad-NCC on = 0.5330 (92 actors) vs
# 0.4340 (61), with the wrong-time negative control at 0.4216 — same gaussians, wrong
# placement, far worse. See actor_map.py's header.
#
# CONDITIONS
#   empty    the control — background+road only, nothing dynamic drawn.
#   objects  the scene's OWN annotated agents, placed and falsified against the
#            reference video. This is the real close-following / cut-in exposure.
# The contrast is objects-vs-empty PER ARM (does the arm react to real close traffic?)
# and flagship-vs-refc within each condition (which arm handles it better?), both paired
# on identical (start, k) windows.
#
# Starts 0..120 x 50 steps keeps every window inside the 200-tick clip.
set -u
export PATH=$HOME/venvs/tanitad-edge/bin:/usr/local/cuda/bin:$PATH
export OMP_NUM_THREADS=6
export PYTHONPATH=$HOME/tanitad_cl/stack:$HOME/tanitad_cl/stack/scripts:$HOME/tanitad_cl/taniteval:$HOME/tanitad_cl/stack/experiments/nurec-gsplat:$HOME/tanitad_cl/stack/experiments/alpasim-gsplat:$HOME/alpasim/src/grpc
cd "$HOME/tanitad_cl/stack/experiments/alpasim-gsplat"

SCENE=$HOME/nurec_scenes/sample_set/26.04_release/7c72937c-c620-4776-9555-d57222c0081f
OUT=${1:-$HOME/scene2_out}
STARTS=${2:-0,15,30,45,60,75,90,105,120}
STEPS=${3:-50}
mkdir -p "$OUT"

declare -A CKPT=(
  [flagship-v1]=$HOME/models/flagship-v1-speedjerk/ckpt.pt
  [refc-base]=$HOME/models/refc-base/ckpt.pt
)

for ARM in flagship-v1 refc-base; do
  for COND in empty objects; do
    D="$OUT/${ARM}_${COND}"
    if [ -s "$D/rollouts_${ARM}_${COND}.json" ]; then
      echo "=== SKIP $ARM/$COND (already present) ==="; continue
    fi
    echo "=== $ARM / $COND  starts=$STARTS steps=$STEPS ==="
    python closedloop_drive.py --scene-dir "$SCENE" --arm "$ARM" --ckpt "${CKPT[$ARM]}" \
        --condition "$COND" --starts "$STARTS" --steps "$STEPS" \
        --out "$D" || echo "FAILED $ARM $COND"
  done
done

# one long rollout per arm WITH frames, for the video deliverable
for ARM in flagship-v1 refc-base; do
  D="$OUT/long_${ARM}_objects"
  [ -s "$D/rollouts_${ARM}_objects.json" ] && continue
  python closedloop_drive.py --scene-dir "$SCENE" --arm "$ARM" --ckpt "${CKPT[$ARM]}" \
      --condition objects --starts 0 --steps 150 --out "$D" --save-video-frames \
      || echo "FAILED long $ARM objects"
done
echo "SCENE2_PANEL_DONE"

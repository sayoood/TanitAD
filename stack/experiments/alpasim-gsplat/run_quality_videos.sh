#!/usr/bin/env bash
# The four closed-loop videos, re-rendered with the 2026-08-03 RENDER-QUALITY config.
#
# WHAT CHANGED vs the 2026-08-03-morning videos in ~/cl_videos, and why (all MEASURED on
# scene 00040136, 5 frames, run dir /home/nvidia/rq_out/panel4_final + panel6_chosen;
# metric is gradient-NCC because PSNR/NCC/MAE all fail the negative control on this clip):
#
#   1. ALL FOUR LAYERS.  Previously `background,road` only — `dynamic_rigids` (115,824
#      gaussians / 30 cuboids) and `dynamic_deformables` (1,039 / 2) were absent from
#      every frame. Mapping is exact (35/35 and 2/2 at best_cost_us == 0) and the
#      wrong-time negative control separates by +0.2358 grad-NCC.
#   2. SCALE CULL 0.95.  Drops the 153,506 static splats whose largest axis exceeds
#      1.4263 m. These are the long horizontal light STREAKS (the "magenta smear" and
#      its cyan companion). grad-NCC 0.2773 -> 0.3460.
#   3. GATED SKY, gain 0.3.  Fills the black upper band, which is a reconstruction hole
#      (NOT an FOV clip: zero pixels exceed the f-theta max_angle of 77.22 deg). The
#      reference shows 0.1439 mean brightness where our alpha < 0.1 and we rendered
#      0.0176. Gain 1.0 over-brightens; 0.3 is the measured knee.
#
#   Combined (arm G / AFTER): grad-NCC 0.2774 -> 0.3424 (+23.4 %), negative-control
#   margin 0.0873 -> 0.1020, at 34.6 ms/frame vs 23.4 — still ~29 FPS, inside the loop.
#
# ⛔ ROLLING SHUTTER IS DELIBERATELY OFF HERE. It is the largest single lever
# (grad-NCC 0.2774 -> 0.3170 on its own, best negative-control margin of any arm) but it
# costs ~3700 ms/frame vs 23 — 161x. Pass ROLL=1 to enable it; expect ~45 min of render.
set -u
export PATH=$HOME/venvs/tanitad-edge/bin:/usr/local/cuda/bin:$PATH
export OMP_NUM_THREADS=6
export PYTHONPATH=$HOME/tanitad_cl/stack:$HOME/tanitad_cl/stack/scripts:$HOME/tanitad_cl/taniteval:$HOME/tanitad_cl/stack/experiments/nurec-gsplat:$HOME/tanitad_cl/stack/experiments/alpasim-gsplat
cd "$HOME/tanitad_cl/stack/experiments/alpasim-gsplat"

SCENE=$HOME/nurec_scenes/sample_set/26.04_release/00040136-e651-4abd-991d-0655ccda9430
OUT=${1:-$HOME/cl_out_hq}
VID=${2:-$HOME/cl_videos_hq}
STEPS=${3:-180}
ROLL=${ROLL:-0}
mkdir -p "$OUT" "$VID"

RQ="--all-dynamic-layers --cull-scale-quantile 0.95 --sky-gain 0.3"
[ "$ROLL" = "1" ] && RQ="$RQ --rolling-shutter"
echo "RENDER-QUALITY FLAGS: $RQ"

declare -A CKPT=(
  [flagship-v1]=$HOME/models/flagship-v1-speedjerk/ckpt.pt
  [refc-base]=$HOME/models/refc-base/ckpt.pt
)

for ARM in flagship-v1 refc-base; do
  for COND in objects empty; do
    D="$OUT/long_${ARM}_${COND}"
    echo "=== ROLLOUT $ARM / $COND ($STEPS steps) ==="
    # `objects` needs the dynamic layers in the scene; `empty` is the matched control
    # and must NOT get them, so the pair stays a clean A/B on actor presence.
    EXTRA="$RQ"
    [ "$COND" = "empty" ] && EXTRA="--cull-scale-quantile 0.95 --sky-gain 0.3"
    python closedloop_drive.py --scene-dir "$SCENE" --arm "$ARM" --ckpt "${CKPT[$ARM]}" \
        --condition "$COND" --starts 0 --steps "$STEPS" \
        --out "$D" --save-video-frames $EXTRA || { echo "FAILED rollout $ARM $COND"; continue; }
  done
done

for ARM in flagship-v1 refc-base; do
  for COND in objects empty; do
    D="$OUT/long_${ARM}_${COND}"
    R="$D/rollouts_${ARM}_${COND}.json"
    F="$D/frames_s0"
    [ -s "$R" ] && [ -d "$F" ] || { echo "SKIP video $ARM/$COND"; continue; }
    case "$COND" in
      objects) NAME="${ARM}_with_objects" ;;
      empty)   NAME="${ARM}_empty_road" ;;
    esac
    echo "=== VIDEO $ARM / $COND -> $NAME.mp4 ==="
    python overlay_video.py --rollouts "$R" --frames "$F" --scene-dir "$SCENE" \
        --tracks "$SCENE/extracted/sequence_tracks.json" \
        --out "$VID/${NAME}.mp4" || echo "FAILED video $ARM $COND"
  done
done
ls -la "$VID"
echo "HQ_VIDEOS_DONE"
